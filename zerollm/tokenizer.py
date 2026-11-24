"""分词器:在训练好的 BPE 词表上做 encode / 流式 encode_iterable / decode。

设计对齐 CS336 assignment1 的分词器规范:
- special tokens 先于普通文本切分,整体成 token、不参与 merge(按最长前缀优先匹配);
- 普通文本按 GPT-2 正则预分词,每个预 token 内按 merge 的创建顺序(rank)贪心合并;
- decode 先拼 bytes 再统一 UTF-8 解码,避免多字节字符被 token 边界切断;
- encode_iterable 流式处理大文件:只在"安全切点"(完整预 token / 完整 special token 之后)吐出 id,
  尾部不完整的片段留在缓冲区,内存不随文件大小增长。

文件格式(本仓库自定,便于自洽校验):
- vocab.json: {"<id>": "<latin-1 字符串>"} ,bytes 与 U+0000..U+00FF 一一对应;
- merges.txt: 每行 "<left_id> <right_id>",按 merge 顺序;
- special tokens 不出现在 merges.txt 中,由调用方单独传入(并写入 vocab.json)。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator

import regex as re

DEFAULT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
        pattern: str = DEFAULT_PATTERN,
    ):
        self.pattern = pattern
        self._pat = re.compile(pattern)
        self.special_tokens = list(special_tokens or [])
        # special token 的 id 只增不改:先取已有 vocab 的最大 id 之后
        self.vocab = dict(vocab)
        # 按 id 排序建反查,保证同一 bytes 对应最小 id(重复 bytes 时行为确定)
        self._bytes_to_id = {b: i for i, b in sorted(self.vocab.items())}
        next_id = max(self.vocab) + 1 if self.vocab else 0
        for s in self.special_tokens:
            b = s.encode("utf-8")
            if b not in self._bytes_to_id:
                self.vocab[next_id] = b
                self._bytes_to_id[b] = next_id
                next_id += 1
        self.merges = list(merges)
        # merge rank:越早的 pair rank 越小,encode 时先合并
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        # 预 token -> id 列表 的缓存(中文语料高度重复,缓存收益很大)
        # 上限防止在超大规模语料上无限增长(多进程编码时每个 worker 各有一份)
        self._cache: dict[bytes, tuple[int, ...]] = {}
        self._cache_limit = 2_000_000

        # special token 正则:按长度倒序,保证最长匹配优先
        self._special_len = max((len(s) for s in self.special_tokens), default=0)
        if self.special_tokens:
            ordered = sorted(self.special_tokens, key=len, reverse=True)
            self._special_rx = re.compile("|".join(re.escape(s) for s in ordered))
        else:
            self._special_rx = None

    # ---------- 构造 / 存取 ----------

    @classmethod
    def from_files(
        cls,
        vocab_path: str,
        merges_path: str,
        special_tokens: list[str] | None = None,
        pattern: str = DEFAULT_PATTERN,
    ) -> "Tokenizer":
        with open(vocab_path, encoding="utf-8") as f:
            raw = json.load(f)
        vocab = {int(k): v.encode("latin-1") for k, v in raw.items()}
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                a, b = line.split()
                merges.append((vocab[int(a)], vocab[int(b)]))
        return cls(vocab, merges, special_tokens=special_tokens, pattern=pattern)

    def save(self, vocab_path: str, merges_path: str) -> None:
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(
                {str(i): b.decode("latin-1") for i, b in sorted(self.vocab.items())},
                f,
                ensure_ascii=True,
            )
        id_of = self._bytes_to_id
        with open(merges_path, "w", encoding="utf-8") as f:
            f.write("#version: zerollm-bpe\n")
            for a, b in self.merges:
                f.write(f"{id_of[a]} {id_of[b]}\n")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __len__(self) -> int:
        return len(self.vocab)

    # ---------- 编码 ----------

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece, is_special in self._split_special(text):
            if is_special:
                ids.append(self._bytes_to_id[piece.encode("utf-8")])
            else:
                ids.extend(self._encode_ordinary(piece))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """流式编码任意大小的文本块序列,内存占用与文件大小无关。"""
        buf = ""
        for chunk in iterable:
            buf += chunk
            cut = self._safe_cut(buf)
            if cut is None:
                if len(buf) < (1 << 20):
                    continue
                cut = len(buf)  # 兜底:1MB 内找不到安全切点,强制切(极端输入)
            if cut <= 0:
                continue
            yield from self.encode(buf[:cut])
            buf = buf[cut:]
        if buf:
            yield from self.encode(buf)

    def _encode_ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for tok in self._pat.findall(text):
            b = tok.encode("utf-8")
            cached = self._cache.get(b)
            if cached is None:
                cached = tuple(self._encode_pretoken(b))
                if len(self._cache) >= self._cache_limit:
                    self._cache.clear()
                self._cache[b] = cached
            ids.extend(cached)
        return ids

    def _encode_pretoken(self, b: bytes) -> list[int]:
        """单个预 token 内:反复合并当前 rank 最小的相邻 pair,直到无可合并。"""
        parts = [bytes([x]) for x in b]
        while len(parts) >= 2:
            best = None
            best_rank = None
            for pair in zip(parts, parts[1:]):
                r = self.merge_ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = pair, r
            if best is None:
                break
            merged = []
            i = 0
            while i < len(parts):
                if i + 1 < len(parts) and (parts[i], parts[i + 1]) == best:
                    merged.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        return [self._bytes_to_id[p] for p in parts]

    def _split_special(self, text: str) -> list[tuple[str, bool]]:
        """把文本切成 (片段, 是否 special) 序列;special 整体保留。"""
        if not self._special_rx:
            return [(text, False)] if text else []
        out: list[tuple[str, bool]] = []
        pos = 0
        for m in self._special_rx.finditer(text):
            if m.start() > pos:
                out.append((text[pos : m.start()], False))
            out.append((m.group(0), True))
            pos = m.end()
        if pos < len(text):
            out.append((text[pos:], False))
        return out

    # ---------- 流式安全切点 ----------

    def _safe_cut(self, buf: str) -> int | None:
        """返回可安全编码的前缀长度;None 表示需要继续缓冲。

        规则:
        a) 若 buf 末尾正好以一个完整的 special token 结尾,可以把它一起编出
           (special 不会向右延续);
        b) 若 buf 末尾是某个 special token 的**真前缀**,退到它之前;
        c) 普通文本退到最后一个完整预 token 的结尾(最后一个匹配可能延续)。
        """
        ends = [m.end() for m in self._pat.finditer(buf)]

        # b) 末尾是 special 的真前缀 → 不能越过该前缀起点
        hold_start = None
        if self.special_tokens:
            for i in range(max(0, len(buf) - self._special_len + 1), len(buf)):
                suffix = buf[i:]
                if any(s.startswith(suffix) for s in self.special_tokens):
                    hold_start = i
                    break

        if hold_start is not None:
            limited = [e for e in ends if e <= hold_start]
            cut = limited[-1] if limited else 0
        elif self._special_rx and self._special_rx.fullmatch(buf) is None:
            # a) 末尾正好是完整 special:看最后一个匹配是不是这个 special 本身
            last_special = None
            for m in self._special_rx.finditer(buf):
                if m.end() == len(buf):
                    last_special = m
                    break
            if last_special is not None:
                cut = len(buf)
            else:
                cut = ends[-2] if len(ends) >= 2 else 0
        elif self._special_rx:
            cut = len(buf)  # 整个 buf 就是一个完整 special token
        else:
            cut = ends[-2] if len(ends) >= 2 else 0

        if cut <= 0:
            return None
        return cut

    # ---------- 解码 ----------

    def decode(self, ids: Iterable[int]) -> str:
        data = b"".join(self.vocab[int(i)] for i in ids)
        return data.decode("utf-8", errors="replace")