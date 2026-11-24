"""BPE 训练(改编自本人 cs336 assignment1-basics 已通过官方测试的实现)。

与课程版的差异:
- 对外暴露 `train_bpe(input_path, vocab_size, special_tokens, pattern=...)` 纯函数接口,
  返回 (vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]),与 CS336 adapter 一致。
- 训练数据流式读取(逐块读文件 + Counter 合并),支持 GB 级语料,不整文件进内存。
- 中文语料沿用 GPT-2 预分词正则:` ?\p{L}+` 对汉字连续串天然有效,空白/标点单独切分。
"""

import regex as re
from collections import Counter

# GPT-2 风格预分词:英文缩写 / 字母串 / 数字串 / 标点 / 尾随空格处理
# r""" 需要显式写;` ?\p{L}+` 对汉字连续串天然有效
gpt2pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

CHUNK_SIZE = 16 * 1024 * 1024  # 流式读取块大小:16MB


def get_stats(word, freq, stats=None):
    stats = {} if stats is None else stats
    for pair in zip(word, word[1:]):
        stats[pair] = stats.get(pair, 0) + freq
    return stats


def get_words(word, words):
    for pair in zip(word, word[1:]):
        if pair not in words:
            words[pair] = set()
        words[pair].add(word)


def merge(word, pair, idx):
    newword = []
    if pair[0] not in word or pair[1] not in word:
        return word
    i = 0
    while i < len(word):
        if i + 1 < len(word) and pair[0] == word[i] and pair[1] == word[i + 1]:
            newword.append(idx)
            i += 2
        else:
            newword.append(word[i])
            i += 1
    return tuple(newword)


def _pretokenize_counts(chunks, pattern: str) -> Counter:
    """对文本块流做预分词并累计词型频次;chunks 是 str 迭代器。"""
    rx = re.compile(pattern)
    counts: Counter = Counter()
    for chunk in chunks:
        for tok in rx.findall(chunk):
            counts[tuple(tok.encode("utf-8"))] += 1
    return counts


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    pattern: str = gpt2pat,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """训练 BPE,返回 (vocab, merges)。

    input_path 为 UTF-8 文本文件,流式读取;special_tokens 参与词表但不参与合并。
    """
    # special token 切分(先于预分词,special token 整体不参与 BPE)
    sp_split = re.compile("|".join(re.escape(s) for s in special_tokens))

    def text_chunks():
        tail = ""
        with open(input_path, encoding="utf-8", errors="replace") as f:
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    if tail:
                        yield tail
                    return
                chunk = tail + chunk
                parts = sp_split.split(chunk)
                # 末片可能是被截断的 special token 或普通文本,留到下一块
                tail = parts.pop()
                yield from parts

    counts = _pretokenize_counts(text_chunks(), pattern)

    # pair -> 全语料频次;pair -> 含该 pair 的词型集合(完全维护)
    stats: dict[tuple[int, int], int] = {}
    words: dict[tuple[int, int], set[tuple[int, ...]]] = {}
    for w, f in counts.items():
        get_stats(w, f, stats)
        get_words(w, words)

    vocab = {i: bytes([i]) for i in range(256)}
    for i, s in enumerate(special_tokens, start=256):
        vocab[i] = s.encode("utf-8")

    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - 256 - len(special_tokens)
    for i in range(num_merges):
        # tie-break:频次 > pair 左字节 id > 右字节 id(与课程实现一致)
        pair = max(stats, key=lambda x: (stats[x], vocab[x[0]], vocab[x[1]]))
        idx = i + 256 + len(special_tokens)
        for curw in words[pair].copy():
            f = counts[curw]
            neww = merge(curw, pair, idx)
            counts[neww] = counts.get(neww, 0) + f
            del counts[curw]
            get_stats(neww, f, stats)
            get_stats(curw, -f, stats)
            for p in zip(curw, curw[1:]):
                words[p].discard(curw)
            get_words(neww, words)
        merges.append((vocab[pair[0]], vocab[pair[1]]))
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

    return vocab, merges
