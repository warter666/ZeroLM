"""把文本语料编码成 token id 数组(.npy),供训练用。

支持断点续跑:每 N 篇文档落一次盘,中断后从 state.json 记录的位置继续。
多进程并行(纯 Python 分词是 CPU 密集)。

用法:
    uv run python scripts/prepare_data.py                      # 全量 train + valid
    uv run python scripts/prepare_data.py --limit-docs 20000   # 只处理前 2 万篇(快速验证)
"""

from __future__ import annotations

import argparse
import array
import json
import mmap
import os
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zerolm.tokenizer import Tokenizer

EOT_BYTES = b"<|endoftext|>\n"

_worker_tok: Tokenizer | None = None
_worker_path: str | None = None


def _open_bin(path, mode: str):
    """打开文件,把 OSError 转成带路径的 SystemExit。

    数据预处理会跑几十分钟,中途单个文件读失败时给一条能看懂的报错,
    比裸 traceback 好得多。
    """
    try:
        return open(path, mode)
    except OSError as e:
        raise SystemExit(f"打不开 {path}: {e}") from e


def _init_worker(vocab_path: str, merges_path: str, special_tokens: list[str], text_path: str) -> None:
    global _worker_tok, _worker_path
    _worker_tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    _worker_path = text_path


def _encode_range(span: tuple[int, int]) -> list[int]:
    assert _worker_tok is not None
    assert _worker_path is not None, "worker 未初始化:_worker_path 还没设置"
    start, end = span
    with _open_bin(_worker_path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    text = data.decode("utf-8")
    # 文档以 <|endoftext|> 结尾;整体编码保证 special token 落成单个 id
    return _worker_tok.encode(text)


def _doc_spans(path: Path) -> list[tuple[int, int]]:
    """扫描文档边界(不把文件读进内存)。"""
    spans: list[tuple[int, int]] = []
    try:
        with open(path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                pos = 0
                n = len(mm)
                while pos < n:
                    idx = mm.find(EOT_BYTES, pos)
                    if idx == -1:
                        spans.append((pos, n))
                        break
                    spans.append((pos, idx + len(EOT_BYTES)))
                    pos = idx + len(EOT_BYTES)
            finally:
                mm.close()
    except OSError as e:
        raise SystemExit(f"读不了语料 {path}: {e}") from e
    return spans


def _finalize(bin_path: Path, npy_path: Path, dtype: str) -> None:
    n_tokens = bin_path.stat().st_size // np.dtype(dtype).itemsize
    with _open_bin(bin_path, "rb") as src, _open_bin(npy_path, "wb") as out:
        header = {"descr": np.dtype(dtype).str, "fortran_order": False, "shape": (n_tokens,)}
        np.lib.format.write_array_header_1_0(out, header)
        shutil.copyfileobj(src, out, length=1 << 22)
    print(f"{npy_path}: {n_tokens/1e6:.1f}M tokens")


def encode_split(
    src: Path,
    out_dir: Path,
    tok: Tokenizer,
    vocab_path: str,
    merges_path: str,
    special_tokens: list[str],
    workers: int,
    batch_docs: int,
    flush_docs: int,
    limit_docs: int,
    dtype: str,
) -> None:
    name = src.stem
    bin_path = out_dir / f"{name}.bin"
    npy_path = out_dir / f"{name}.npy"
    state_path = out_dir / f"{name}.state.json"
    itemsize = np.dtype(dtype).itemsize

    spans = _doc_spans(src)
    if limit_docs:
        spans = spans[:limit_docs]
    total = len(spans)
    print(f"{name}: {total} 篇文档")

    done_docs = 0
    done_tokens = 0
    if state_path.exists() and bin_path.exists():
        try:
            st = json.loads(state_path.read_text())
            done_docs = int(st["docs"])
            done_tokens = int(st["tokens"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            raise SystemExit(f"断点状态 {state_path} 不可用: {e}") from e
        # 回滚到上次落盘状态
        with _open_bin(bin_path, "r+b") as f:
            f.truncate(done_tokens * itemsize)
        print(f"续跑:已完成 {done_docs}/{total} 篇,{done_tokens/1e6:.1f}M tokens")
    else:
        try:
            bin_path.write_bytes(b"")
        except OSError as e:
            raise SystemExit(f"无法创建 {bin_path}: {e}") from e

    if done_docs >= total:
        if not npy_path.exists():
            _finalize(bin_path, npy_path, dtype)
        else:
            print(f"{npy_path} 已存在,跳过")
        return

    assert len(tok.vocab) <= 65536, "uint16 词表上限 65536,请用 --dtype uint32"
    typecode = {"uint16": "H", "uint32": "I"}[dtype]

    with Pool(workers, initializer=_init_worker, initargs=(vocab_path, merges_path, special_tokens, str(src))) as pool:
        with _open_bin(bin_path, "ab") as fout, tqdm(total=total - done_docs, desc=name, unit="doc") as bar:
            pending_since_flush = 0
            for batch_start in range(done_docs, total, batch_docs):
                batch = spans[batch_start : batch_start + batch_docs]
                buf = array.array(typecode)
                for toks in pool.imap(_encode_range, batch, chunksize=4):
                    buf.extend(toks)
                fout.write(buf.tobytes())
                fout.flush()
                done_docs = min(batch_start + len(batch), total)
                done_tokens += len(buf)
                pending_since_flush += len(batch)
                bar.update(len(batch))

                if pending_since_flush >= flush_docs or done_docs >= total:
                    tmp = state_path.with_suffix(".json.tmp")
                    try:
                        tmp.write_text(json.dumps({"docs": done_docs, "tokens": done_tokens}))
                        tmp.replace(state_path)
                    except OSError as e:
                        raise SystemExit(f"写不了断点状态 {state_path}: {e}") from e
                    pending_since_flush = 0

    if not npy_path.exists():
        _finalize(bin_path, npy_path, dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", default="data/zh_train.txt")
    ap.add_argument("--valid-file", default="data/zh_valid.txt")
    ap.add_argument("--tokenizer-dir", default="data/tokenizer")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--special-tokens", default="<|endoftext|>")
    ap.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    ap.add_argument("--batch-docs", type=int, default=2000)
    ap.add_argument("--flush-docs", type=int, default=20000, help="每 N 篇落一次盘(断点粒度)")
    ap.add_argument("--limit-docs", type=int, default=0, help="只处理前 N 篇(0=全量)")
    ap.add_argument("--dtype", default="uint16", choices=["uint16", "uint32"])
    ap.add_argument("--skip-valid", action="store_true")
    args = ap.parse_args()

    tok_dir = Path(args.tokenizer_dir)
    vocab_path = str(tok_dir / "vocab.json")
    merges_path = str(tok_dir / "merges.txt")
    special_tokens = [s for s in args.special_tokens.split(",") if s]
    tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    print(f"vocab={len(tok.vocab)}, merges={len(tok.merges)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encode_split(
        Path(args.train_file), out_dir, tok, vocab_path, merges_path, special_tokens,
        args.workers, args.batch_docs, args.flush_docs, args.limit_docs, args.dtype,
    )
    if not args.skip_valid:
        valid = Path(args.valid_file)
        if valid.exists():
            encode_split(
                valid, out_dir, tok, vocab_path, merges_path, special_tokens,
                args.workers, args.batch_docs, args.flush_docs, 0, args.dtype,
            )
        else:
            print(f"没有验证集 {valid},跳过")


if __name__ == "__main__":
    main()
