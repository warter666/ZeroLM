"""训练 BPE 分词器并保存 vocab.json / merges.txt。

默认从全量语料的**前 N MB** 取样训练(10k 词表不需要几 GB 数据,取样能省几十分钟)。
用法:
    uv run python scripts/train_tokenizer.py --vocab-size 10000 --sample-mb 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zerollm.tokenizer import DEFAULT_PATTERN, Tokenizer
from zerollm.train_bpe import train_bpe

SPECIAL_TOKENS = ["<|endoftext|>"]


def sample_text(src: Path, dst: Path, sample_bytes: int) -> Path:
    """按字节截取前 sample_bytes,在文档边界收尾。"""
    if dst.exists() and dst.stat().st_size > 0:
        print(f"采样文件已存在:{dst}")
        return dst
    data = src.read_bytes()[:sample_bytes]
    # 回退到最后一个 <|endoftext|> 之后,保证结尾是完整文档
    idx = data.rfind(b"<|endoftext|>")
    if idx > 0:
        data = data[: idx + len(b"<|endoftext|>")]
    dst.write_bytes(data)
    print(f"采样 {len(data)/1e6:.1f}MB -> {dst}")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", default="data/zh_train.txt")
    ap.add_argument("--out-dir", default="data/tokenizer")
    ap.add_argument("--vocab-size", type=int, default=10000)
    ap.add_argument("--sample-mb", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=200, help="每 N 个 merge 打印一次进度(0=不打印)")
    args = ap.parse_args()

    src = Path(args.train_file)
    if not src.exists():
        raise SystemExit(f"找不到 {src},先跑 scripts/download_data.py")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_path = out_dir / "sample.txt"
    sample_text(src, sample_path, args.sample_mb * 1024 * 1024)

    t0 = time.time()
    print(f"训练 BPE: vocab_size={args.vocab_size}, special={SPECIAL_TOKENS}")
    vocab, merges = train_bpe(str(sample_path), args.vocab_size, SPECIAL_TOKENS, log_every=args.log_every)
    print(f"完成,耗时 {time.time()-t0:.1f}s,vocab={len(vocab)},merges={len(merges)}")

    tok = Tokenizer(vocab, merges, SPECIAL_TOKENS)
    vocab_path = out_dir / "vocab.json"
    merges_path = out_dir / "merges.txt"
    tok.save(str(vocab_path), str(merges_path))
    print(f"已保存:{vocab_path} , {merges_path}")

    # 自检:roundtrip + 压缩率(字节数 / token 数)
    text = sample_path.read_text(encoding="utf-8")[:20000]
    ids = tok.encode(text)
    assert tok.decode(ids) == text, "roundtrip 失败"
    ratio = len(text.encode("utf-8")) / max(len(ids), 1)
    print(f"roundtrip OK;压缩率约 {ratio:.2f} bytes/token(前 2 万字符)")


if __name__ == "__main__":
    main()