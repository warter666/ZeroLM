"""下载中文 TinyStories 语料(RobinChen2001/TinyStories-Zh-2M)并生成 train/valid 文本。

输出:
    data/zh_train.txt
    data/zh_valid.txt
每个文档之间用 <|endoftext|> 分隔(与英文 TinyStories 的文档切分一致)。

用法:
    uv run python scripts/download_data.py --valid-ratio 0.01
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

REPO = "RobinChen2001/TinyStories-Zh-2M"
FILES = [
    "data/train-00000-of-00003.parquet",
    "data/train-00001-of-00003.parquet",
    "data/train-00002-of-00003.parquet",
]
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
EOT = "<|endoftext|>"


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"已存在,跳过:{dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"下载 {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--valid-ratio", type=float, default=0.01, help="验证集比例(按文档数)")
    ap.add_argument("--max-docs", type=int, default=0, help="限制文档数(0=全量,调试用)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw"
    for rel in FILES:
        download(BASE + rel, raw_dir / Path(rel).name)

    train_path = data_dir / "zh_train.txt"
    valid_path = data_dir / "zh_valid.txt"
    if train_path.exists() and valid_path.exists():
        print(f"已存在,跳过生成:{train_path} / {valid_path}")
        return

    n_train = n_valid = 0
    written = 0
    with open(train_path, "w", encoding="utf-8") as ftr, open(valid_path, "w", encoding="utf-8") as fva:
        for rel in FILES:
            pf = pq.ParquetFile(raw_dir / Path(rel).name)
            total = pf.metadata.num_rows
            for batch in pf.iter_batches(batch_size=10000, columns=["text"]):
                texts = batch.column("text").to_pylist()
                for t in texts:
                    if args.max_docs and written >= args.max_docs:
                        print(f"达到 --max-docs {args.max_docs},停止")
                        print(f"train docs: {n_train}, valid docs: {n_valid}")
                        return
                    t = (t or "").strip()
                    if not t:
                        continue
                    # 每 100 篇取 1 篇进验证集(按 valid_ratio 粗粒度分配)
                    if (written % max(int(round(1 / args.valid_ratio)), 1)) == 0:
                        fva.write(t + "\n" + EOT + "\n")
                        n_valid += 1
                    else:
                        ftr.write(t + "\n" + EOT + "\n")
                        n_train += 1
                    written += 1
            print(f"{rel}: {total} rows 已处理")
    print(f"train docs: {n_train}, valid docs: {n_valid}")
    print(f"输出:{train_path} , {valid_path}")


if __name__ == "__main__":
    main()