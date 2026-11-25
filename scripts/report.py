"""报告一次训练的可比指标,重点是 bits/byte(跨分词器可比的 compression 指标)。

为什么需要 bits/byte:
    val loss 是「每 token 的交叉熵(nats)」,换个词表这个数就变了 —— 词表越大,
    同样文本的 token 数越少,每 token 的 loss 通常越大。bits/byte 把它折算回
    「每个原始字节需要多少 bit」,不同分词器/不同语料之间才能横向比较。

公式:
    bits/token = loss(nats) / ln2
    bits/byte  = bits/token / (语料字节数 / token 数)

用法:
    uv run python scripts/report.py --run-dir experiments/zh-1ep
    uv run python scripts/report.py --run-dir experiments/zh-1ep --train-file data/zh_train.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

LN2 = math.log(2.0)


def read_metrics(path: Path) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """返回 (train 点, valid 点),每个是 [(step, loss), ...]。"""
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError as e:
        raise SystemExit(f"读不了 {path}: {e}") from e
    out: dict[str, list[tuple[int, float]]] = {"train": [], "valid": []}
    for r in rows:
        split = (r.get("split") or "").strip()
        if split not in out:
            continue
        try:
            out[split].append((int(float(r["step"])), float(r["loss"])))
        except (KeyError, TypeError, ValueError):
            continue
    for pts in out.values():
        pts.sort(key=lambda p: p[0])
    return out["train"], out["valid"]


def bytes_per_token(npy_path: Path, text_path: Path) -> float | None:
    """原始语料字节数 / token 数(分词器的平均压缩率)。"""
    if not (npy_path.exists() and text_path.exists()):
        return None
    try:
        n_tokens = len(np.load(npy_path, mmap_mode="r"))
        n_bytes = text_path.stat().st_size
    except OSError:
        return None
    return n_bytes / n_tokens if n_tokens else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--train-file", default="data/zh_train.txt")
    ap.add_argument("--npy", default="data/zh_train.npy")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg_path = run_dir / "config.json"
    try:
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
    except OSError as e:
        raise SystemExit(f"读不了 {cfg_path}: {e}") from e

    train, valid = read_metrics(run_dir / "metrics.csv")
    if not valid:
        raise SystemExit(f"{run_dir}/metrics.csv 里没有 valid 记录")

    best_step, best_loss = min(valid, key=lambda p: p[1])
    last_step, last_loss = valid[-1]
    init = train[0][1] if train else float("nan")

    bpt = bytes_per_token(Path(args.npy), Path(args.train_file))

    print(f"run: {run_dir.name}")
    print(f"  参数量        {config.get('d_model')}d × {config.get('num_layers')}L"
          f" / vocab {config.get('vocab_size')} / ctx {config.get('context_length')}")
    print(f"  步数          {last_step}  (batch {config.get('batch_size')} × ctx {config.get('context_length')})")
    print(f"  初始 train    {init:.4f}")
    print(f"  best valid    {best_loss:.4f}  @ step {best_step}")
    print(f"  末次 valid    {last_loss:.4f}  @ step {last_step}")
    print(f"  bits/token    {best_loss / LN2:.4f}")
    if bpt:
        print(f"  bytes/token   {bpt:.4f}")
        print(f"  bits/byte     {best_loss / LN2 / bpt:.4f}   ← 跨分词器可比")
        print(f"  等价压缩率    {8 / (best_loss / LN2 / bpt):.2f}x(每字节 8 bit 压到该值)")
        print("\n  说明:bits/byte 用来跨分词器比较。参考仓库的英文 val loss 是「另一个词表下的")
        print("        每 token 交叉熵」,和本仓库的中文 loss 不能直接比大小;要横比就算 bits/byte。")
    else:
        print(f"  (找不到 {args.npy} 或 {args.train_file},跳过 bits/byte)")


if __name__ == "__main__":
    main()
