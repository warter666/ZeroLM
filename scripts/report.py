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
    # 多个 run 时输出一行一个 run 的横向对照表(1ep / 3ep / tie 消融)
    uv run python scripts/report.py --run-dir experiments/zh-1ep experiments/zh-3ep
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np

LN2 = math.log(2.0)


def read_metrics(path: Path) -> tuple[list[tuple[int, float, float]], list[tuple[int, float, float]]]:
    """返回 (train 点, valid 点),每个是 [(step, loss, elapsed_seconds), ...]。"""
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError as e:
        raise SystemExit(f"读不了 {path}: {e}") from e
    out: dict[str, list[tuple[int, float, float]]] = {"train": [], "valid": []}
    for r in rows:
        split = (r.get("split") or "").strip()
        if split not in out:
            continue
        try:
            step = int(float(r["step"]))
            loss = float(r["loss"])
            elapsed = float(r.get("elapsed_seconds") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        out[split].append((step, loss, elapsed))
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


def corpus_tokens(npy_path: Path) -> int | None:
    """语料里的唯一 token 数(用来把“看过多少 token”换算成 epoch)。"""
    try:
        return len(np.load(npy_path, mmap_mode="r"))
    except (OSError, ValueError):
        return None


def estimate_params(config: dict) -> float:
    """按 config 估算参数量(百万)。tie_embeddings=False 时 lm_head 与 embedding 各一份。"""
    d = config.get("d_model") or 0
    layers = config.get("num_layers") or 0
    d_ff = config.get("d_ff") or 0
    vocab = config.get("vocab_size") or 0
    emb = vocab * d * (1 if config.get("tie_embeddings") else 2)
    per_layer = 4 * d * d + 3 * d * d_ff  # q,k,v,o + gate/up/down
    return (emb + layers * per_layer) / 1e6


def summarize(run_dir: Path, npy: Path, train_file: Path, unique_tokens: int | None = None) -> dict | None:
    """读一个 run 的 config.json / metrics.csv,算出可横向比较的一组数。"""
    cfg_path = run_dir / "config.json"
    try:
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
    except OSError as e:
        print(f"跳过 {run_dir}:读不了 {cfg_path} ({e})")
        return None

    train, valid = read_metrics(run_dir / "metrics.csv")
    if not valid:
        print(f"跳过 {run_dir}:metrics.csv 里没有 valid 记录")
        return None

    best_step, best_loss, _ = min(valid, key=lambda p: p[1])
    last_step, last_loss, _ = valid[-1]
    batch = config.get("batch_size") or 0
    ctx = config.get("context_length") or 0
    total = config.get("num_iters") or 0
    eval_every = config.get("eval_interval") or 1
    finished = not total or last_step >= total - eval_every
    seen_tokens = last_step * batch * ctx
    # 看过 token / 唯一 token:>1 表示同一份语料重复过
    epochs = seen_tokens / unique_tokens if unique_tokens else None
    # 每步耗时:取相邻 train 点差分的中位数。
    # 续训会把 elapsed_seconds 归零(zh-1ep 在 step 29200 就发生过),
    # 用首末差分会被这种回跳污染;中位数能自然跳过。
    deltas: list[float] = []
    for (s0, _, t0), (s1, _, t1) in zip(train, train[1:], strict=False):
        if s1 > s0 and t1 >= t0:
            deltas.append((t1 - t0) / (s1 - s0) * 1000)
    ms_per_step = statistics.median(deltas) if deltas else None

    return {
        "run": run_dir.name,
        "config": config,
        "init": train[0][1] if train else float("nan"),
        "best_loss": best_loss,
        "best_step": best_step,
        "last_loss": last_loss,
        "last_step": last_step,
        "bpt": bytes_per_token(npy, train_file),
        "tokens": seen_tokens,
        "epochs": epochs,
        "ms_per_step": ms_per_step,
        "finished": finished,
    }


def print_single(r: dict, npy: str, train_file: str) -> None:
    """单个 run 的详细版(与老输出一致)。"""
    config, bpt = r["config"], r["bpt"]
    print(f"run: {r['run']}")
    print(f"  参数量        {config.get('d_model')}d × {config.get('num_layers')}L"
          f" / vocab {config.get('vocab_size')} / ctx {config.get('context_length')}")
    print(f"  步数          {r['last_step']}  (batch {config.get('batch_size')} × ctx {config.get('context_length')})")
    print(f"  初始 train    {r['init']:.4f}")
    print(f"  best valid    {r['best_loss']:.4f}  @ step {r['best_step']}")
    print(f"  末次 valid    {r['last_loss']:.4f}  @ step {r['last_step']}")
    print(f"  bits/token    {r['best_loss'] / LN2:.4f}")
    if bpt:
        print(f"  bytes/token   {bpt:.4f}")
        print(f"  bits/byte     {r['best_loss'] / LN2 / bpt:.4f}   ← 跨分词器可比")
        print(f"  等价压缩率    {8 / (r['best_loss'] / LN2 / bpt):.2f}x(每字节 8 bit 压到该值)")
        print("\n  说明:bits/byte 用来跨分词器比较。参考仓库的英文 val loss 是「另一个词表下的")
        print("        每 token 交叉熵」,和本仓库的中文 loss 不能直接比大小;要横比就算 bits/byte。")
    else:
        print(f"  (找不到 {npy} 或 {train_file},跳过 bits/byte)")


def print_table(rows: list[dict]) -> None:
    """多个 run 一行一个,便于把 1ep / 3ep / tie 消融摆在一起看。"""
    print("| run | 参数量(M,估算) | tie | 步数 | 看过 token | epoch | ms/step | best valid "
          "| 末次 valid | bits/token | bits/byte | 状态 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        cfg, bpt = r["config"], r["bpt"]
        bits_byte = r["best_loss"] / LN2 / bpt if bpt else None
        cells = [
            r["run"],
            f"{estimate_params(cfg):.2f}",
            "on" if cfg.get("tie_embeddings") else "off",
            f"{r['last_step']}",
            f"{r['tokens'] / 1e6:.1f}M",
            f"{r['epochs']:.2f}" if r["epochs"] else "—",
            f"{r['ms_per_step']:.0f}" if r["ms_per_step"] else "—",
            f"{r['best_loss']:.4f} @ {r['best_step']}",
            f"{r['last_loss']:.4f}",
            f"{r['best_loss'] / LN2:.4f}",
            f"{bits_byte:.4f}" if bits_byte else "—",
            "跑完" if r["finished"] else "进行中",
        ]
        print("| " + " | ".join(cells) + " |")
    print("\n说明:参数量按 config 估算(embedding + 每层 4d² + 3d·d_ff),tie=on 时 lm_head 复用 embedding。")
    print("      epoch = 看过 token / 唯一 token,>1 表示同一份语料重复过(不等于新数据)。")
    print("      ms/step 由 metrics.csv 的 elapsed_seconds 差分算,含 eval 与存档开销。")
    print("      bits/byte 只在同一份语料、同一分词器下横向比较才公平;换语料要重算 bytes/token。")
    print("      它只是「理想算术编码器在该模型分布下的期望码长」,不是实测压缩率:")
    print("      真实压缩还要加上模型权重体积与编码器开销。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, nargs="+",
                    help="一个或多个 run 目录;多个时输出横向对照表")
    ap.add_argument("--train-file", default="data/zh_train.txt")
    ap.add_argument("--npy", default="data/zh_train.npy")
    args = ap.parse_args()
    unique_tokens = corpus_tokens(Path(args.npy))

    rows = [
        r
        for r in (
            summarize(Path(d), Path(args.npy), Path(args.train_file), unique_tokens)
            for d in args.run_dir
        )
        if r
    ]
    if not rows:
        raise SystemExit("没有可用的 run")
    if len(rows) == 1:
        print_single(rows[0], args.npy, args.train_file)
    else:
        print_table(rows)


if __name__ == "__main__":
    main()
