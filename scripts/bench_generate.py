"""KV cache 加速基准:同一前缀、固定种子,use_cache=True/False 交替跑 N 次取中位数。

为什么用中位数而不是单次:
    单次测量会被编译、缓存预热、调度抖动影响,第一次往往偏慢或偏快。
    交替执行(不开→开→不开→开)还能抹掉"机器越跑越热"这类单向漂移。

指标:
    - 每次的墙钟时间与中位数
    - tokens/s(生成 token 数 / 时间),批量按 B × max_new_tokens 计
    - 加速倍数 = 关闭中位数 / 开启中位数
    - 两条路径的输出必须逐 id 相同(否则快也没意义)

用法:
    # CPU(不占 GPU,可与训练并行)
    uv run python scripts/bench_generate.py --run-dir experiments/zh-1ep --device cpu --repeats 3

    # GPU(需空闲,顺带记录峰值显存)
    uv run python scripts/bench_generate.py --run-dir experiments/zh-1ep --device cuda --repeats 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from zerolm.generating import generate_ids
from zerolm.model import TransformerLM
from zerolm.tokenizer import Tokenizer


def load_run(run_dir: Path, device: str):
    """从 experiment 目录恢复模型与分词器(与 scripts/generate.py 同口径)。"""
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    ckpt = torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=False)
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config.get("rope_theta", 10000.0),
        tie_embeddings=config.get("tie_embeddings", False),
        use_sdpa=config.get("use_sdpa", False),
    ).to(device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()

    tok_dir = Path(config.get("tokenizer_dir", "data/tokenizer"))
    special = config.get("special_tokens") or ["<|endoftext|>"]
    tok = Tokenizer.from_files(
        str(tok_dir / "vocab.json"),
        str(tok_dir / "merges.txt"),
        list(special),
    )
    return model, tok, config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--prompt", default="从前有一只小兔子,它")
    ap.add_argument("--batch", type=int, default=1, help="同一前缀复制成 B 条(等长,无 padding)")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--compile", action="store_true",
                    help="用 torch.compile 包装模型。不编译时每步的 CPU 派发开销远大于 GPU 计算,"
                         "cache 省下的计算会被淹没;想看 cache 的真实收益必须加这个")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    raw_model, tok, config = load_run(run_dir, args.device)
    model: torch.nn.Module = raw_model
    ctx = config["context_length"]
    prompts = [args.prompt] * args.batch

    if args.compile:
        # mode=default 而非 reduce-overhead:带 cache 时每个 token 的序列长度都不同,
        # CUDA Graphs 会把图覆盖掉并报 "output overwritten by a subsequent run"。
        model = torch.compile(model, mode="default")  # type: ignore[assignment]
        print("已启用 torch.compile(mode=default)")

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    print(f"run {run_dir.name}: {raw_model.num_parameters()/1e6:.2f}M 参数, "
          f"ctx {ctx}, device {args.device}")
    print(f"配置: B={args.batch} × {args.max_new_tokens} token, top_p={args.top_p}, "
          f"seed={args.seed}, {args.repeats} 次取中位数\n")

    # 预热一次,排除首次调用的编译/分配开销
    generate_ids(prompts, model, tok, device=args.device, context_size=ctx,
                 max_new_tokens=8, top_p=args.top_p, use_cache=False)

    results: dict[bool, list[float]] = {False: [], True: []}
    outputs: dict[bool, list[list[int]]] = {}
    order = [False, True] * args.repeats  # 交替,抹掉单向漂移
    for use_cache in order:
        torch.manual_seed(args.seed)  # 同 seed:两条路径应产出完全相同的 token
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ids, _lens, _stats = generate_ids(
            prompts, model, tok, device=args.device, context_size=ctx,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_p=args.top_p, use_cache=use_cache,
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        results[use_cache].append(dt)
        outputs[use_cache] = ids
        print(f"  use_cache={use_cache!s:5s}  第 {len(results[use_cache])} 次: {dt:7.2f}s")

    med = {k: statistics.median(v) for k, v in results.items()}
    n_new = args.batch * args.max_new_tokens
    print()
    for use_cache in (False, True):
        v = results[use_cache]
        print(f"use_cache={use_cache!s:5s}  中位数 {med[use_cache]:7.2f}s   "
              f"最快 {min(v):7.2f}s   最慢 {max(v):7.2f}s   "
              f"{n_new / med[use_cache]:8.1f} tok/s")
    speedup = med[False] / med[True]
    print(f"\n加速 {speedup:.2f}x(关闭 {med[False]:.2f}s → 开启 {med[True]:.2f}s)")

    same = outputs[False] == outputs[True]
    print(f"两条路径输出逐 id 相同: {same}")
    if not same:
        raise SystemExit("缓存路径输出不一致,加速比无意义")

    if args.device.startswith("cuda"):
        peak = torch.cuda.max_memory_allocated() / 1024**2
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"GPU 峰值显存: {peak:.0f} MB / {total:.0f} MB ({peak/total*100:.1f}%)")


if __name__ == "__main__":
    main()
