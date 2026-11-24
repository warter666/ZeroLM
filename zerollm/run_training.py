"""训练入口:CLI、评估、断点续训、metrics.csv 记录。

用法:
    uv run python -m zerollm.run_training \
        --experiment-name zh-10k --train-data data/zh_train.npy --valid-data data/zh_valid.npy \
        --vocab-size 10000 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch

from zerollm.loss_optimizer import AdamW, clip_gradients, cosine_lr, cross_entropy
from zerollm.model import TransformerLM
from zerollm.training import get_batch, load_checkpoint, save_checkpoint


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ZeroLLM 训练")
    g = p.add_argument_group("experiment")
    g.add_argument("--experiment-name", required=True)
    g.add_argument("--run-root", default="experiments")
    g.add_argument("--random-seed", type=int, default=0)

    g = p.add_argument_group("data")
    g.add_argument("--train-data", required=True, help="一维 token id .npy")
    g.add_argument("--valid-data", required=True)

    g = p.add_argument_group("model")
    g.add_argument("--vocab-size", type=int, required=True)
    g.add_argument("--context-length", type=int, default=256)
    g.add_argument("--d-model", type=int, default=512)
    g.add_argument("--num-layers", type=int, default=4)
    g.add_argument("--num-heads", type=int, default=16)
    g.add_argument("--d-ff", type=int, default=1344)
    g.add_argument("--rope-theta", type=float, default=10000.0)
    g.add_argument(
        "--tie-embeddings",
        action="store_true",
        help="lm_head 复用 embedding 矩阵(GPT-2 式),参数量 22.70M -> 17.58M",
    )
    g.add_argument(
        "--sdpa",
        action="store_true",
        help="注意力改走 F.scaled_dot_product_attention(sm80+ 的 flash 内核),更快更省显存",
    )

    g = p.add_argument_group("optimizer")
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--min-lr", type=float, default=3e-5)
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--beta1", type=float, default=0.9)
    g.add_argument("--beta2", type=float, default=0.95)
    g.add_argument("--eps", type=float, default=1e-8)
    g.add_argument("--max-grad-norm", type=float, default=1.0)

    g = p.add_argument_group("training")
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--num-iters", type=int, default=10000)
    g.add_argument(
        "--stop-at",
        type=int,
        default=None,
        help="本次运行跑到第几步就停(默认跑到 num-iters)。分段训练用:LR 调度始终按 num-iters 计算,"
        "所以分段只是提前退出,不影响学习率曲线",
    )
    g.add_argument("--warmup-iters", type=int, default=100)
    g.add_argument("--eval-interval", type=int, default=100)
    g.add_argument("--eval-iters", type=int, default=10)
    g.add_argument("--save-interval", type=int, default=1000)
    g.add_argument("--log-interval", type=int, default=10)
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--checkpoint-in", default=None, help="从此 checkpoint 续训")
    g.add_argument("--checkpoint-out", default="checkpoint.pt")
    g.add_argument("--compile", action="store_true", help="torch.compile(首次较慢)")
    return p


def evaluate(model, valid_data, args, ctx) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(args.eval_iters):
            x, y = get_batch(valid_data, args.batch_size, args.context_length, args.device)
            with ctx:
                logits = model(x)
            total += cross_entropy(logits, y).item()
    return total / args.eval_iters


def main(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_root) / args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    device = torch.device(args.device)
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        tie_embeddings=args.tie_embeddings,
        use_sdpa=args.sdpa,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    assert args.warmup_iters < args.num_iters, "warmup 必须小于总步数"

    start_step = 0
    if args.checkpoint_in:
        start_step = load_checkpoint(args.checkpoint_in, model, optimizer)
        print(f"从 {args.checkpoint_in} 续训,起始步 {start_step}", flush=True)

    train_data = np.load(args.train_data, mmap_mode="r")
    valid_data = np.load(args.valid_data, mmap_mode="r")
    print(
        f"model params: {model.num_parameters()/1e6:.2f}M "
        f"(不含 embedding/head: {model.num_parameters(exclude_embeddings=True)/1e6:.2f}M)"
    )
    print(f"train tokens: {len(train_data)/1e6:.1f}M, valid tokens: {len(valid_data)/1e6:.1f}M")

    metrics_path = run_dir / "metrics.csv"
    write_header = not metrics_path.exists()
    # 句柄活到函数结束(csv 边跑边 flush),不用 with 块包住整个训练循环
    try:
        csv_f = open(metrics_path, "a", newline="", encoding="utf-8")  # noqa: SIM115
    except OSError as e:
        raise SystemExit(f"无法写 {metrics_path}: {e}") from e
    writer = csv.writer(csv_f)
    if write_header:
        writer.writerow(["step", "elapsed_seconds", "split", "loss", "learning_rate"])

    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
    # forward 用可能被 compile 的模型;checkpoint / eval 一律对原始模块(model),
    # 否则 state_dict 会带 "_orig_mod." 前缀,续训和 generate.py 加载全失败。
    fwd: torch.nn.Module = model
    if args.compile:
        fwd = cast("torch.nn.Module", torch.compile(model))

    t0 = time.time()
    step = start_step
    best_val, best_step = float("inf"), 0
    stop_at = args.stop_at if args.stop_at is not None else args.num_iters
    assert stop_at <= args.num_iters, "--stop-at 不能大于 --num-iters"
    while step < stop_at:
        lr = cosine_lr(
            step, args.lr, args.min_lr, args.warmup_iters, args.num_iters - args.warmup_iters
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        fwd.train()
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            logits = fwd(x)
            loss = cross_entropy(logits, y)
        loss.backward()
        grad_norm = clip_gradients(model.parameters(), args.max_grad_norm)
        optimizer.step()
        step += 1

        elapsed = time.time() - t0
        if step % args.log_interval == 0 or step == 1:
            tok_s = step * args.batch_size * args.context_length / max(elapsed, 1e-9)
            print(
                f"step {step:6d} | loss {loss.item():.4f} | lr {lr:.2e} "
                f"| gn {float(grad_norm):.2f} | {tok_s/1e3:.1f}k tok/s | {elapsed:.0f}s",
                flush=True,  # 重定向到文件时 stdout 是块缓冲,不 flush 会看不到进度
            )
            # train 行也落 csv,画 loss 曲线才有密集点
            writer.writerow([step, f"{elapsed:.2f}", "train", f"{loss.item():.6f}", f"{lr:.6e}"])
            csv_f.flush()

        if step % args.eval_interval == 0 or step == stop_at:
            val_loss = evaluate(fwd, valid_data, args, ctx)
            writer.writerow([step, f"{elapsed:.2f}", "valid", f"{val_loss:.6f}", f"{lr:.6e}"])
            csv_f.flush()
            if val_loss < best_val:
                best_val, best_step = val_loss, step
            print(
                f"step {step:6d} | VALID loss {val_loss:.4f} (best {best_val:.4f} @ {best_step})",
                flush=True,
            )

        if step % args.save_interval == 0 or step == stop_at:
            save_checkpoint(model, optimizer, step, run_dir / args.checkpoint_out)

    save_checkpoint(model, optimizer, step, run_dir / args.checkpoint_out)
    csv_f.close()
    print(f"完成:{step} 步,产物在 {run_dir}")
    print(f"best valid loss {best_val:.4f} @ step {best_step};画图:uv run python scripts/plot_loss.py --run-dir {run_dir}")


if __name__ == "__main__":
    main(build_argparser().parse_args())