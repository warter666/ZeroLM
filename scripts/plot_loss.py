"""画 loss 曲线(读 experiments/<name>/metrics.csv)。

metrics.csv 由 run_training.py 写入,列为:
    step, elapsed_seconds, split, loss, learning_rate
其中 split ∈ {train, valid}:train 每 log_interval 一行,valid 每 eval_interval 一行。

用法:
    uv run python scripts/plot_loss.py --run-dir experiments/zh-10k
    uv run python scripts/plot_loss.py --run-dir experiments/zh-10k experiments/smoke --out figures/cmp.png
    uv run python scripts/plot_loss.py --run-dir experiments/zh-10k --smooth 0.99 --log-x
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

COLORS = ["#2b6cb0", "#c05621", "#2f855a", "#805ad5", "#b83280"]

# 上图默认用英文标签(可移植);只有环境里真装了中文字体时才用中文,
# 否则中文图例会渲染成方框。
CJK_FONTS = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "Source Han Sans SC", "SimHei"]


def has_cjk_font() -> bool:
    try:
        installed = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:  # noqa: BLE001
        return False
    for name in CJK_FONTS:
        if name in installed:
            matplotlib.rcParams["font.sans-serif"] = [name, *matplotlib.rcParams["font.sans-serif"]]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


def ema(values: list[float], alpha: float) -> list[float]:
    """指数滑动平均(alpha 越接近 1 越平滑)。"""
    out: list[float] = []
    acc = None
    for v in values:
        acc = v if acc is None else alpha * acc + (1 - alpha) * v
        out.append(acc)
    return out


def read_metrics(path: Path) -> dict[str, list[tuple[int, float, float]]]:
    """返回 {split: [(step, loss, lr), ...]}。"""
    splits: dict[str, list[tuple[int, float, float]]] = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError as e:
        raise SystemExit(f"读不了 {path}: {e}") from e
    for row in rows:
        split = (row.get("split") or "").strip()
        if not split:
            continue
        try:
            step = int(float(row["step"]))
            loss = float(row["loss"])
        except (KeyError, TypeError, ValueError):
            continue  # 空行 / 被中断写一半的行
        splits.setdefault(split, []).append((step, loss, float(row.get("learning_rate") or 0.0)))
    for pts in splits.values():
        pts.sort(key=lambda p: p[0])
    return splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", nargs="+", required=True, help="一个或多个 experiments/<name> 目录")
    ap.add_argument("--out", default=None, help="输出图片(默认 figures/<第一个 run 的名字>.png)")
    ap.add_argument("--smooth", type=float, default=0.9, help="train 曲线 EMA 平滑系数,0=不平滑")
    ap.add_argument("--log-x", action="store_true", help="x 轴用 log 刻度")
    ap.add_argument("--no-lr", action="store_true", help="不画下面的学习率子图")
    ap.add_argument("--valid-only", action="store_true", help="只画 valid 曲线(消融对比用)")
    ap.add_argument("--label", nargs="+", default=None, help="每个 run 的图例名,个数须与 --run-dir 一致")
    ap.add_argument("--title", default=None, help="图标题(默认自动生成)")
    ap.add_argument("--xlim", nargs=2, type=float, default=None, metavar=("LO", "HI"), help="x 轴范围")
    ap.add_argument("--ylim", nargs=2, type=float, default=None, metavar=("LO", "HI"), help="y 轴范围")
    args = ap.parse_args()
    if not has_cjk_font():
        print("提示:未找到中文字体,中文图例会显示为方框;请用英文 --label")

    runs = []
    for d in args.run_dir:
        path = Path(d) / "metrics.csv"
        if not path.exists():
            raise SystemExit(f"找不到 {path}")
        runs.append((Path(d).name, read_metrics(path)))
    labels = list(args.label) if args.label else [name for name, _ in runs]
    if len(labels) != len(runs):
        raise SystemExit(f"--label 给了 {len(labels)} 个,但 --run-dir 有 {len(runs)} 个")

    n_panel = 1 if args.no_lr else 2
    fig, axes = plt.subplots(
        n_panel, 1, figsize=(10, 4.2 * n_panel), sharex=True, gridspec_kw={"height_ratios": [3, 1][:n_panel]}
    )
    ax = axes[0] if n_panel > 1 else axes
    ax_lr = None if args.no_lr else axes[1]

    for i, (_, splits) in enumerate(runs):
        color = COLORS[i % len(COLORS)]
        train = splits.get("train", [])
        valid = splits.get("valid", [])
        label = labels[i] if len(runs) > 1 else ""

        if train and not args.valid_only:
            xs = [p[0] for p in train]
            ys = [p[1] for p in train]
            ax.plot(xs, ys, color=color, alpha=0.25, linewidth=0.8)
            ax.plot(xs, ema(ys, args.smooth), color=color, linewidth=1.6,
                    label=f"{label} train".strip() if train else None)
        if valid:
            xs = [p[0] for p in valid]
            ys = [p[1] for p in valid]
            ax.plot(xs, ys, color=color, marker="o", markersize=3.5, linewidth=1.4,
                    label=f"{label} valid".strip() if valid else None)
            best_step = min(valid, key=lambda p: p[1])
            # 标注靠近右边缘时往左摆,否则文字会被画布切掉
            near_right = best_step[0] > 0.7 * max(p[0] for p in valid)
            ax.annotate(
                f"best {best_step[1]:.4f} @ {best_step[0]}",
                xy=(best_step[0], best_step[1]),
                xytext=(-8 if near_right else 8, 8 + 16 * (i % 3)),
                textcoords="offset points",
                horizontalalignment="right" if near_right else "left",
                fontsize=8, color=color,
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8},
            )
            if ax_lr is not None:
                ax_lr.plot(xs, [p[2] for p in valid], color=color, linewidth=1.2)

        if train and ax_lr is not None:
            ax_lr.plot([p[0] for p in train], [p[2] for p in train], color=color, linewidth=0.8, alpha=0.6)

    ax.set_ylabel("cross-entropy loss")
    if args.title:
        ax.set_title(args.title)
    else:
        ax.set_title("ZeroLM valid loss" if args.valid_only else "ZeroLM loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    if args.log_x:
        ax.set_xscale("log")
    if args.xlim:
        ax.set_xlim(*args.xlim)
    if args.ylim:
        ax.set_ylim(*args.ylim)
    if ax_lr is not None:
        ax_lr.set_ylabel("learning rate")
        ax_lr.grid(alpha=0.3)
        ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # n_panel==1 时 subplots 返回单个 Axes,不能下标
    (axes[-1] if n_panel > 1 else ax).set_xlabel("step")

    out = Path(args.out) if args.out else Path("figures") / f"{runs[0][0]}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"已保存 {out}")

    for i, (_, splits) in enumerate(runs):
        valid = splits.get("valid", [])
        train = splits.get("train", [])
        if valid:
            best = min(valid, key=lambda p: p[1])
            print(f"{labels[i]}: train 点 {len(train)} 个, valid 点 {len(valid)} 个, "
                  f"best valid {best[1]:.4f} @ {best[0]}, 末次 valid {valid[-1][1]:.4f}")


if __name__ == "__main__":
    main()
