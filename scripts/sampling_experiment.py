"""采样旋钮对比实验:temperature / top-k / top-p。

设计:
- 固定同一条前缀(可给多条),每个 cell 用同一个 RNG 种子,保证可比;
- 一次只动一个旋钮,其余保持"不过滤 + temperature=1.0";
- 每个 cell 采 N 条,统计三件事:
  连贯性 self_ppl / eos_rate / valid_char_rate,
  多样性 distinct-1/2/3,
  重复模式 rep4_rate / max_ngram4_freq。

输出:
    experiments/sampling-<run>/metrics.csv   每 cell 一行的汇总
    experiments/sampling-<run>/samples.md    汇总表 + 每组 2 条样例(写结论用)
    figures/sampling_<run>.png               三张旋钮曲线(可选,需要 matplotlib)

用法:
    uv run python scripts/sampling_experiment.py --run-dir experiments/zh-xxxx
    uv run python scripts/sampling_experiment.py --run-dir experiments/zh-xxxx --num-samples 8 --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zerollm.generating import generate_ids
from zerollm.model import TransformerLM
from zerollm.tokenizer import Tokenizer
from zerollm.training import load_checkpoint

DEFAULT_PROMPTS = [
    "从前有一只小兔子，它",
    "一天早上，小熊发现",
    "小猫很饿，于是",
]
# 一次只动一个旋钮,其余保持中性
SWEEPS = {
    "temperature": {
        "values": [0.5, 0.7, 1.0, 1.3, 1.6, 2.0],
        "fixed": {"top_p": 1.0, "top_k": None},
    },
    "top_k": {
        "values": [1, 5, 20, 50, 200, 1000],
        "fixed": {"temperature": 1.0, "top_p": 1.0},
    },
    "top_p": {
        "values": [0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
        "fixed": {"temperature": 1.0, "top_k": None},
    },
}


def ngrams(seq: list[int], n: int) -> list[tuple[int, ...]]:
    return [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]


def distinct_n(seq: list[int], n: int) -> float:
    grams = ngrams(seq, n)
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def rep_n(seq: list[int], n: int) -> float:
    """重复 n-gram 占比:n-gram 中"该 gram 出现次数 > 1"的比例。"""
    grams = ngrams(seq, n)
    if not grams:
        return 0.0
    counts = Counter(grams)
    dup = sum(c for c in counts.values() if c > 1)
    return dup / len(grams)


def max_freq(seq: list[int], n: int) -> int:
    grams = ngrams(seq, n)
    return max(Counter(grams).values()) if grams else 0


def self_perplexity(model, ids: list[int], prompt_len: int, device) -> float:
    """模型对自己这整段输出的平均 token 负对数似然(指数化)。只看续写部分。"""
    if len(ids) <= prompt_len + 1:
        return float("nan")
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"
    ):
        logits = model(x)
    logits = logits.float()
    logp = torch.log_softmax(logits, dim=-1)
    tgt = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)[0]
    cont = tgt[prompt_len:]  # 只统计续写部分
    return float(math.exp(-cont.mean().item())) if cont.numel() else float("nan")


def valid_char_rate(text: str) -> float:
    """有效字符比例:中文 / 字母数字 / 常见标点 之外的字符算"无效"。"""
    if not text:
        return 0.0
    ok = 0
    for ch in text:
        if ch.isalnum() or "\u4e00" <= ch <= "\u9fff" or ch in "，。！？、；：\"''（）《》…— \n":
            ok += 1
    return ok / len(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default="checkpoint.pt")
    ap.add_argument("--tokenizer-dir", default="data/tokenizer")
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--prompt", action="append", default=None, help="可重复;默认用内置三条")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg_path = run_dir / "config.json"
    try:
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
    except OSError as e:
        raise SystemExit(f"读不了 {cfg_path}: {e}") from e
    except json.JSONDecodeError as e:
        raise SystemExit(f"{cfg_path} 不是合法 JSON: {e}") from e
    tok = Tokenizer.from_files(
        str(Path(args.tokenizer_dir) / "vocab.json"),
        str(Path(args.tokenizer_dir) / "merges.txt"),
        ["<|endoftext|>"],
    )
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        tie_embeddings=bool(config.get("tie_embeddings", False)),
        use_sdpa=bool(config.get("sdpa", False)),
    ).to(args.device)
    step = load_checkpoint(run_dir / args.checkpoint, model)
    model.eval()
    prompts = args.prompt or DEFAULT_PROMPTS
    eos_id = next((i for i, b in tok.vocab.items() if b == b"<|endoftext|>"), None)
    print(f"checkpoint step {step}, {model.num_parameters()/1e6:.2f}M 参数, {len(prompts)} 条前缀\n")

    out_dir = Path("experiments") / f"sampling-{run_dir.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    examples: dict[tuple[str, float], list[str]] = {}

    for knob, spec in SWEEPS.items():
        for value in spec["values"]:
            kwargs = dict(spec["fixed"])
            if knob == "temperature":
                kwargs["temperature"] = value
            elif knob == "top_k":
                kwargs["top_k"] = int(value)
            else:
                kwargs["top_p"] = float(value)
            kwargs.setdefault("temperature", 1.0)
            kwargs.setdefault("top_p", 1.0)
            kwargs.setdefault("top_k", None)

            agg = {"d1": [], "d2": [], "d3": [], "rep4": [], "mx4": [], "ppl": [], "eos": [], "valid": []}
            texts_for_examples: list[str] = []
            for prompt in prompts:
                # 同一个 cell 的同一个 prompt 用固定种子,跨 cell 可比
                torch.manual_seed(args.seed)
                ids_batch, lens = generate_ids(
                    [prompt] * args.num_samples, model, tok,
                    device=args.device, context_size=config["context_length"],
                    max_new_tokens=args.max_new_tokens, eos_id=eos_id, **kwargs,
                )
                for ids, plen in zip(ids_batch, lens, strict=True):
                    cont = ids[plen:]
                    if eos_id is not None and eos_id in cont:
                        cont = cont[: cont.index(eos_id)]
                        agg["eos"].append(1.0)
                    else:
                        agg["eos"].append(0.0)
                    if not cont:
                        continue
                    text = tok.decode(cont)
                    agg["d1"].append(distinct_n(cont, 1))
                    agg["d2"].append(distinct_n(cont, 2))
                    agg["d3"].append(distinct_n(cont, 3))
                    agg["rep4"].append(rep_n(cont, 4))
                    agg["mx4"].append(max_freq(cont, 4))
                    agg["valid"].append(valid_char_rate(text))
                    agg["ppl"].append(self_perplexity(model, ids, plen, args.device))
                    if len(texts_for_examples) < 2 and len(text) > 20:
                        texts_for_examples.append(text.strip())

            def mean(k: str, agg: dict[str, list[float]] = agg) -> float:
                vals = [v for v in agg[k] if not math.isnan(v)]
                return sum(vals) / max(1, len(vals))

            row = {
                "knob": knob,
                "value": value,
                "temperature": kwargs["temperature"],
                "top_k": kwargs["top_k"],
                "top_p": kwargs["top_p"],
                "distinct_1": round(mean("d1"), 4),
                "distinct_2": round(mean("d2"), 4),
                "distinct_3": round(mean("d3"), 4),
                "rep4_rate": round(mean("rep4"), 4),
                "max_ngram4_freq": round(mean("mx4"), 2),
                "self_ppl": round(mean("ppl"), 2),
                "eos_rate": round(mean("eos"), 3),
                "valid_char_rate": round(mean("valid"), 4),
                "samples": len(agg["d1"]),
            }
            rows.append(row)
            examples[(knob, value)] = texts_for_examples
            print(
                f"{knob:11s} = {value:<6} d2 {row['distinct_2']:.3f} rep4 {row['rep4_rate']:.3f} "
                f"ppl {row['self_ppl']:8.2f} eos {row['eos_rate']:.2f} valid {row['valid_char_rate']:.3f}"
            )
        print()

    csv_path = out_dir / "metrics.csv"
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    except OSError as e:
        raise SystemExit(f"写不了 {csv_path}: {e}") from e
    print(f"已写 {csv_path}")

    # ---- markdown:汇总表 + 样例,直接当结论底稿 ----
    md = [f"# 采样旋钮对比 — {run_dir.name}(step {step})\n",
          f"每条前缀 {args.num_samples} 条样本,{len(prompts)} 条前缀,固定种子 {args.seed}。"
          "一次只动一个旋钮,其余为 temperature=1.0 / 不过滤。\n"]
    for knob in SWEEPS:
        sub = [r for r in rows if r["knob"] == knob]
        md.append(f"\n## {knob}\n\n")
        md.append("| 值 | distinct-2 | rep4_rate | max 4-gram 频次 | self_ppl | EOS 命中率 | 有效字符率 |\n")
        md.append("| --- | --- | --- | --- | --- | --- | --- |\n")
        for r in sub:
            md.append(
                f"| {r['value']} | {r['distinct_2']:.3f} | {r['rep4_rate']:.3f} | "
                f"{r['max_ngram4_freq']:.1f} | {r['self_ppl']:.2f} | {r['eos_rate']:.2f} | "
                f"{r['valid_char_rate']:.3f} |\n"
            )
        md.append("\n样例:\n")
        for r in sub:
            ex = examples.get((knob, r["value"])) or []
            if ex:
                md.append(f"\n- **{knob}={r['value']}**:{ex[0][:160]}\n")
    (out_dir / "samples.md").write_text("".join(md), encoding="utf-8")
    print(f"已写 {out_dir / 'samples.md'}")

    # ---- 图 ----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("没有 matplotlib,跳过画图")
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for knob, ax in zip(SWEEPS, axes, strict=True):
        sub = [r for r in rows if r["knob"] == knob]
        xs = [r["value"] for r in sub]
        ax.plot(xs, [r["distinct_2"] for r in sub], "o-", label="distinct-2 ↑多样性")
        ax.plot(xs, [r["rep4_rate"] for r in sub], "s-", label="rep4_rate ↑重复")
        ax2 = ax.twinx()
        ax2.plot(xs, [r["self_ppl"] for r in sub], "^--", color="gray", label="self_ppl ↓自洽")
        ax.set_xlabel(knob)
        ax.set_title(f"{knob} 扫描")
        ax.grid(alpha=0.3)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="best")
    fig.tight_layout()
    fig_path = Path("figures") / f"sampling_{run_dir.name}.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"已写 {fig_path}")


if __name__ == "__main__":
    main()
