"""从 checkpoint 加载模型,做中文故事续写。

用法:
    uv run python scripts/generate.py \
        --run-dir experiments/zh-10k \
        --tokenizer-dir data/tokenizer \
        --prompt "从前有一只小兔子" --num-samples 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zerolm.generating import generate
from zerolm.model import TransformerLM
from zerolm.tokenizer import Tokenizer
from zerolm.training import load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="包含 config.json 与 checkpoint 的训练目录")
    ap.add_argument("--checkpoint", default="checkpoint.pt")
    ap.add_argument("--tokenizer-dir", default="data/tokenizer")
    ap.add_argument("--special-tokens", default="<|endoftext|>")
    ap.add_argument("--prompt", default="从前")
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))

    tok = Tokenizer.from_files(
        str(Path(args.tokenizer_dir) / "vocab.json"),
        str(Path(args.tokenizer_dir) / "merges.txt"),
        [s for s in args.special_tokens.split(",") if s],
    )
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        # 老 run 的 config.json 没有这两个键,默认 False
        tie_embeddings=bool(config.get("tie_embeddings", False)),
        use_sdpa=bool(config.get("sdpa", False)),
    ).to(args.device)
    step = load_checkpoint(run_dir / args.checkpoint, model)
    model.eval()
    print(f"已加载 step {step} 的模型({model.num_parameters()/1e6:.2f}M 参数)\n")

    eos_id = None
    if "<|endoftext|>" in tok.special_tokens:
        eos_id = next(i for i, b in tok.vocab.items() if b == b"<|endoftext|>")

    outs = generate(
        [args.prompt] * args.num_samples,
        model,
        tok,
        device=args.device,
        context_size=config["context_length"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_id=eos_id,
    )
    for i, o in enumerate(outs, 1):
        print(f"--- 样本 {i} ---")
        print(o.strip())
        print()


if __name__ == "__main__":
    main()
