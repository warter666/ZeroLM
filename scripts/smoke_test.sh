#!/usr/bin/env bash
# 端到端冒烟:下载少量文档 → 训小词表 → 编码 → 训练 30 步 → 生成。
# 用法(仓库根目录):
#   bash scripts/smoke_test.sh                # CPU,约几分钟
#   DEVICE=cuda bash scripts/smoke_test.sh    # 用本地 GPU
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cpu}"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

echo "== 1/5 下载 2000 篇文档 =="
uv run python scripts/download_data.py --max-docs 2000

echo "== 2/5 训练 2000 词表(取 20MB 样本) =="
uv run python scripts/train_tokenizer.py --vocab-size 2000 --sample-mb 20 --out-dir data/tokenizer_smoke

echo "== 3/5 编码(只用前 500 篇) =="
uv run python scripts/prepare_data.py \
  --tokenizer-dir data/tokenizer_smoke --out-dir data/smoke \
  --limit-docs 500 --workers 4 --skip-valid

echo "== 4/5 训练 30 步(小模型) =="
uv run python -m zerollm.run_training \
  --experiment-name smoke --run-root experiments \
  --train-data data/smoke/zh_train.npy --valid-data data/smoke/zh_train.npy \
  --vocab-size 2000 --context-length 64 --d-model 128 --num-layers 2 --num-heads 4 --d-ff 256 \
  --batch-size 8 --num-iters 30 --warmup-iters 5 --eval-interval 10 --eval-iters 2 \
  --save-interval 30 --log-interval 5 --device "$DEVICE"

echo "== 5/5 生成 =="
uv run python scripts/generate.py \
  --run-dir experiments/smoke --tokenizer-dir data/tokenizer_smoke \
  --prompt "从前" --num-samples 1 --max-new-tokens 30 --device "$DEVICE"

echo "冒烟通过。"