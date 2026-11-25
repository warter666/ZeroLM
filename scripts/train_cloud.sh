#!/usr/bin/env bash
# 云 GPU 初始化 + 完整训练。租好机器、clone 本仓库后,在仓库根目录执行:
#   bash scripts/train_cloud.sh
#
# 分步跑(推荐先冒烟再全量):
#   bash scripts/train_cloud.sh --stage smoke
#   bash scripts/train_cloud.sh --stage prep
#   bash scripts/train_cloud.sh --stage train
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${1:---all}"

# 云端索引:默认官方 PyPI;国内机器可 export UV_DEFAULT_INDEX=镜像
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.org/simple}"

install_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
}

check_gpu() {
  echo "== GPU 检查 =="
  nvidia-smi || { echo "没有 nvidia-smi,请确认驱动已装"; exit 1; }
  uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
}

do_smoke() {
  DEVICE=cuda bash scripts/smoke_test.sh
}

do_prep() {
  echo "== 下载全量语料 =="
  uv run python scripts/download_data.py --valid-ratio 0.01
  echo "== 训练 10k BPE(取 200MB 样本) =="
  uv run python scripts/train_tokenizer.py --vocab-size 10000 --sample-mb 200
  echo "== 全量编码(多进程,可断点续跑) =="
  uv run python scripts/prepare_data.py --workers "$(nproc)"
}

do_train() {
  echo "== 训练(默认配置,见 README) =="
  uv run python -m zerolm.run_training \
    --experiment-name "zh-10k-$(date +%m%d-%H%M)" \
    --train-data data/zh_train.npy --valid-data data/zh_valid.npy \
    --vocab-size 10000 --context-length 256 \
    --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 --rope-theta 10000 \
    --lr 3e-4 --min-lr 3e-5 --weight-decay 0.01 --beta1 0.9 --beta2 0.95 \
    --batch-size 32 --num-iters 10000 --warmup-iters 200 \
    --eval-interval 100 --eval-iters 20 --save-interval 500 \
    --device cuda
}

install_uv
check_gpu
case "$STAGE" in
  smoke) do_smoke ;;
  prep)  do_prep ;;
  train) do_train ;;
  --all)
    do_prep
    do_train
    ;;
  *) echo "用法:$0 [smoke|prep|train|--all]"; exit 1 ;;
esac