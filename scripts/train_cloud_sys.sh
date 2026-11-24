#!/usr/bin/env bash
# 云端跑法 B:直接用镜像自带的 PyTorch(不装 uv、不重下 torch)。
# 适用:云平台选了 PyTorch 镜像(torch 已装、CUDA 与驱动匹配)。
#
#   bash scripts/train_cloud_sys.sh smoke    # 验证云端 GPU
#   bash scripts/train_cloud_sys.sh prep     # 全量数据
#   bash scripts/train_cloud_sys.sh train    # 正式训练
#
# 与 train_cloud.sh 的区别:全部用系统 python,依赖只补 4 个小包。
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

STAGE="${1:---all}"
MIRROR="${PIP_MIRROR:-}"

echo "== GPU / torch 自检 =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
  echo "没有 nvidia-smi,这台机器没有可用 GPU"; exit 1; }
python - <<'PY'
import torch, sys
print("python", sys.version.split()[0])
print("torch ", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu   ", torch.cuda.get_device_name(0),
          f"{torch.cuda.get_device_properties(0).total_memory/2**30:.0f}GB")
else:
    print("!! torch 看不到 CUDA:镜像里的 torch 与驱动不匹配,换镜像或改用 uv 自装(torch 需驱动 >= 580)")
PY

echo "== 补依赖(只装 4 个小包,不碰 torch) =="
if [ -n "$MIRROR" ]; then
  pip install -q -i "$MIRROR" regex tqdm pyarrow numpy
else
  pip install -q regex tqdm pyarrow numpy
fi

prep() {
  python scripts/download_data.py --valid-ratio 0.01
  python scripts/train_tokenizer.py --vocab-size 10000 --sample-mb 200
  python scripts/prepare_data.py --workers "$(nproc)"
}

train() {
  python -m zerollm.run_training \
    --experiment-name "zh-10k-$(date +%m%d-%H%M)" \
    --train-data data/zh_train.npy --valid-data data/zh_valid.npy \
    --vocab-size 10000 --context-length 256 \
    --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 --rope-theta 10000 \
    --lr 3e-4 --min-lr 3e-5 --weight-decay 0.01 --beta1 0.9 --beta2 0.95 \
    --batch-size 32 --num-iters 10000 --warmup-iters 200 \
    --eval-interval 100 --eval-iters 20 --save-interval 500 \
    --device cuda
}

smoke() {
  python scripts/download_data.py --max-docs 2000
  python scripts/train_tokenizer.py --vocab-size 2000 --sample-mb 20 --out-dir data/tokenizer_smoke
  python scripts/prepare_data.py --tokenizer-dir data/tokenizer_smoke --out-dir data/smoke \
    --limit-docs 500 --workers 4 --skip-valid
  python -m zerollm.run_training --experiment-name smoke --run-root experiments \
    --train-data data/smoke/zh_train.npy --valid-data data/smoke/zh_train.npy \
    --vocab-size 2000 --context-length 64 --d-model 128 --num-layers 2 --num-heads 4 --d-ff 256 \
    --batch-size 16 --num-iters 30 --warmup-iters 5 --eval-interval 10 --eval-iters 2 \
    --save-interval 30 --log-interval 5 --device cuda
  python scripts/generate.py --run-dir experiments/smoke --tokenizer-dir data/tokenizer_smoke \
    --prompt "从前" --num-samples 1 --max-new-tokens 30 --device cuda
  echo "冒烟通过(系统 python + 镜像自带 torch)。"
}

case "$STAGE" in
  smoke) smoke ;;
  prep)  prep ;;
  train) train ;;
  --all) prep; train ;;
  *) echo "用法:$0 [smoke|prep|train|--all]"; exit 1 ;;
esac