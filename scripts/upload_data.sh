#!/usr/bin/env bash
# 本地把「已经处理好的数据」推到云机,云端就完全跳过 download/tokenizer/prepare 三步 CPU 活。
#
#   bash scripts/upload_data.sh user@host:~/ZeroLM
#   bash scripts/upload_data.sh user@host:~/ZeroLM -p 2222      # 自定义 SSH 端口
#   DRY=1 bash scripts/upload_data.sh user@host:~/ZeroLM       # 只打印不传输
#
# 传的东西(约 0.8GB):
#   data/tokenizer/{vocab.json,merges.txt}   ~300KB
#   data/zh_train.npy                        ~760MB (uint16 token id)
#   data/zh_valid.npy                        ~8MB
# 不传 parquet / *.txt(约 2.3GB 可再生,云端不需要)。
set -euo pipefail
cd "$(dirname "$0")/.."

DEST="${1:-}"
shift || true
if [ -z "$DEST" ]; then
  echo "用法:$0 user@host:/remote/path [-p 端口]" >&2
  exit 1
fi
PORT_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    -p) PORT_ARGS=(-p "$2"); shift 2 ;;
    *) echo "未知参数:$1" >&2; exit 1 ;;
  esac
done

FILES=(data/tokenizer/vocab.json data/tokenizer/merges.txt data/zh_train.npy data/zh_valid.npy)
for f in "${FILES[@]}"; do
  [ -s "$f" ] || { echo "缺少 $f,先在本地跑完 ① 全流程" >&2; exit 1; }
done

HOST="${DEST%%:*}"
RPATH="${DEST#*:}"
[ "$RPATH" != "$DEST" ] || { echo "目标要写成 user@host:/path" >&2; exit 1; }

echo "== 本地校验和(到云端后可核对)=="
sha256sum "${FILES[@]}" | tee /tmp/zerollm_upload.sha256
echo "== 大小 =="
du -h "${FILES[@]}"

echo "== 建远端目录 =="
ssh "${PORT_ARGS[@]}" "$HOST" "mkdir -p '$RPATH/data/tokenizer'"

if [ "${DRY:-0}" = "1" ]; then
  echo "[DRY] 不传输。正式命令:"
  echo "rsync -avP -e 'ssh ${PORT_ARGS[*]}' ${FILES[*]} $HOST:$RPATH/data/"
  exit 0
fi

echo "== 传输(优先 rsync,断点续传)=="
if command -v rsync >/dev/null; then
  rsync -avP -e "ssh ${PORT_ARGS[*]}" \
    data/tokenizer/vocab.json data/tokenizer/merges.txt "$HOST:$RPATH/data/tokenizer/"
  rsync -avP -e "ssh ${PORT_ARGS[*]}" data/zh_train.npy data/zh_valid.npy "$HOST:$RPATH/data/"
else
  echo "本机没有 rsync,退回 scp(scp 不支持断点续传,大文件重传痛苦,建议装 rsync)"
  scp "${PORT_ARGS[@]}" data/tokenizer/vocab.json data/tokenizer/merges.txt "$HOST:$RPATH/data/tokenizer/"
  scp "${PORT_ARGS[@]}" data/zh_train.npy data/zh_valid.npy "$HOST:$RPATH/data/"
fi

echo "== 远端校验 =="
ssh "${PORT_ARGS[@]}" "$HOST" "cd '$RPATH' && sha256sum data/tokenizer/vocab.json data/tokenizer/merges.txt data/zh_train.npy data/zh_valid.npy"
echo
echo "核对上面的 sha256 与本地一致即可。云端接着跑:"
echo "  cd $RPATH && export PYTHONPATH=\$PWD"
echo "  python -m zerollm.run_training --experiment-name smoke ...   # 冒烟"
echo "  bash scripts/train_cloud_sys.sh train                        # 正式训练"
