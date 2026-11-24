# ZeroLLM

从零训练一个中文小语言模型。路线对齐 [Hurricane0698/TransformerLM-from-scratch](https://github.com/Hurricane0698/TransformerLM-from-scratch)(CS336 assignment1 的 TinyStories 复现),语料换成中文。

## 语料

- [RobinChen2001/TinyStories-Zh-2M](https://huggingface.co/datasets/RobinChen2001/TinyStories-Zh-2M)(MIT):~200 万条中文小故事,parquet,约 842MB,单列 `text`。
- 英文原版 TinyStories 证明 1–10M 参数模型可在受限语料上学到连贯叙事;中文版沿用同一思路。

## 模块划分(对应 `zerollm/`)

| 模块 | 文件 | 内容 |
| --- | --- | --- |
| BPE 训练 | `train_bpe.py` | 改编自本人 `cs336/assignment1-basics` 已通过官方测试的实现 |
| 分词器 | `tokenizer.py` | encode / encode_iterable(流式)/ decode,special token 处理 |
| 模型 | `model.py` | Linear/Embedding/RMSNorm/SwiGLU/RoPE/MHA/TransformerBlock/TransformerLM |
| 损失与优化 | `loss_optimizer.py` | cross_entropy、AdamW、warmup+cosine LR、梯度裁剪 |
| 训练组件 | `training.py` | get_batch 数据加载、checkpoint 存取 |
| 训练循环 | `run_training.py` | CLI:断点续训、eval 记录 csv、checkpoint |
| 生成 | `generating.py` | top-p / temperature 采样 |

## 默认模型规模

vocab 10000 / context 256 / d_model 512 / 4 层 / 16 头 / d_ff 1344 / RoPE theta 10000 ≈ 17M 参数。

## 快速开始

```bash
uv sync                       # 安装依赖(含本地包 zerollm)
uv run pytest tests/ -v       # 模块测试

# 1. 下载语料
uv run python scripts/download_data.py

# 2. 训练 BPE(10k 词表)
uv run python scripts/train_tokenizer.py

# 3. 全量编码为 token id(.npy,可断点续跑)
uv run python scripts/prepare_data.py

# 4. 本地冒烟训练(RTX 4060 8GB 可跑小步数)
uv run python -m zerollm.run_training --experiment-name smoke ...

# 5. 云 GPU 完整训练(见 scripts/train_remote.py)
```

## 目录

```
zerollm/        模型与训练库
scripts/       数据下载、BPE 训练、预处理、生成、远程训练
tests/         模块测试
experiments/   训练产物(metrics.csv / config / checkpoint,大文件不入库)
data/          语料与 token id(不入库)
figures/       loss 曲线等
```

## 提交节奏

每完成一个模块提交一次;提交信息用 `module: <模块名>` 前缀。
