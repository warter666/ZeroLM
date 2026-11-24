# ZeroLLM

From zero to LLM:从零开始(不依赖任何框架内置的 Transformer 实现)训练一个中文小语言模型,作为 [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/) Assignment 1 的课程实践项目。

课程作业的原题是在英文 TinyStories 上复现一个 GPT 风格的语言模型;本项目把语料换成中文,把整条流水线——分词、模型、优化、训练、采样——全部亲手实现一遍,以检验课程内容是否真正掌握。

## 为什么是 TinyStories

英文原版 [TinyStories](https://arxiv.org/abs/2305.07759)(Eldan & Li, 2023)证明了一个关键结论:**当语料足够简单时,1–10M 参数的小模型也能生成连贯的叙事文本**。这让它成为小算力下学习语言模型训练的理想语料——不需要数据中心,一张消费级/入门级云 GPU 就能完整走完"数据 → 分词 → 预训练 → 采样"的全流程。

本项目使用其中文版本 [TinyStories-Zh-2M](https://huggingface.co/datasets/RobinChen2001/TinyStories-Zh-2M)(MIT):约 200 万条中文小故事,842MB。

## 模块划分

每个模块独立成文件、独立成测试,对应 CS336 assignment1 的组件划分:

| 模块 | 文件 | 内容 |
| --- | --- | --- |
| BPE 训练 | `zerollm/train_bpe.py` | 字节级 BPE:词型频次统计、增量合并、tie-break 规则(实现思路参考 Karpathy 的 [minbpe](https://github.com/karpathy/minbpe)) |
| 分词器 | `zerollm/tokenizer.py` | encode / 流式 encode_iterable / decode,special token 处理 |
| 模型 | `zerollm/model.py` | Linear / Embedding / RMSNorm / SwiGLU / RoPE / causal MHA / pre-norm Block / TransformerLM |
| 损失与优化 | `zerollm/loss_optimizer.py` | cross_entropy(数值稳定)、AdamW(decoupled weight decay)、warmup+cosine LR、梯度裁剪 |
| 训练组件 | `zerollm/training.py` | get_batch 随机批采样、checkpoint 存取 |
| 训练循环 | `zerollm/run_training.py` | CLI:断点续训、eval 记录 csv、自动 checkpoint |
| 生成 | `zerollm/generating.py` | temperature + top-k + top-p(nucleus)采样 |

全部模型组件(注意力、归一化、优化器)均为手写 PyTorch 实现,不调用 `nn.Transformer`、`torch.optim.AdamW` 等现成封装。

## 默认模型规模

| 超参 | 值 |
| --- | --- |
| vocab size | 10000 |
| context length | 256 |
| d_model | 512 |
| layers / heads | 4 / 16 |
| d_ff | 1344 |
| RoPE theta | 10000 |
| 参数量 | **22.70M**(不绑定);`--tie-embeddings` 后 **17.58M** |

参数量的账(实测 `model.num_parameters()`):

| 组成 | 参数 |
| --- | --- |
| token embedding | 5.12M |
| lm_head(默认独立) | 5.12M |
| 4 层 × (attn 1.05M + SwiGLU 2.06M + 2×RMSNorm) | 12.46M |
| **合计(默认)** | **22.70M** |

早先文档写的“≈17M”是**权重绑定**(lm_head 复用 embedding)时的数字 22.70−5.12=17.58M。代码默认不绑定,
所以真实是 22.70M。要 17M 就加 `--tie-embeddings`(注意:绑定时 embedding 必须用 std=0.02 初始化,
否则 logits 尺度爆炸、初始 loss 直接 44.7 —— 已内置处理)。

速度(本地 RTX 4060 Laptop 实测,bs32×ctx256,毫秒/步):默认 217 → `--sdpa` 112 → `--sdpa --compile` 83。
A10 约为 4060 的 2–3 倍,10k 步 5–8 分钟,30k 步 20 分钟量级。

## 快速开始

```bash
uv sync                       # 安装依赖(含本地包 zerollm)
uv run pytest tests/ -v       # 模块测试(59 个)

# 1. 下载语料
uv run python scripts/download_data.py

# 2. 训练 BPE(10k 词表,语料采样 200MB)
uv run python scripts/train_tokenizer.py

# 3. 全量编码为 token id(.npy,多进程,可断点续跑)
uv run python scripts/prepare_data.py

# 4. 本地训练(小模型冒烟)
uv run python -m zerollm.run_training --experiment-name smoke \
  --train-data data/zh_train.npy --valid-data data/zh_valid.npy --vocab-size 10000

# 5. 生成
uv run python scripts/generate.py --run-dir experiments/<name> --prompt "从前"

# 6. 画 loss 曲线(读 metrics.csv)
uv run python scripts/plot_loss.py --run-dir experiments/<name>

# 7. 采样旋钮对比实验(temperature / top-k / top-p)
uv run python scripts/sampling_experiment.py --run-dir experiments/<name>
```

云 GPU 一键流程(冒烟 → 数据 → 训练):`bash scripts/train_cloud_sys.sh --all`

## 目录

```
zerollm/        模型与训练库
scripts/        数据下载、BPE 训练、预处理、生成、画图、采样实验、上传、云训练脚本
tests/          模块测试
experiments/    训练产物(metrics.csv / config / checkpoint,大文件不入库)
data/           语料与 token id(不入库)
figures/        loss 曲线等
```

## 训练配方

- 优化器:AdamW,β=(0.9, 0.95),weight decay 0.01,梯度裁剪 max_norm 1.0
- 学习率:3e-4 峰值,200 步线性 warmup,cosine 衰减到 3e-5
- batch:32 × 256 tokens;混合精度:bf16 autocast(Ampere 及以上)
- 规模:31,000 步 ≈ 1 epoch(全量编码后 train 254.1M tokens)
- 加速:`--sdpa`(flash 内核)与 `--compile`,见上节实测
- 评估:每 500 步在验证集上算 loss(20 个 batch),与 train loss 一并写入 `metrics.csv`

## 采样旋钮实验

`zerollm/generating.py` 支持三个旋钮,作用和调节方式不同:

| 旋钮 | 改什么 | 特点 |
| --- | --- | --- |
| temperature | 分布**形状**:`logits / t`,t<1 变尖、t>1 变平 | 同时影响候选集合大小和相对概率 |
| top-k | **候选数量固定**为 k 个 | 不随分布形状变化,分布很尖时仍会带上 k−1 个坏候选 |
| top-p | 保留累计概率达到 p 的**最小集合** | 候选数量自适应:分布尖时少、平时多 |

`scripts/sampling_experiment.py` 做对比实验:固定同一前缀与 RNG 种子,一次只动一个旋钮,
统计连贯性(self_ppl / EOS 命中率 / 有效字符率)、多样性(distinct-1/2/3)、
重复模式(重复 4-gram 占比 / max 4-gram 频次),输出 `metrics.csv`、`samples.md` 和三旋钮曲线图。

结论模板(对本仓库 checkpoint 实测后填):对这个 checkpoint,top-p 比固定 top-k 更灵活,
temperature 到 2.0 就基本崩坏。

## 参考与致谢

- [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/) Assignment 1:模块划分、bpe/模型/优化的接口与测试基准
- Andrej Karpathy 的 [minbpe](https://github.com/karpathy/minbpe):字节级 BPE 的实现思路(词型计数、增量 pair 统计、merge 顺序与特殊 token 处理)
- Andrej Karpathy 的 [nanoGPT](https://github.com/karpathy/nanoGPT):pre-norm + RoPE 结构、AdamW 超参、warmup+cosine 训练配方
- [TinyStories](https://arxiv.org/abs/2305.07759)(Eldan & Li, 2023):语料设计与小模型可行性结论
- [TinyStories-Zh-2M](https://huggingface.co/datasets/RobinChen2001/TinyStories-Zh-2M)(MIT):中文语料
