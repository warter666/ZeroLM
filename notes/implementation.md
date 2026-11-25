# 实现细节

每个模块独立成文件、独立成测试。全部模型组件(注意力、归一化、优化器、BPE)均为手写 PyTorch,
不调用 `nn.Transformer`、`torch.optim.AdamW`、`nn.functional.cross_entropy` 等现成封装。

## 模块

| 模块 | 文件 | 内容 |
| ---------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| BPE 训练 | `zerolm/train_bpe.py` | 字节级 BPE:词型频次统计、增量 pair 统计、tie-break 规则(思路参考 Karpathy 的 [minbpe](https://github.com/karpathy/minbpe)) |
| 分词器 | `zerolm/tokenizer.py` | encode / 流式 `encode_iterable` / decode,special token 的匹配与边界处理 |
| 模型 | `zerolm/model.py` | Linear / Embedding / RMSNorm / SwiGLU / RoPE / causal MHA / pre-norm Block / TransformerLM |
| 损失与优化 | `zerolm/loss_optimizer.py` | cross_entropy(数值稳定)、AdamW(decoupled weight decay)、warmup + cosine LR、梯度裁剪 |
| 训练组件 | `zerolm/training.py` | `get_batch` 随机批采样、checkpoint 存取 |
| 训练循环 | `zerolm/run_training.py` | CLI:断点续训、eval 写 csv、自动 checkpoint、峰值显存记录 |
| 生成 | `zerolm/generating.py` | temperature + top-k + top-p(nucleus)采样、KV cache |

## 参数量

默认 `22.70M`,实测 `model.num_parameters()`:

| 组成 | 参数 |
| ---------------------------------------------- | ---------- |
| token embedding(10000 × 512) | 5.12M |
| lm_head(默认独立,512 × 10000) | 5.12M |
| 4 层 × (attn 1.05M + SwiGLU 2.06M + 2×RMSNorm) | 12.46M |
| **合计(默认)** | **22.70M** |

`--tie-embeddings` 让 lm_head 复用 embedding 矩阵,省掉一份 5.12M → **17.58M**。

**绑定时必须改初始化**:embedding 默认 std=1.0,绑定后 logits 尺度约为 `sqrt(d_model) × std`,
初始 loss 会飙到 44.7(正常应为 ln10000 ≈ 9.21),训不起来。代码因此在 `tie_embeddings=True`
时把 embedding 的 `init_std` 设为 0.02。

> 早期文档写的“≈17M”其实是权重绑定的数字,代码默认不绑定,真实是 22.70M。
> 顺带:砍掉 SwiGLU 的 w3(不少简化实现会这么做)会掉到 19.94M —— 那是**静默降质**,不是优化。

## 每步耗时(本地 RTX 4060 Laptop 8GB,bs32 × ctx256)

| 配置 | ms/step | token/s |
| ------------------ | ------- | ------- |
| 默认 | 217 | 37.8k |
| `--sdpa` | 112 | 73.2k |
| `--sdpa --compile` | 83–87 | 98.2k |

`--compile` 有个必须注意的陷阱:被 compile 的模块 state_dict 会带 `_orig_mod.` 前缀,
存档和 eval 必须用**原始模块**而不是 compile 后的包装,否则 checkpoint 存出来无法加载。
`run_training.py` 里用 `fwd`(可能被 compile)/ `model`(原始)两个名字区分这一点。

## 数据规模

| 阶段 | 结果 |
| -------- | -------------------------------------------------------------------- |
| 原始语料 | 3 个 parquet,842MB |
| 明文 | `zh_train.txt` 1.5GB(197.4 万篇)/ `zh_valid.txt` 15MB(2 万篇) |
| BPE | 10000 词表 / 9743 merges,语料采样 30MB,耗时 73.8 分钟,roundtrip 无损 |
| 压缩率 | **5.96 bytes/token**(20000 字样本实测;全量语料 5.9178) |
| 全量编码 | `zh_train.npy` **254.1M tokens / 485MB**(uint16),16 进程约 9 分钟 |

BPE 采样 30MB 而不是全量:实测 200MB 要 2–3 小时(纯 Python 单核,云端也不快),
而 10k 词表在 30MB 上已经收敛得够好。

## 测试

`uv run pytest tests/ -q` → **72 passed**。覆盖 BPE/tokenizer 的 roundtrip 与边界、
模型各组件形状与因果性、RoPE 性质、损失与优化器数值、训练组件、生成(含 KV cache 等价性)。
