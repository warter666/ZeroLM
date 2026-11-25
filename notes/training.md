# 训练配置与结果

## 配方

| 项 | 值 |
| ------------ | ------------------------------------------------------------------------ |
| 优化器 | AdamW,β=(0.9, 0.95) |
| weight decay | 0.01(加在非 bias/norm 参数上) |
| 梯度裁剪 | max_norm 1.0 |
| 学习率 | 3e-4 峰值 → cosine 衰减到 3e-5 |
| warmup | 200 步线性 |
| batch | 32 × 256 tokens |
| 精度 | bf16 autocast(Ampere 及以上) |
| 规模 | 31,000 步 ≈ 1 epoch(254.1M tokens) |
| 加速 | `--sdpa` + `--compile`(87 ms/step) |
| 评估 | 每 500 步在验证集上算 loss(20 个 batch),train/valid 一并写 `metrics.csv` |

## 1 个 epoch 的结果(`experiments/zh-1ep`)

本地 RTX 4060 Laptop(8GB)跑完 **31,000 步 / 254.1M tokens**,耗时 45 分钟。

| 指标 | 值 |
| --------------------- | --------------------------- |
| 初始 loss | 9.25(≈ ln 10000,随机初始化) |
| **best valid loss** | **2.2711 @ step 28,500** |
| 末次 valid loss | 2.2812 |
| bits/token | 3.2765 |
| bytes/token(语料实测) | 5.9178 |
| bits/byte | 0.5537 |

曲线见 `figures/zh-1ep.png`(上:train 原始 + EMA、valid;下:cosine LR)。
valid 全程位于 train 下方,1 个 epoch 没有过拟合迹象。

## 关于 bits/byte 的口径

`bits/byte = loss / ln2 / (语料字节数 / token 数)`,本仓库 `2.2711 / 0.6931 / 5.9178 = 0.5537`。

**它的含义要说清**:交叉熵是「该模型分布下编码这个 token 所需的期望信息量(bit)」,
折到每个原始字节上,等于**一个理想算术编码器在这段文本上的期望码长**。
这是熵意义上的估计,不是拿压缩工具实测出来的压缩率 —— 真实压缩还要加上模型权重体积、
编码器实现损耗等开销,不能说成"把语料压到了 1/14"。

它的用途只有一个:**换分词器后 loss 不可比**。词表越大,同一段文本的 token 越少、
每 token 的交叉熵通常越大,所以参考仓库的英文 val 1.36 与本仓库中文 2.2711 不是同一把尺子。
要比就得折到 bytes 这个共同的物理量上。

一键重算:`uv run python scripts/report.py --run-dir experiments/zh-1ep`。

## 多 epoch 与重复数据

1 epoch = 254M token,3 epoch = 762M token,即**同一份语料重复 3 遍**。
判断"重复是否已经变成过拟合"看 valid 曲线:

- valid 还在降 → 重复数据还没被背完,继续训有价值;
- valid 连续 2–3 次 eval(每 500 步一次)**稳定抬头** → 该停,而不是单点抖动就停。

代码里**没有 dropout**。1 epoch 下 valid 始终低于 train,是欠拟合,加 dropout 只会先掉一截
valid 并把基线搞脏。真到过拟合那步再加,位置是 Block 的两个 residual 之后,`p=0.1` 起。

## 断点续训的注意点

- 续训必须保持 `--num-iters` / `--sdpa` / `--tie-embeddings` 与首段**完全一致**:
  LR 调度按 `num_iters` 算,`--tie-embeddings` 会改变 state_dict 的键。
- 用 `--checkpoint-in` 续训并**修改** `--num-iters` 会造成 LR 跳变(相当于 warm restart),
  曲线上能看到一段先升后降。需要改变总预算时,更干净的做法是从零起一个新 run。
- 分段跑用 `--stop-at`,LR 仍按总 `--num-iters` 计算,曲线不会断。
- 吞吐打印用 `(step - start_step)`,否则续训时 `elapsed` 从 0 重算会得出荒谬的 tok/s。
- `--compile` 下存档必须用原始模块,否则 state_dict 带 `_orig_mod.` 前缀,续训和生成都加载失败。

## 显存

`run_training.py` 结束时会打印并落盘峰值显存(`peak_memory_mb.txt`),
用于支撑"单卡 8GB 能跑"这类说法。bs32 × ctx256 + `--sdpa --compile` 的实测占用见该文件。

## 云端

数据上传只需 0.5GB(tokenizer + 两个 `.npy`),`scripts/upload_data.sh` 做 rsync + 双向 sha256 校验。
A10 上单卡 31k 步约 30 分钟量级(4060 的 2–3 倍算力)。
