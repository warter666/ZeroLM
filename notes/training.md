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

本地 RTX 4060 Laptop(8GB)的两次完整训练:

| 指标 | 1 epoch | 3 epoch |
| ------------------- | ---------------- | --------------------------- |
| 步数 | 31,000 | 93,000 |
| 唯一 token | 254.1M | 254.1M(重复 3 遍,共 761.9M) |
| 耗时 | 45 分钟 | 130 分钟(83 ms/步) |
| 初始 loss | 9.25(≈ ln 10000) | 9.25 |
| **best valid loss** | 2.2711 @ 28,500 | **2.1004 @ 90,000** |
| 末次 valid loss | 2.2812 | 2.1320 |
| bits/token | 3.2765 | 3.0302 |
| bits/byte | 0.5537 | 0.5121 |

曲线见 `figures/zh-3ep.png`(上:train 原始 + EMA、valid;下:cosine LR),1 epoch 版见 `figures/zh-1ep.png`。
前两个 epoch valid 全程位于 train 下方;末尾 6 次 eval 中 valid 从 2.1004 回升到 2.13、
连续 4 次上升,说明 3 个 epoch 的预算已经用尽,但幅度小(约 1.5%),尚不构成明显过拟合。
**继续堆 epoch 的收益已很小,下一步该试容量或数据,而不是步数。**

## tie-embeddings 消融(`experiments/zh-3ep-tie`)

同 seed、同 3 epoch、同所有超参,**只切 `--tie-embeddings`**(lm_head 与 token embedding 共享权重):

| 指标 | 未绑定(22.69M) | 绑定(17.57M) |
| ------------ | --------------- | ------------------- |
| 参数量 | 22.69M | **17.57M**(-5.12M) |
| best valid | 2.1004 @ 90,000 | **2.0438 @ 90,000** |
| 末次 valid | 2.1320 | **2.0789** |
| bits/byte | 0.5121 | **0.4982** |
| ms/step | 83 | **81** |
| GPU 峰值显存 | — | **1,767 MB** |

**三个维度全部胜出:更准、更快、更小。**

为什么参数少反而更好:本模型在 3 epoch 下仍属**欠拟合**(valid 全程低于 train,末尾才刚出现微弱抬头)。
在这种状态下,共享 embedding 相当于对词嵌入做了正则约束——输入侧和输出侧共享同一套语义空间,
两边梯度互相加强,泛化优于单纯多 5.12M 自由参数带来的容量。
若模型已经过拟合,这个结论会反过来。

一个必预的坑:**tie 时初始化标准差必须改小**。本项目默认 `init_std=1.0`,
绑定后 logits 尺度会爆炸(初始 loss 44.7,正常应接近 ln 10000 = 9.21);
所以 tie 分支用 `init_std=0.02`。两个 run 的初始 train loss 分别是 9.2479(未绑定)与 9.3218(绑定),
差异正来自这个初始化差异,后续会收敛掉。

曲线对比见 `figures/tie_ablation.png`(两条 valid 曲线同图,横轴从 step 15,000 起放大)。

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
