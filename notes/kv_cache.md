# KV cache:实现、正确性与实测

## 问题

不加 cache 时,每生成一个 token 都要把**整段前缀**重新前向一遍:

- 生成 200 token = 200 次长度 256 的前向,复杂度 O(n²);
- 实际代价随生成长度线性增长,**每次都用了一整窗的计算量,只为产出一个 token**。

加 cache 后:1 次 prefill + 200 次**长度 1** 的 decode。

## 实现

改动落在 `zerolm/model.py`(`use_cache` 参数,默认 `False`)与 `zerolm/generating.py`(默认开)。

三处要点,每一处写错都会静默给出错误结果:

1. **缓存的是已经旋转过的 k/v**。RoPE 只对**新 token** 施加,位置用绝对位置
   (`offset = past_len`)。如果对已缓存的 k 再旋转一次,attention 分数立刻错,但形状完全正常 ——
   这是最典型的写法错误。
2. **mask 从 `[T, T]` 扩成 `[T_new, past + T_new]`**,即列 `j <= past_len + 行 i` 才可见
   (`_offset_causal_mask`)。SDPA 的 bool mask 语义是 **True = 参与**。
3. **位置不能超过 `context_length`**。RoPE 的 cos/sin 是预计算表,长度就是 `context_length`;
   绝对位置一越界就取到表外。

## 两个启用前提(否则自动退回整窗重算)

```python
cache_ok = use_cache and max_len <= context_size and all(len(e) == max_len for e in encoded)
```

1. **无 padding**。现有 `generate_ids` 用右填充(pad id=0)对齐不等长 prompt。
   pad 会真的进入 attention,污染同批里较长的 prompt —— 而右填充下"谁被污染"还取决于长度差,
   非常难察觉。所以只要批内长度不等就关掉 cache。
2. **绝对位置 < `context_size`**。超出后改为整窗重算(位置重新从 0 数),
   代价回到 O(n²),但**结果与 cache 路径逐 token 相同**。

这个回退不是权宜之计,而是等价性的保证:两条路径在数学上是同一个函数。

## 正确性验证

`tests/test_generating.py::TestKVCache` 六条:

| 测试 | 断言 |
| ------------------------------------------ | ----------------------------------------------------------------------------- |
| `test_prefill_matches_full_forward` | prefill 的 logits 与整段前向 `allclose`(实测差 **0.0**) |
| `test_incremental_decode_matches_full` | decode 1 token 与整段重算最后一位 `allclose`(差 **5.4e-07**,仅内核路径浮点差) |
| `test_greedy_identical` | `top_k=1` 下两条路径 id 序列**完全相同** |
| `test_sampling_identical_with_same_seed` | 固定 seed 采样(top_p 0.9 / 0.95、t=0.8)id 序列**完全相同** |
| `test_unequal_length_batch_disables_cache` | 不等长批自动关 cache,与不开 cache 一致 |
| `test_falls_back_past_context_size` | 超 `context_size` 回退后仍一致 |

采样路径能不能逐 token 相同,取决于**随机数的消耗顺序**是否一致 —— 重构时把采样逻辑留在原位、
只把"下一步 logits"换成缓存版本,正是为了这一点。

## 实测

脚本 `scripts/bench_generate.py`:同一前缀、固定种子,`use_cache` **交替**跑 N 次取中位数。
交替而非连续,是为了抹掉"机器越跑越热"这类单向漂移;脚本还会断言两条路径输出逐 id 相同,
不一致就直接报错退出 —— 快但算错没有意义。

```bash
uv run python scripts/bench_generate.py --run-dir experiments/zh-1ep --device cpu --repeats 3
uv run python scripts/bench_generate.py --run-dir experiments/zh-1ep --device cuda --repeats 5 --batch 5
```

checkpoint `zh-1ep`(22.70M),200 new token,固定种子,`use_cache` 交替跑 5 次取中位数。
`--compile` 列控制是否用 `torch.compile(mode="default")` 包装模型(不用 cudagraphs,
因为带 cache 时每步序列长度都变,图会被覆盖):

| 设备 | 编译 | 配置 | 无 cache | 有 cache | 加速 |
| ---- | ---- | ---- | ------------------ | ------------------- | --------- |
| CPU | 否 | B=1 | 8.46s(23.7 tok/s) | 2.20s(91.0 tok/s) | **3.85×** |
| CPU | 否 | B=5 | 29.63s(33.8 tok/s) | 3.56s(281.2 tok/s) | **8.33×** |
| GPU | 否 | B=1 | 2.75s(72.7 tok/s) | 2.81s(71.2 tok/s) | **0.98×** |
| GPU | 否 | B=5 | 2.90s(344.8 tok/s) | 2.85s(350.6 tok/s) | **1.02×** |
| GPU | 是 | B=1 | 0.88s(227.9 tok/s) | 0.96s(209.0 tok/s) | **0.92×** |
| GPU | 是 | B=5 | 1.19s(672.3 tok/s) | 0.99s(1009.7 tok/s) | **1.20×** |

(编译后 `use_cache=True` 的第一次会比后续慢一个量级——那是 torch.compile 的编译时间,
取中位数时会被自动排除。)

**结论:cache 的收益 = 「省下的计算」/ 「总耗时」,而这个比值在小模型短序列上很小。**

### GPU 为什么没加速:用 profiler 查到的实情

实测(zh-1ep,B=1,单次 forward,`torch.profiler`):

| 指标 | 数值 |
| --------------------------------- | ----------------------------------- |
| 每步 **CPU 侧派发**时间 | **16.0 ms** |
| 每步 **GPU 实际计算**时间 | **1.32 ms** |
| 每步 kernel 启动次数 | 217(`aten` 元操作共约 1264 个) |
| 生成期间 GPU 利用率(每 0.5s 采样) | 均值 **18.6%**,71% 的采样点低于 20% |

**所以 GPU 确实没跑满**,但原因不在 GPU:PyTorch 逐 op 派发的开销(16 ms)
是 GPU 实际计算(1.3 ms)的 **12 倍**。GPU 大部分时间在等 CPU 把下一个 op 发过来。

这直接决定了 cache 的上限:cache 只能省掉那 1.32 ms 里的一部分(`O(T²)` 的注意力重算),
也就是每步约 0.8 ms,占 16 ms 总耗时的 **4.8%** —— 理论加速上限只有 **1.05×**,
与实测的 0.98× 完全对得上。

一个能反证“瓶颈在 CPU 而非序列长度”的测量:**seq=1 和 seq=200 的 forward 几乎一样慢**
(12.0 ms vs 11.6 ms,`no_grad` 下)。序列长度差 200 倍而耗时不变,
说明耗时几乎全部来自与长度无关的固定开销。

### 编译之后:cache 的收益才露出来,但分批量

`torch.compile` 把 CPU 派发压下去后,整体生成快了约 **3.1×**(GPU B=1 2.75s → 0.88s)。
此时计算占比上升,cache 的收益才显形:

- **B=5 时 1.20×**(无可否认的加速:省下的计算终于能盖过 cache 自身的开销);
- **B=1 时 0.92×**(反而略慢:cache 每步要做 K/V 拼接与形状守卫,这点开销超过了省下的那点计算)。

趋势与 CPU 一致(批量越大,收益越大:CPU 3.85× → 8.33×;GPU 编译后 0.92× → 1.20×),
只是绝对值小很多。机制是同一个:**收益取决于“省下的计算”占总耗时的比例。**

### 到底能宣称什么

1. 与不开 cache 相比,输出**逐 id 完全相同**(所有配置下都断言过);
2. 每步 attention 计算 `O(T²)` → `O(T)`,decode 复杂度 `O(T²)` → `O(T)`;渐近结论,与设备无关;
3. 实测:CPU **3.85× / 8.33×**;GPU **0.92×–1.20×**,且**必须注明是否编译**
   —— 不编译时瓶颈在 CPU 派发,cache 的优势根本显现不出来。

**不要宣称“GPU 上 10–20×”。** GPU 上真正的大头是 `torch.compile`(3.1×),
不是 KV cache;两者是不同层次的优化,别混为一谈。

复现:`--compile` 已加到基准脚本上。

## 顺带的收益

采样扫参(`sampling_experiment.py`)是 3 组旋钮 × 多个取值 × 3 前缀 × 5 样本 × 200 token,
每个 cell 都是一次完整生成。cache 让单 cell 从 ~30 秒降到 ~3.5 秒,扫参才变得可以反复跑。
