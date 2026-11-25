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

checkpoint `zh-1ep`(22.70M),200 new token,固定种子,`use_cache` 交替跑 5 次取中位数:

| 设备 | 配置 | 无 cache | 有 cache | 加速 |
| ---- | ---- | ------------------ | ------------------ | --------- |
| CPU | B=1 | 8.46s(23.7 tok/s) | 2.20s(91.0 tok/s) | **3.85×** |
| CPU | B=5 | 29.63s(33.8 tok/s) | 3.56s(281.2 tok/s) | **8.33×** |
| GPU | B=1 | 2.75s(72.7 tok/s) | 2.81s(71.2 tok/s) | **0.98×** |
| GPU | B=5 | 2.90s(344.8 tok/s) | 2.85s(350.6 tok/s) | **1.02×** |

**结论:cache 的收益取决于「省下的计算」在总耗时里占多大比重。**

- **CPU 上明显**(3.85× / 8.33×):CPU 做那个随 T 增长的窗口矩阵乘法很贵,
  每多生成一个 token 都要把整个前缀重算一遍;cache 把这部分省掉。
- **GPU 上几乎没有**(两个批量都在 1.0× 附近):4060 上 22.70M 模型处理长度 ≤200 的序列时,
  `O(T²)` 那部分重算只要零点几毫秒,而每步的固定开销(Python 循环、内核启动、同步)要 14 ms。
  **此时生成循环是“开销受限”而不是“计算受限”,省计算自然省不出时间。**

旁证:GPU 上有 cache 是 2.81 s / 200 token ≈ **14 ms/token**,和 CPU 的 11 ms/token 同量级;
如果 GPU 真在跑计算,这两个数应该差一到两个数量级 —— 说明此刻瓶颈根本不在设备算力上。

**所以不要宣称“GPU 上 10–20×”。** 能站住的表述只有三条:

1. 与不开 cache 相比,输出**逐 id 完全相同**;
2. 每步 attention 的计算量从 `O(T²)` 降到 `O(T)`,decode 复杂度从 `O(T²)` 降到 `O(T)`
   —— 这是渐近结论,与设备无关;
3. 实测加速:CPU **3.85× / 8.33×**;GPU 在当前配置(短序列、小批量、Python 逐 token 循环)下 **~1.0×**。

要在 GPU 上真看到收益,得先消掉每步固定开销(CUDA graphs、静态 shape),
或者把 batch × context 放大到计算主导的量级。

## 顺带的收益

采样扫参(`sampling_experiment.py`)是 3 组旋钮 × 多个取值 × 3 前缀 × 5 样本 × 200 token,
每个 cell 都是一次完整生成。cache 让单 cell 从 ~30 秒降到 ~3.5 秒,扫参才变得可以反复跑。
