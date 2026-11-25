# KV cache:实现、正确性与实测

## 问题

不加 cache 时,每生成一个 token 都要把**整段前缀**重新前向一遍:

- 生成 200 token = 200 次长度 256 的前向,复杂度 O(n²);
- 实际代价随生成长度线性增长,**每次都用了一整窗的计算量,只为产出一个 token**。

加 cache 后:1 次 prefill + 200 次**长度 1** 的 decode。

## 实现

改动落在 `zerollm/model.py`(`use_cache` 参数,默认 `False`)与 `zerollm/generating.py`(默认开)。

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

checkpoint `zh-1ep`(22.70M),200 new token,CPU:

| 配置 | 无 cache | 有 cache | 加速 |
| ---- | -------------------- | -------------------- | --------- |
| B=1 | 8.46s(23.7 token/s) | 2.20s(91.0 token/s) | **3.85×** |
| B=5 | 29.63s(33.8 token/s) | 3.56s(281.2 token/s) | **8.33×** |

**加速比与批量、设备强相关,不能只报一个数。** CPU 上每步的固定开销(Python 调度、内核启动)
占比很高,B=1 时这部分开销摊不下去,cache 省下的矩阵计算占比就小;B=5 摊薄后升到 8.33×。
GPU 上固定开销小、decode 的单 token 前向又便宜,加速比应在两位数 —— 具体数字待训练腾卡后补测,
**在此之前不要引用"10–20×"这个区间**。

## 顺带的收益

采样扫参(`sampling_experiment.py`)是 3 组旋钮 × 多个取值 × 3 前缀 × 5 样本 × 200 token,
每个 cell 都是一次完整生成。cache 让单 cell 从 ~30 秒降到 ~3.5 秒,扫参才变得可以反复跑。
