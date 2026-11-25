# 交接日志

> 最新在最上面。每条写清「做了什么 / 当前状态 / 下一步 / 坑」。

## 2026-11-25 · 收尾:图入库、top_k 测试、bits/byte、LICENSE、锁文件

**做了什么**
- README 贴图:`figures/zh-1ep.png`(loss + cosine LR)进「训练结果」节,
  `figures/sampling_zh-1ep.png` 进「采样旋钮实验」节;删掉 benchmark 临时图,两图共 240KB 入库。
- 补 `top_k` 测试(之前只测 top_p/temperature):`top_k=1` 等价贪心、候选数恒等 k 且 `std==0`、
  top-p 的 `std>0` 作对照、top-k+top-p 叠加、非法参数、`generate_ids` 的 prompt 长度。测试 59 → **66**。
- 新增 `scripts/report.py`:一键报 best valid / bits-token / bytes-token / bits-byte。
  本仓库 **bits/byte = 0.5537**(2.2711 / ln2 / 5.9178),等价压缩 14.45 倍。
  README 写清口径:英文 val 1.36 与中文 2.2711 不是同一把尺子,跨分词器必须折成 bits/byte。
- 新增 `LICENSE`(MIT © 2026 warter666)与 README「许可」节。
- **提交了 `uv.lock`**:之前一直是旧状态(`requires-python >=3.12` + tsinghua 源)与 pyproject 不一致,
  `uv lock --check` 验证一致后再提交,以后 `uv sync` 不会再产生 1900 行漂移。

**当前状态**
- 工作区干净,66 测试全绿,8 个 commit。
- 全流程产物齐备:数据 / BPE / 1 epoch 训练 / 生成 / loss 曲线 / 采样实验与结论 / 图入库。

**下一步(可选,按收益排序)**
1. 再训 1–2 epoch(每轮约 45 分钟,loss 尾部仍在降),或加 `--tie-embeddings`(17.58M)做消融。
2. 上 A10 试更大模型:`bash scripts/upload_data.sh <user>@<host>:~/ZeroLLM` 只传 0.5GB。
3. 可选工程项:生成加 KV cache(现在每步重算整个上下文)、加 GitHub Actions 跑 pytest。

**坑**
- 多 epoch 前注意代码里**没有 dropout**,虽然目前 valid 一直在 train 下方(未过拟合),但多轮后要看紧。
- 改 `--sample-mb` / 改配置重跑时,旧的 `*.state.json`、`sample.txt` 会被复用,先清目录。

## 2026-11-25 · 训练完成 1 epoch,采样实验与结论落地

**做了什么**
- 续训到 **31000 步(= 1 epoch,254.1M tokens)**:`best valid 2.2711 @ 28500`,末次 2.2813。
- 采样旋钮实验(替代之前的“模板”):`scripts/sampling_experiment.py` 三组扫描 × 3 前缀 × 5 样本,
  固定 seed。给 `generate_ids` 加了 `avg_candidates / std_candidates` 两个统计
  —— 这是“top-k 数量固定 / top-p 数量自适应”的直接证据(top-k 全档 std=0.0)。
- 结论写进 `notes/sampling.md`(完整报告 + 样例)和 README「采样旋钮实验」节:
  - temperature 改**形状**:候选数 6212→10000,`self_ppl` 3.23→18424;`t=1.6` EOS 命中率掉到 0.73,`t=2.0` 只剩 0.13(崩坏)。
  - top-k 改**数量且固定**:候选数恒等于 k、std 0.0,全档 EOS 1.00;但 k 200→1000 时 distinct-2 不涨(0.962),ppl 9.95→11.77。
  - top-p 改**数量且自适应**:p=0.9 候选数 147.6±109.9(波动 74%),p=0.99 为 1795±756;同样不崩坏。
    相同多样性下更省:top-p=0.9(148 候选/d2 0.952/ppl 9.31) vs top-k=1000(1000 候选/d2 0.962/ppl 11.77)。
- `generate.py` 实测生成:连贯中文故事 + 正常 EOS 收尾,验收标准 5 达成。
- 修 `run_training` 续训时 tok/s 虚高(用了绝对 step,改成 `step - start_step`)。

**当前状态**
- **全流程已跑完**:语料 → BPE(5.96 B/token)→ 254.1M tokens → 1 epoch 训练(val 2.2711)→ 生成 → loss 曲线 → 采样实验与结论。
- 产物:`experiments/zh-1ep/`(checkpoint 31000)、`figures/`(loss + 三旋钮曲线)、`notes/sampling.md`。

**下一步(可选)**
1. 画 loss 曲线:`python scripts/plot_loss.py --run-dir experiments/zh-1ep`(已跑过一次,可重画)。
2. 继续训更多 epoch / 加大模型:4060 上 1 epoch 约 45 分钟;要更快的迭代就上 A10
   (`bash scripts/upload_data.sh <user>@<host>:~/ZeroLLM` 只传 0.5GB,云端 `train_cloud_sys.sh train`)。
3. `--tie-embeddings`(17.58M)做一次消融对比。

**坑**
- 续训必须保持 `--num-iters` / `--sdpa` / `--compile` 与首段一致,否则 LR 曲线断、state_dict 前缀对不上。
- matplotlib 画中文要显式设 `font.sans-serif`(WenQuanYi Micro Hei),否则全是豆腐块 —— 已在脚本里设好。
- `self_ppl` 天然偏爱重复文本,必须和 distinct-n 一起看(反例:top-k=1 贪心 ppl 2.37 最低但多样性也最低)。

## 2026-11-24 深夜 · 数据全跑完 + 本地训到 28200 步(后续已续完,见上条)

**做了什么**
- 语料:3 个 parquet(842MB)→ `data/zh_train.txt`(1.5GB,197.4 万篇)+ `zh_valid.txt`(15MB,2 万篇),20 秒。
- BPE:`vocab 10000 / merges 9743`,**73.8 分钟**,roundtrip OK,**5.96 bytes/token**(30MB 采样)。
- 编码:`zh_train.npy` **254.1M tokens / 485MB**(uint16),`zh_valid.npy` 2.6M tokens,16 进程约 9 分钟。
- 训练:`experiments/zh-1ep`,31000 步 = 1 epoch,`--sdpa --compile` 后 **87ms/step**(4060 Laptop,62k tok/s)。
  - 训练中 val loss:9.25(初始化) → 6.50@50 → 2.69@6000 → **2.285@28000**。
  - **已手动停在第 28200 步**(用户要求暂停);`checkpoint.pt` 停在 **28000** 步,259.8MB。
- 代码 4 个 commit(见 `git log`):A10 适配 / 数据链路可观测 / 采样实验 / 本交接。

**当前状态**
- 本地可完整跑通全流程,唯一未做:训到 31000 步收尾、采样旋钮实验、生成样例。
- `experiments/zh-1ep/` 里有 config.json / metrics.csv / checkpoint.pt(28000)。

**下一步**
1. 续训到 1 epoch(还有 2200 步,约 3 分钟):
   `python -m zerollm.run_training --experiment-name zh-1ep ...同上参数... --checkpoint-in experiments/zh-1ep/checkpoint.pt`
   (必须保持 `--num-iters 31000` 与 `--sdpa --compile` 一致,否则 LR 曲线断、state_dict 不匹配)
2. `python scripts/sampling_experiment.py --run-dir experiments/zh-1ep` → 跑三旋钮对比,
   把真实数值填进 README 的「采样旋钮实验」节和 `experiments/sampling-zh-1ep/samples.md`。
3. `python scripts/generate.py --run-dir experiments/zh-1ep --prompt "从前"` 看生成效果;
   `python scripts/plot_loss.py --run-dir experiments/zh-1ep` 出 loss 曲线。
4. 要上云:只有 0.5GB 的 npy + tokenizer 需要传 → `bash scripts/upload_data.sh <user>@<host>:~/ZeroLLM`,
   云端 `bash scripts/train_cloud_sys.sh train`(prep 完全跳过;A10 上 31k 步约 30 分钟)。

**实测数据(可复用)**
- 4060 Laptop 稳态:base 217ms/step → `--sdpa` 112ms → `--sdpa --compile` **83-87ms**;峰值显存 bs32 **2.12GB**。
- bs64 吞吐不变(无收益),bs128 在 8GB 上掉到 20.9k tok/s(颠簸) → **8GB 卡就用 bs32**。
- A10 约为 4060 的 2–3 倍,31k 步 30 分钟量级;原方案「3–4 小时」高估约 30 倍。
- 纯 Python BPE 成本随语料近线性涨:5.2MB+2k merges=196s;30MB+10k merges=73.8min;200MB 估 2–3 小时。

**坑**
- `--compile` 会把 `state_dict` 键加 `_orig_mod.` 前缀 → 存出的 checkpoint 无法被未 compile 模型加载(续训/生成全挂)。已修:forward 用 fwd,存档/eval 用原始 model。
- 权重绑定(`--tie-embeddings`)必须配 `std=0.02` 的 embedding 初始化;用本仓库默认 std=1.0 会让 logits 尺度爆炸,初始 loss 44.7。已内置处理。
- `print` 重定向到文件是块缓冲:**日志看起来卡住其实在跑**,以 `metrics.csv` 为准(已给 print 加 flush)。
- `pkill -f train_tokenizer.py` 会把发起命令的 bash 自己也匹配杀掉(命令行含同串),用 `setsid` + 启动脚本。
- `train_tokenizer.sample_text` 发现 `sample.txt` 已存在就复用:**改 `--sample-mb` 前必须先 rm -rf data/tokenizer**。
- 断点续训不能改 `--num-iters`/`--sdpa`/`--tie-embeddings`;分段用 `--stop-at`。

**遗留**
- `uv.lock`(上次会话切 python 版本的遗留改动)与 `figures/`(benchmark 产物)未提交,用户未表态。
- 词表压缩率 5.96 B/token 比早期冒烟(4.16)好很多,所以全量只有 254M tokens,不是先前估的 380M。


## 2026-11-24 · 全链路冒烟通过,等云机器开训

**做了什么**
- 7 个核心模块 + 数据脚本 + 测试(59 绿)+ 两次端到端冒烟(CPU 与 CUDA 都通过):
  - 下载 2000 篇 → 2k 词表(67s,压缩率 4.16 bytes/token)→ 500 篇编码(0.1M tokens)
  - → 30 步训练 loss 7.68→7.37(CUDA,4060)→ checkpoint 加载生成正常。
- 5 次提交,仓库干净(实验产物已 ignore)。

**当前状态**
- 本地全部就绪。data/ 807MB(2 个完整 parquet + 冒烟产物),云端需要全量 3 个。

**下一步(用户开云机器后)**
1. clone 仓库(或 scp),`bash scripts/train_cloud.sh --stage smoke` 先验证云端 GPU。
2. `--stage prep`:全量下载 → 10k BPE(取 200MB 样本,预计几分钟到十几分钟)→ 全量编码(2M 篇,多进程;~1.4G tokens 预计 30-60 分钟,可断点续跑)。
3. `--stage train`:默认 10k 步 / bs32 / d512·4L·16H ≈ 17M 参数。参考:英文版 B200 用 24 分钟(225k tok/s);L4/A10 量级的卡预计 2-6 小时。
4. 训完 `scripts/generate.py --run-dir experiments/<name> --prompt "从前"`;把 experiments/<name>/metrics.csv 拉回来画 loss 曲线。
5. 若要在 GitHub 建远端仓库:`git remote add origin <url> && git push -u origin main`。

**坑**
- uv 必须走镜像:WSL `export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple`;
  云在国外机器用默认官方源即可。torch 2.15 + cu132 轮子 900MB+。
- tokenizer._safe_cut:special token 恰好结束在缓冲区末尾时要整段编出,否则被正则拆碎。
- prepare_data 的 worker initargs 必须带语料路径(修过一次)。
- 中文 1 字 = 3 字节;10k 词表压缩率约 4.16 bytes/token(2k 词表冒烟值),预估 token 总量按这个除。
- AdamW 顺序:moment → decoupled decay → 参数更新,测试与 CS336 参考式逐位对齐。

## 2026-11-24 · 项目立项 ZeroLLM

**做了什么**
- 定名 ZeroLLM(from zero to LLM),`/home/tbtb/projects/ZeroLLM`;init-project-wsl 接入共享技能。
- 参考仓库结构已摸清:cs336_basics 七模块 + modal 云训练;英文 TinyStories 17M 参数,val 1.36(B200)。
- 语料:RobinChen2001/TinyStories-Zh-2M(MIT,~2M 篇中文故事,842MB parquet)。

**坑**
- 课程仓库 AGENTS.md 只约束课程仓库;本仓库独立,不回写、不在课程仓库补作业。
- 本机 4060 8GB 只做冒烟;完整训练上云。

