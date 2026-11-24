# 交接日志

> 最新在最上面。每条写清「做了什么 / 当前状态 / 下一步 / 坑」。

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

