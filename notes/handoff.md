# 交接日志

> 最新在最上面。每条写清「做了什么 / 当前状态 / 下一步 / 坑」。

## 2026-11-24 · 项目立项 ZeroLLM

**做了什么**
- 项目定名 ZeroLLM(from zero to LLM),目录 `/home/tbtb/projects/ZeroLLM`。
- 参考仓库 Hurricane0698/TransformerLM-from-scratch 已摸清结构:`cs336_basics/` 七模块(tokenizer / train_bpe / TransformerLM / loss_optimizer / Training / run_training / generating)+ `modal_tinystories.py` 云训练;英文 TinyStories 10k vocab / 256 ctx / d512 / 4L / 16H / d_ff 1344 ≈ 17M 参数,val loss 1.36(B200)。
- 语料选型:[RobinChen2001/TinyStories-Zh-2M](https://huggingface.co/datasets/RobinChen2001/TinyStories-Zh-2M)(MIT,~200 万条中文故事,842MB parquet,单列 text)。
- 骨架:init-project-wsl + AGENTS.md/README/pyproject/.gitignore,包名 `zerollm`。

**当前状态**
- 骨架已提交(首次 commit)。模块全部未开工。

**下一步**
1. 模块 1:`zerollm/train_bpe.py`(复制课程仓库已通过测试的实现并适配)。
2. 模块 2:`zerollm/tokenizer.py`(encode/encode_iterable/decode,special token)。
3. 依次:模型 → loss/optimizer → data/checkpoint → 训练循环 → 生成脚本。
4. 数据脚本(download / train_tokenizer / prepare_data)。
5. 云训练配置(用户开机器后再动)。

**坑**
- 课程仓库 `cs336/assignment1-basics` 的课程 AGENTS.md 规定 AI 只做助教;本仓库是独立项目,不受该规则约束,但不要在课程仓库里补作业或回写。
- 本机 RTX 4060 只有 8GB,完整训练按用户要求上云 GPU;本地只做冒烟。
- `data/` 有 800MB+ 语料,已在 `.gitignore`,别 `git add .` 加进去。
