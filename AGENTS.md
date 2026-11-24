# ZeroLLM

## 目标与验收

- 目标：按 CS336 assignment1 的模块划分,从零训练一个中文小语言模型。语料为中文 TinyStories(~200 万条简短故事),BPE 分词沿用 `cs336/assignment1-basics` 已完成的 `train_bpe.py`,模型/训练/采样模块参考 Hurricane0698/TransformerLM-from-scratch 的结构重写,最终在云 GPU 上完成一次完整训练,能采样生成连贯中文小故事。
- 验收:
  1. BPE 训练在中文语料上产出 `vocab.json` + `merges.txt`,roundtrip(编码→解码)无损;
  2. 全量数据预处理(语料→token id 的 `.npy`)可断点续跑;
  3. 训练脚本支持断点续训、eval 记录(csv)、checkpoint;
  4. 云 GPU 训练 ~10k 步后 val loss 收敛(参考仓库英文版 ~1.36,中文版量级相当即可);
  5. `generate.py` 用 prompt 采样出可读的中文故事。

## 技术与运行

- 技术栈:Python 3.12 + uv、PyTorch(CUDA)、numpy、regex、einops、jaxtyping、tqdm、matplotlib;云侧另加 modal(可选)。
- 模型规模(默认):vocab 10000 / context 256 / d_model 512 / num_layers 4 / num_heads 16 / d_ff 1344 / RoPE theta 10000,约 17M 参数。
- 本地(WSL,RTX 4060 8GB):跑冒烟训练与单测;真实训练上云 GPU,配置见 `scripts/`。
- 启动 / 测试命令:
  - 测试:`uv run pytest tests/ -v`
  - 预处理:`uv run python scripts/prepare_data.py`
  - 本地冒烟训练:`uv run python zerollm/run_training.py --experiment-name smoke ...`(小步数)
  - 生成:`uv run python scripts/generate.py --prompt ...`
- 输出位置:`data/`(语料与 token id,gitignore)、`experiments/<name>/`(metrics.csv、config.json、checkpoint)、`figures/`。

## 边界

- 本仓库独立于课程仓库 `cs336/assignment1-basics`;只复制其中 BPE 成果,不回写课程仓库,不在课程仓库里补作业(课程 AGENTS.md 的助教规则继续适用于课程仓库)。
- 语料与数据文件(大文件)不入库;`.gitignore` 覆盖 `data/`、`*.npy`、`checkpoint*.pt`。
- 只使用占位 token;真实密钥、cookie、账号不写入仓库。

## 本项目使用的技能

- `verification`:每个模块完成后的自验清单(roundtrip、断点续跑、loss 下降)。
- `code-reviewer`:提交前对模块做 review。

## 交接

- 当前进度与下一步见 `notes/handoff.md`
