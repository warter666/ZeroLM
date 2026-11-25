# ZeroLM

> 展示名 **ZeroLM**(与 GitHub 仓库/Badge 一致);本地目录仍为 `~/projects/ZeroLLM`,Python 包名 `zerollm`。

## 目标与验收

- 目标：作为 CS336 assignment1 的课程实践,从零训练一个中文小语言模型。语料为中文 TinyStories(~200 万条简短故事),BPE 分词沿用 `cs336/assignment1-basics` 已完成的 `train_bpe.py`,其余模块(模型/优化/训练/采样)全部手写实现,最终在云 GPU 上完成一次完整训练,能采样生成连贯中文小故事。
- 验收:
  1. BPE 训练在中文语料上产出 `vocab.json` + `merges.txt`,roundtrip(编码→解码)无损;
  2. 全量数据预处理(语料→token id 的 `.npy`)可断点续跑;
  3. 训练脚本支持断点续训、eval 记录(csv)、checkpoint;
  4. 云 GPU 训练 ~10k 步后 val loss 收敛(参考仓库英文版 ~1.36,中文版量级相当即可);
  5. `generate.py` 用 prompt 采样出可读的中文故事。

## 技术与运行

- 技术栈:Python 3.12 + uv、PyTorch(CUDA)、numpy、regex、einops、jaxtyping、tqdm、matplotlib;云侧另加 modal(可选)。
- 模型规模(默认):vocab 10000 / context 256 / d_model 512 / num_layers 4 / num_heads 16 / d_ff 1344 / RoPE theta 10000,**22.70M 参数**(embedding 5.12M + lm_head 5.12M + 4 层 12.46M);加 `--tie-embeddings` 为 17.58M。早先写的“17M”是权重绑定下的数字。
- 本地(WSL,RTX 4060 8GB):跑冒烟训练与单测;真实训练上云 GPU,配置见 `scripts/`。
- GPU 上的加速开关:`--sdpa`(换 flash 内核,1.9x)、`--compile`(再 1.3x)、`--tie-embeddings`(参数量 22.70M→17.58M)。老 run 续训必须保持这些开关与 `--num-iters` 一致。
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
