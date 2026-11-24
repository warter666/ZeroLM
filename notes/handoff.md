# 交接日志

> 最新在最上面。每条写清「做了什么 / 当前状态 / 下一步 / 坑」。

## 2026-11-24 · 核心模块完成,冒烟验证中

**做了什么**
- 全部 7 个核心模块写完并各自带测试,**59 个测试全绿**:
  - `train_bpe.py`(课程实现改编 + 流式读取)、`tokenizer.py`(encode/流式/decode/save/load)、
    `model.py`(pre-norm + RoPE decoder-only)、`loss_optimizer.py`(CE/AdamW/cosine LR/裁剪)、
    `training.py`(get_batch/checkpoint)、`run_training.py`(CLI/断点续训/csv)、`generating.py`(top-p)。
- 数据脚本:`download_data.py`(HF 下载 + train/valid 切分)、`train_tokenizer.py`(采样训 BPE)、
  `prepare_data.py`(多进程编码 + 断点续跑,输出 .npy)。
- `scripts/smoke_test.sh` 端到端冒烟、`scripts/train_cloud.sh` 云端三阶段(冒烟/预处理/训练)。
- 两次提交:立项骨架、核心模块。

**当前状态**
- 冒烟测试在后台跑(2k 篇 → 2k 词表 → 500 篇编码 → 30 步 CPU 训练 → 生成)。

**下一步**
1. 确认冒烟通过(修暴露出的问题)。
2. 用户开云机器后:`bash scripts/train_cloud.sh`(--stage smoke → prep → train)。
3. 训完:`scripts/generate.py` 看生成质量;`figures/` 画 loss 曲线(可选)。

**坑**
- uv 默认直连 PyPI,torch 900MB+ 下载极慢/429。统一用镜像:
  `export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple`(云在国外机器则不用)。
- `tokenizer.py` 的 `_safe_cut`:special token 恰好结束在缓冲区末尾时必须整段编出,
  否则会被正则拆碎(测试 test_special_across_chunk_boundary 覆盖)。
- 中文一个汉字 3 字节,10k 词表下"整字 token"是压缩主力;别用英文语料的直觉预估 token 数。
- AdamW 更新顺序:先 moment,再 decoupled decay,最后参数更新(与参考式对齐的顺序,别乱调)。


