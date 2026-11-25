"""训练组件:随机批采样(get_batch)与 checkpoint 存取。"""

from __future__ import annotations

import os
import typing

import numpy as np
import torch


def get_batch(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """从一维 token id 数组随机采样 batch。

    返回 (inputs, targets),均为 [batch_size, context_length] 的 int64;
    targets 是 inputs 在序列上右移一位的结果。
    起点范围 [0, len(data) - context_length - 1],保证 targets 不越界。
    """
    n = len(data)
    if n < context_length + 1:
        raise ValueError(f"数据长度 {n} 不足以切出 context_length={context_length} 的样本")
    starts = np.random.randint(0, n - context_length, size=batch_size)
    offsets = np.arange(context_length)
    idx = starts[:, None] + offsets[None, :]
    inputs = torch.from_numpy(data[idx].astype(np.int64)).to(device)
    targets = torch.from_numpy(data[idx + 1].astype(np.int64)).to(device)
    return inputs, targets


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    """恢复 model(和 optimizer 若提供),返回训练步数。"""
    saved = torch.load(src, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"])
    if optimizer is not None and "optimizer" in saved:
        optimizer.load_state_dict(saved["optimizer"])
    return int(saved.get("iteration", saved.get("step", 0)))
