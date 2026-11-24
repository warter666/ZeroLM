"""训练用到的数值组件:交叉熵、AdamW、学习率调度、梯度裁剪。

对齐 CS336 assignment1 的接口约定,AdamW 的更新式与作业测试的 numpy 参考实现一致。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """数值稳定的交叉熵。

    inputs: [..., vocab_size] logits;targets: [...] int64。返回标量均值。
    """
    vocab = inputs.shape[-1]
    logits = inputs.reshape(-1, vocab)
    tgt = targets.reshape(-1)
    m = logits.max(dim=-1, keepdim=True).values
    shifted = logits - m
    logsumexp = torch.log(torch.exp(shifted).sum(dim=-1))
    tgt_logit = shifted.gather(1, tgt.unsqueeze(1)).squeeze(1)
    return (logsumexp - tgt_logit).mean()


class AdamW(torch.optim.Optimizer):
    """AdamW(decoupled weight decay)。

    超参放在 defaults 里,允许 param group 覆盖(不同层用不同 lr / wd)。
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"lr 必须 >= 0,得到 {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        # torch 各版本 Optimizer.step 的返回类型标注不一致(float / None),
        # 这里返回闭包结果(Optional[float]),调用方都不看返回值。
        loss: float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    # 必须 zeros 而不是 empty:empty 里是未定义值,会把矩估计污染
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                t = state["t"] + 1
                state["t"] = t
                m, v = state["m"], state["v"]

                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                # decoupled weight decay:先按 lr 收缩参数
                p.mul_(1 - lr * wd)

                bias1 = 1 - beta1**t
                bias2 = 1 - beta2**t
                step_size = lr * math.sqrt(bias2) / bias1
                p.addcdiv_(m, v.sqrt().add_(eps), value=-step_size)

        return loss


def cosine_lr(
    t: int,
    lr_max: float,
    lr_min: float,
    warmup_steps: int,
    cosine_steps: int,
) -> float:
    """warmup(线性 0 -> lr_max)+ cosine(lr_max -> lr_min,共 cosine_steps 步)。"""
    if t < warmup_steps:
        return lr_max * t / warmup_steps
    if t >= warmup_steps + cosine_steps:
        return lr_min
    progress = (t - warmup_steps) / cosine_steps
    return lr_min + 0.5 * (1 + math.cos(math.pi * progress)) * (lr_max - lr_min)


def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_norm: float) -> torch.Tensor:
    """按全局 L2 范数裁剪(原地),返回裁剪前的总范数。"""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    total = torch.sqrt(torch.stack([(g.detach() ** 2).sum() for g in grads]).sum())
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for g in grads:
            g.mul_(scale)
    return total.detach()