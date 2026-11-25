"""zerolm.loss_optimizer 测试:交叉熵、AdamW 更新式、学习率调度、梯度裁剪。"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn.functional as F

from zerolm.loss_optimizer import AdamW, clip_gradients, cosine_lr, cross_entropy


class TestCrossEntropy:
    def test_matches_torch(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 7, 11)
        targets = torch.randint(0, 11, (4, 7))
        ours = cross_entropy(logits, targets)
        ref = F.cross_entropy(logits.reshape(-1, 11), targets.reshape(-1))
        assert torch.allclose(ours, ref, atol=1e-5)

    def test_numerically_stable(self):
        logits = torch.tensor([[1e4, 0.0, -1e4]])
        targets = torch.tensor([0])
        loss = cross_entropy(logits, targets)
        assert torch.isfinite(loss)
        assert loss.item() < 1e-3  # 正确类别概率≈1

    def test_gradient_finite(self):
        logits = torch.randn(2, 5, requires_grad=True)
        targets = torch.randint(0, 5, (2,))
        cross_entropy(logits, targets).backward()
        assert torch.isfinite(logits.grad).all()


class TestAdamW:
    def _reference_step(self, p, grad, m, v, t, lr, betas, eps, wd):
        """与 CS336 handout 一致的更新式参考实现(纯 python float)。

        顺序:先 moment 更新,再 decoupled weight decay,最后参数更新。
        """
        b1, b2 = betas
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad * grad
        p = p - lr * wd * p
        alpha_t = lr * math.sqrt(1 - b2 ** t) / (1 - b1 ** t)
        p = p - alpha_t * m / (math.sqrt(v) + eps)
        return p, m, v

    def test_single_step_matches_reference(self):
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.tensor([1.5, -0.5, 0.25], dtype=torch.float64))
        grad = torch.tensor([0.3, -0.7, 0.1], dtype=torch.float64)
        lr, betas, eps, wd = 1e-2, (0.9, 0.95), 1e-8, 0.1

        opt = AdamW([p], lr=lr, betas=betas, eps=eps, weight_decay=wd)
        p.grad = grad.clone()
        opt.step()

        ref_p, _, _ = self._reference_step(1.5, 0.3, 0.0, 0.0, 1, lr, betas, eps, wd)
        assert abs(p[0].item() - ref_p) < 1e-12

    def test_multi_step(self):
        torch.manual_seed(0)
        init = [1.0, 2.0]
        p = torch.nn.Parameter(torch.tensor(init, dtype=torch.float64))
        grads = [[0.5, -0.2], [0.1, 0.4], [-0.3, 0.05]]
        lr, betas, eps, wd = 1e-2, (0.9, 0.95), 1e-8, 0.01
        opt = AdamW([p], lr=lr, betas=betas, eps=eps, weight_decay=wd)

        ref = list(init)
        m = [0.0, 0.0]
        v = [0.0, 0.0]
        for t, g in enumerate(grads, start=1):
            p.grad = torch.tensor(g, dtype=torch.float64)
            opt.step()
            for j in range(2):
                ref[j], m[j], v[j] = self._reference_step(ref[j], g[j], m[j], v[j], t, lr, betas, eps, wd)

        assert torch.allclose(p.data, torch.tensor(ref, dtype=torch.float64), atol=1e-10)

    def test_zero_grad_skipped(self):
        p = torch.nn.Parameter(torch.tensor([1.0]))
        opt = AdamW([p], lr=1e-2)
        opt.step()  # grad 为 None,不应崩溃、不应改参数
        assert p.item() == 1.0

    def test_param_groups_override(self):
        p1 = torch.nn.Parameter(torch.tensor([1.0]))
        p2 = torch.nn.Parameter(torch.tensor([1.0]))
        opt = AdamW([{"params": [p1], "lr": 1e-2}, {"params": [p2], "lr": 0.0}], lr=1e-3)
        p1.grad = torch.tensor([1.0])
        p2.grad = torch.tensor([1.0])
        opt.step()
        assert p1.item() != 1.0
        assert p2.item() == 1.0


class TestCosineLR:
    def test_warmup_linear(self):
        assert cosine_lr(0, 1.0, 0.1, 10, 100) == 0.0
        assert abs(cosine_lr(5, 1.0, 0.1, 10, 100) - 0.5) < 1e-9

    def test_endpoints(self):
        assert abs(cosine_lr(10, 1.0, 0.1, 10, 100) - 1.0) < 1e-9
        assert abs(cosine_lr(110, 1.0, 0.1, 10, 100) - 0.1) < 1e-9

    def test_monotone_decay(self):
        vals = [cosine_lr(t, 1.0, 0.1, 10, 100) for t in range(10, 111, 5)]
        assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))


class TestClipGradients:
    def test_clips(self):
        p = torch.nn.Parameter(torch.zeros(4))
        p.grad = torch.tensor([3.0, 4.0, 0.0, 0.0])  # norm = 5
        norm = clip_gradients([p], max_norm=1.0)
        assert torch.isclose(norm, torch.tensor(5.0))
        assert torch.isclose(p.grad.norm(), torch.tensor(1.0), atol=1e-5)

    def test_noop_below_threshold(self):
        p = torch.nn.Parameter(torch.zeros(4))
        p.grad = torch.tensor([0.1, 0.1, 0.0, 0.0])
        before = p.grad.clone()
        clip_gradients([p], max_norm=10.0)
        assert torch.equal(p.grad, before)
