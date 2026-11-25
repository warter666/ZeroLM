"""zerolm.model 测试:形状、因果性、RoPE 性质、初始化、梯度。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from zerolm.model import (
    Embedding,
    Linear,
    MultiHeadSelfAttention,
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLU,
    TransformerLM,
    scaled_dot_product_attention,
    softmax,
    swiglu_d_ff,
)


class TestLinear:
    def test_shape_and_no_bias(self):
        lin = Linear(16, 32)
        assert lin.weight.shape == (32, 16)
        assert not hasattr(lin, "bias")
        assert lin(torch.randn(4, 16)).shape == (4, 32)

    def test_init_std(self):
        torch.manual_seed(0)
        lin = Linear(512, 512)
        expected = (2 / (512 + 512)) ** 0.5
        assert abs(lin.weight.std().item() - expected) < 0.005


class TestEmbedding:
    def test_lookup(self):
        emb = Embedding(10, 8)
        x = torch.tensor([[0, 3, 9]])
        out = emb(x)
        assert out.shape == (1, 3, 8)
        assert torch.equal(out[0, 1], emb.weight[3])


class TestRMSNorm:
    def test_rms_is_one(self):
        norm = RMSNorm(16)
        x = torch.randn(2, 5, 16) * 7
        out = norm(x)
        rms = out.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)

    def test_weight_scaling(self):
        norm = RMSNorm(16)
        with torch.no_grad():
            norm.weight.fill_(2.0)
        x = torch.randn(1, 4, 16)
        assert torch.allclose(norm(x), 2 * norm.weight.new_ones(1) * (x / x.pow(2).mean(-1, keepdim=True).add(1e-5).sqrt()) * 1, atol=1e-3)


class TestSwiGLU:
    def test_shape(self):
        f = SwiGLU(16, 32)
        assert f(torch.randn(2, 3, 16)).shape == (2, 3, 16)

    def test_default_d_ff(self):
        assert swiglu_d_ff(512) % 64 == 0
        assert swiglu_d_ff(512) >= 8 * 512 / 3

    def test_has_three_projections(self):
        f = SwiGLU(16, 32)
        assert hasattr(f, "w1") and hasattr(f, "w2") and hasattr(f, "w3")


class TestRoPE:
    def test_norm_preserved(self):
        rope = RotaryPositionalEmbedding(10000.0, 8, 16)
        x = torch.randn(2, 5, 8)
        pos = torch.arange(5)
        out = rope(x, pos)
        assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5)

    def test_position_zero_identity(self):
        rope = RotaryPositionalEmbedding(10000.0, 8, 16)
        x = torch.randn(1, 4, 8)
        out = rope(x, torch.zeros(4, dtype=torch.long))
        assert torch.allclose(out, x, atol=1e-6)

    def test_relative_shift(self):
        """同样内容、不同起点位置,RoPE 输出的相对关系应当只依赖差值。

        比较 0..3 与 5..8 两个位置序列拼上同样向量后的内积;不要求相等,
        但要求 "同偏移对" 的内积一致(这是 RoPE 相对性的定义)。
        """
        rope = RotaryPositionalEmbedding(10000.0, 8, 32)
        torch.manual_seed(0)
        q = torch.randn(1, 2, 8)
        k = torch.randn(1, 2, 8)

        def dot(pos):
            qp = rope(q, torch.tensor([pos, pos + 1]))
            kp = rope(k, torch.tensor([pos, pos + 1]))
            return (qp * kp).sum(-1)

        a = dot(0)
        b = dot(7)
        assert torch.allclose(a, b, atol=1e-4)


class TestSoftmax:
    def test_sums_to_one(self):
        x = torch.randn(3, 5)
        assert torch.allclose(softmax(x, -1).sum(-1), torch.ones(3), atol=1e-6)

    def test_stable_with_large_values(self):
        x = torch.tensor([[1e4, 0.0, -1e4]])
        out = softmax(x, -1)
        assert torch.isfinite(out).all()
        assert torch.allclose(out.sum(-1), torch.ones(1), atol=1e-6)


class TestAttention:
    def test_causal_prefix_invariance(self):
        """改后面的 token,前面的输出不应变化。"""
        torch.manual_seed(0)
        attn = MultiHeadSelfAttention(16, 4, max_seq_length=8, theta=10000.0)
        x = torch.randn(1, 6, 16)
        y1 = attn(x)
        x2 = x.clone()
        x2[:, 3:] = torch.randn(1, 3, 16)  # 只改 t>=3
        y2 = attn(x2)
        assert torch.allclose(y1[:, :3], y2[:, :3], atol=1e-5)

    def test_sdpa_mask_semantics(self):
        q = torch.eye(3).unsqueeze(0).unsqueeze(0)  # [1,1,3,3]
        k = torch.eye(3).unsqueeze(0).unsqueeze(0)
        v = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]]])
        mask = torch.tril(torch.ones(3, 3, dtype=torch.bool))
        out = scaled_dot_product_attention(q, k, v, mask)
        # 第 0 行只能看第 0 个 v
        assert torch.allclose(out[0, 0, 0], v[0, 0, 0], atol=1e-5)


class TestTransformerLM:
    @pytest.fixture
    def model(self):
        torch.manual_seed(0)
        return TransformerLM(
            vocab_size=64, context_length=16, d_model=32, num_layers=2, num_heads=4, d_ff=64, rope_theta=10000.0
        )

    def test_output_shape(self, model):
        x = torch.randint(0, 64, (3, 16))
        assert model(x).shape == (3, 16, 64)

    def test_causal(self, model):
        x = torch.randint(0, 64, (1, 8))
        y1 = model(x)
        x2 = x.clone()
        x2[0, 4:] = torch.randint(0, 64, (4,))
        y2 = model(x2)
        assert torch.allclose(y1[:, :4], y2[:, :4], atol=1e-5)

    def test_grad_flows(self, model):
        x = torch.randint(0, 64, (2, 16))
        model(x).sum().backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"{name} 没有梯度"

    def test_param_count(self, model):
        n = model.num_parameters()
        assert n == sum(p.numel() for p in model.parameters())
        assert model.num_parameters(exclude_embeddings=True) < n
