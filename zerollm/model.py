"""Transformer 语言模型(decoder-only,pre-norm + RoPE),对齐 CS336 assignment1 的组件划分。

组件:Linear / Embedding / RMSNorm / SwiGLU / RotaryPositionalEmbedding /
      scaled_dot_product_attention / MultiHeadSelfAttention / TransformerBlock / TransformerLM
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class Linear(nn.Module):
    """无 bias 线性层,截断正态初始化,std = sqrt(2 / (in + out))。"""

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        std = math.sqrt(2.0 / (in_features + out_features))
        w = torch.empty(out_features, in_features, device=device, dtype=dtype)
        self.weight = nn.Parameter(nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3 * std, b=3 * std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    """查表嵌入,N(0,1) 截断到 [-3, 3](权重绑定时改用小方差,见 init_std)。"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device=None,
        dtype=None,
        init_std: float = 1.0,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        w = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        bound = 3 * init_std
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(w, mean=0.0, std=init_std, a=-bound, b=bound)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]


class RMSNorm(nn.Module):
    """RMSNorm:在 float32 下计算再回到输入 dtype(避免 fp16/bf16 溢出)。"""

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms * self.weight).to(in_dtype)


def swiglu_d_ff(d_model: int, multiple_of: int = 64) -> int:
    """SwiGLU 隐藏维度默认取 8/3 * d_model,向上取到 multiple_of 的整数倍。"""
    d_ff = int(8 * d_model / 3)
    if d_ff % multiple_of != 0:
        d_ff = multiple_of * ((d_ff + multiple_of - 1) // multiple_of)
    return d_ff


class SwiGLU(nn.Module):
    """FFN(x) = W2( SiLU(W1 x) * W3 x )。"""

    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        self.d_ff = d_ff if d_ff is not None else swiglu_d_ff(d_model)
        self.w1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)  # gate
        self.w3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)  # up
        self.w2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)  # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    """RoPE:cos/sin 查表,支持任意 token_positions(推理时可用于 KV cache)。"""

    def __init__(self, theta: float, d_k: int, max_seq_length: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "RoPE 需要偶数 head_dim"
        self.d_k = d_k
        seq = torch.arange(max_seq_length, device=device, dtype=torch.float32)[:, None]  # [S,1]
        idx = torch.arange(d_k // 2, device=device, dtype=torch.float32)[None, :]  # [1, d/2]
        freq = 1.0 / (theta ** (2.0 * idx / d_k))
        angle = seq * freq  # [S, d/2]
        self.cos_table: torch.Tensor
        self.sin_table: torch.Tensor
        self.register_buffer("cos_table", torch.cos(angle), persistent=False)
        self.register_buffer("sin_table", torch.sin(angle), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_table[token_positions].to(x.dtype)  # [..., S, d/2]
        sin = self.sin_table[token_positions].to(x.dtype)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = cos * x_even - sin * x_odd
        out[..., 1::2] = sin * x_even + cos * x_odd
        return out


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x - x.max(dim, keepdim=True).values
    e = x.exp()
    return e / e.sum(dim, keepdim=True)


def scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """mask 语义:True 保留,False 屏蔽(内部转成 -inf 后 softmax)。"""
    d_k = keys.shape[-1]
    scores = queries @ keys.transpose(-1, -2) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return softmax(scores, -1) @ values


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_length: int | None = None,
        theta: float | None = None,
        device=None,
        dtype=None,
        use_sdpa: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.theta = theta
        # sm80+ 走 flash / mem-efficient 内核,不用物化 [B,H,S,S] 分数矩阵
        self.use_sdpa = use_sdpa
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = (
            RotaryPositionalEmbedding(theta, self.head_dim, max_seq_length)
            if theta and max_seq_length is not None
            else None
        )
        self.causal_mask: torch.Tensor
        if max_seq_length is not None:
            # True = 保留(下三角含对角);使用时按序列长度切片
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(max_seq_length, max_seq_length, dtype=torch.bool)),
                persistent=False,
            )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        *batch, seq, _ = x.shape
        q = self.q_proj(x).view(*batch, seq, self.num_heads, self.head_dim).transpose(-2, -3)
        k = self.k_proj(x).view(*batch, seq, self.num_heads, self.head_dim).transpose(-2, -3)
        v = self.v_proj(x).view(*batch, seq, self.num_heads, self.head_dim).transpose(-2, -3)
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        if self.use_sdpa:
            ctx = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            mask = self.causal_mask[:seq, :seq]
            ctx = scaled_dot_product_attention(q, k, v, mask)
        ctx = ctx.transpose(-2, -3).contiguous().view(*batch, seq, self.d_model)
        return self.output_proj(ctx)


class TransformerBlock(nn.Module):
    """pre-norm: x = x + Attn(LN(x)); x = x + FFN(LN(x))。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None,
        max_seq_length: int,
        theta: float,
        use_sdpa: bool = False,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, max_seq_length, theta, use_sdpa=use_sdpa
        )
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int | None,
        rope_theta: float,
        device=None,
        dtype=None,
        tie_embeddings: bool = False,
        use_sdpa: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        # 权重绑定时 lm_head 复用这张矩阵,必须用小方差初始化:否则 logits 尺度 = sqrt(d_model)*std,
        # 会直接把初始 loss 推到 40+ (实测 44.7 vs 正常 9.21),训不起来。
        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype,
            init_std=0.02 if tie_embeddings else 1.0,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, num_heads, d_ff, context_length, rope_theta, use_sdpa=use_sdpa
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        # 权重绑定(GPT-2 式):lm_head 直接复用 embedding 矩阵,省 vocab*d_model 个参数
        self.tie_embeddings = tie_embeddings
        self.lm_head = None if tie_embeddings else Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, S] 的 token id -> logits: [B, S, vocab_size]。"""
        if token_positions is None:
            token_positions = torch.arange(x.shape[-1], device=x.device)
        h = self.token_embeddings(x)
        for layer in self.layers:
            h = layer(h, token_positions)
        h = self.ln_final(h)
        if self.lm_head is None:
            return h @ self.token_embeddings.weight.T
        return self.lm_head(h)

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.token_embeddings.weight.numel()
            if self.lm_head is not None:
                n -= self.lm_head.weight.numel()
        return n