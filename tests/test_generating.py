"""zerolm.generating 测试:top-p 采样输出形状、prompt 保留、批量、温度校验。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from zerolm.generating import generate, generate_ids
from zerolm.model import TransformerLM
from zerolm.tokenizer import Tokenizer
from zerolm.train_bpe import train_bpe


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("gen")
    corpus = d / "c.txt"
    corpus.write_text(("从前有一只小猫。它喜欢喝牛奶！" * 20 + "<|endoftext|>\n") * 5, encoding="utf-8")
    vocab, merges = train_bpe(str(corpus), 300, ["<|endoftext|>"])
    tok = Tokenizer(vocab, merges, ["<|endoftext|>"])
    torch.manual_seed(0)
    model = TransformerLM(
        vocab_size=len(vocab), context_length=32, d_model=32, num_layers=1, num_heads=2, d_ff=64, rope_theta=10000.0
    )
    return tok, model


class TestGenerate:
    def test_single_prompt(self, tiny):
        tok, model = tiny
        out = generate("从前", model, tok, max_new_tokens=10)
        assert isinstance(out, list) and len(out) == 1
        assert out[0].startswith("从前")
        assert len(out[0]) >= 2

    def test_batch_prompts(self, tiny):
        tok, model = tiny
        outs = generate(["从前", "小猫"], model, tok, max_new_tokens=5)
        assert len(outs) == 2
        assert outs[0].startswith("从前") and outs[1].startswith("小猫")

    def test_deterministic_with_seed(self, tiny):
        tok, model = tiny
        torch.manual_seed(123)
        a = generate("从前", model, tok, max_new_tokens=8)
        torch.manual_seed(123)
        b = generate("从前", model, tok, max_new_tokens=8)
        assert a == b

    def test_temperature_zero_raises(self, tiny):
        tok, model = tiny
        with pytest.raises(AssertionError):
            generate("从前", model, tok, temperature=0.0)

    def test_greedy_like_low_top_p(self, tiny):
        """top_p 很小 → 每步退化到几乎确定,同 seed 下两次结果一致。"""
        tok, model = tiny
        torch.manual_seed(7)
        a = generate("从前", model, tok, max_new_tokens=12, top_p=0.05, temperature=0.5)
        torch.manual_seed(7)
        b = generate("从前", model, tok, max_new_tokens=12, top_p=0.05, temperature=0.5)
        assert a == b

    def test_context_truncation(self, tiny):
        """prompt 比 context_size 长时不应崩溃。"""
        tok, model = tiny
        long_prompt = "从前有一只小猫。" * 20
        out = generate(long_prompt, model, tok, context_size=16, max_new_tokens=5)
        assert out[0].startswith(long_prompt)


class TestTopK:
    """top-k 硬截断:候选数量必须精确等于 k,且与分布形状无关。"""

    def test_top_k_one_is_greedy(self, tiny):
        """top_k=1 时只剩一个候选,采样退化成贪心 → 换种子结果不变。"""
        tok, model = tiny
        torch.manual_seed(1)
        a = generate("从前", model, tok, max_new_tokens=10, top_k=1, top_p=1.0)
        torch.manual_seed(999)
        b = generate("从前", model, tok, max_new_tokens=10, top_k=1, top_p=1.0)
        assert a == b

    def test_candidate_count_equals_k(self, tiny):
        """平均保留候选数 == k,标准差 == 0(top-k 与分布无关)。"""
        tok, model = tiny
        for k in (1, 3, 7):
            _, _, stats = generate_ids(
                "从前", model, tok, max_new_tokens=8, top_k=k, top_p=1.0
            )
            assert stats["avg_candidates"] == k
            assert stats["std_candidates"] == 0.0, f"top_k={k} 时 std 应为 0"

    def test_top_p_candidates_vary(self, tiny):
        """对照:top-p 的候选数随步变化 → std 应大于 0。"""
        tok, model = tiny
        _, _, stats = generate_ids(
            "从前有一只小猫", model, tok, max_new_tokens=12, top_p=0.9
        )
        assert stats["std_candidates"] > 0.0

    def test_top_k_and_top_p_stack(self, tiny):
        """先 top-k 再 top-p:最终候选数不超过 k。"""
        tok, model = tiny
        _, _, stats = generate_ids(
            "从前", model, tok, max_new_tokens=8, top_k=2, top_p=0.999
        )
        assert stats["avg_candidates"] <= 2

    def test_invalid_top_k_raises(self, tiny):
        tok, model = tiny
        with pytest.raises(AssertionError):
            generate("从前", model, tok, top_k=0)
        with pytest.raises(AssertionError):
            generate("从前", model, tok, top_p=0.0)


class TestGenerateIds:
    def test_returns_prompt_length(self, tiny):
        """返回的 prompt 长度必须等于 tokenizer.encode 的长度,便于切出纯续写。"""
        tok, model = tiny
        prompts = ["从前", "小猫"]
        ids, lens, _ = generate_ids(prompts, model, tok, max_new_tokens=5)
        assert lens == [len(tok.encode(p)) for p in prompts]
        for seq, n, prompt in zip(ids, lens, prompts, strict=True):
            assert len(seq) >= n
            assert seq[:n] == tok.encode(prompt)

    def test_batch_shapes_consistent(self, tiny):
        tok, model = tiny
        ids, lens, stats = generate_ids(["从前", "小猫喝水"], model, tok, max_new_tokens=4)
        assert len(ids) == len(lens) == 2
        assert set(stats) == {"avg_candidates", "std_candidates"}


class TestKVCache:
    """KV cache 的正确性:带 cache 与不带 cache 必须逐 token 完全相同。"""

    def test_prefill_matches_full_forward(self, tiny):
        """prefill 的 logits 要和整段前向 allclose(位置编码没写错的必要条件)。"""
        _, model = tiny
        x = torch.randint(0, 50, (2, 8))
        full = model(x)
        logits, kv = model(x, use_cache=True)
        assert torch.allclose(full, logits, atol=1e-5)
        assert len(kv) == len(model.layers)
        assert kv[0][0].shape[-2] == 8

    def test_incremental_decode_matches_full(self, tiny):
        """decode 一个 token 的 logits 要和"整段重算的最后一个位置"一致。"""
        _, model = tiny
        x = torch.randint(0, 50, (1, 6))
        nxt = torch.randint(0, 50, (1, 1))
        _, kv = model(x, use_cache=True)
        one, kv2 = model(nxt, token_positions=torch.tensor([6]), past_kv=kv, use_cache=True)
        ref = model(torch.cat([x, nxt], dim=1))[:, -1:, :]
        assert torch.allclose(one, ref, atol=1e-5)
        assert kv2[0][0].shape[-2] == 7

    def test_greedy_identical(self, tiny):
        """top_k=1 是确定性的:两条路径的 id 序列必须一模一样。"""
        tok, model = tiny
        a = generate_ids("从前", model, tok, max_new_tokens=10, top_k=1, use_cache=True)[0]
        b = generate_ids("从前", model, tok, max_new_tokens=10, top_k=1, use_cache=False)[0]
        assert a == b

    def test_sampling_identical_with_same_seed(self, tiny):
        """采样:同 seed 下两条路径消耗相同的随机数,id 序列应相同。"""
        tok, model = tiny
        cases = (("top_p=0.9", 0.9, 1.0), ("top_p=0.95", 0.95, 1.0), ("t=0.8", 0.95, 0.8))
        for name, top_p, temperature in cases:
            torch.manual_seed(11)
            a = generate_ids(
                "从前", model, tok, max_new_tokens=10, use_cache=True,
                top_p=top_p, temperature=temperature,
            )[0]
            torch.manual_seed(11)
            b = generate_ids(
                "从前", model, tok, max_new_tokens=10, use_cache=False,
                top_p=top_p, temperature=temperature,
            )[0]
            assert a == b, f"{name} 下两条路径不一致"

    def test_unequal_length_batch_disables_cache(self, tiny):
        """不等长 batch 有右填充 pad,cache 应自动关闭,且结果与不开 cache 一致。"""
        tok, model = tiny
        prompts = ["从前", "从前有一只小猫"]
        torch.manual_seed(5)
        a = generate_ids(prompts, model, tok, max_new_tokens=6, top_k=1, use_cache=True)[0]
        torch.manual_seed(5)
        b = generate_ids(prompts, model, tok, max_new_tokens=6, top_k=1, use_cache=False)[0]
        assert a == b

    def test_falls_back_past_context_size(self, tiny):
        """长度超过 context_size 后回退到整窗重算,结果仍须一致(RoPE 表只有那么长)。"""
        tok, model = tiny
        torch.manual_seed(7)
        a = generate_ids("从前", model, tok, max_new_tokens=14, top_k=1,
                         context_size=8, use_cache=True)[0]
        torch.manual_seed(7)
        b = generate_ids("从前", model, tok, max_new_tokens=14, top_k=1,
                         context_size=8, use_cache=False)[0]
        assert a == b
