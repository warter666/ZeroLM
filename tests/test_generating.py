"""zerollm.generating 测试:top-p 采样输出形状、prompt 保留、批量、温度校验。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from zerollm.generating import generate
from zerollm.model import TransformerLM
from zerollm.tokenizer import Tokenizer
from zerollm.train_bpe import train_bpe


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