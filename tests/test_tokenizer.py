"""zerollm.tokenizer 测试:roundtrip、special token、流式与批量等价、save/load。"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from zerollm.tokenizer import Tokenizer
from zerollm.train_bpe import train_bpe

TEXT = """从前有一只小猫,它住在山谷里。每天早上,小猫都会去溪边喝水。
有一天,它遇到了一只小狗,小狗说:“你好!”小猫回答:“你好!”
两者成了好朋友。They played together all day.
<|endoftext|>
第二篇故事:12345 个星星落在了屋顶上。“真漂亮!”小猫说。"""

SPECIAL = ["<|endoftext|>", "<|bos|>"]


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    d = tmp_path_factory.mktemp("bpe")
    p = d / "corpus.txt"
    p.write_text(TEXT * 5, encoding="utf-8")
    vocab, merges = train_bpe(str(p), 400, SPECIAL)
    return Tokenizer(vocab, merges, SPECIAL)


class TestRoundtrip:
    def test_roundtrip_chinese(self, tok):
        assert tok.decode(tok.encode(TEXT)) == TEXT

    def test_roundtrip_empty(self, tok):
        assert tok.encode("") == []
        assert tok.decode([]) == ""

    def test_roundtrip_utf8_boundary(self, tok):
        s = "汉字和标点，还有emojis🙂不算太多。"
        assert tok.decode(tok.encode(s)) == s

    def test_special_token_single_id(self, tok):
        ids = tok.encode("<|endoftext|>")
        assert len(ids) == 1
        assert tok.vocab[ids[0]] == b"<|endoftext|>"

    def test_special_not_merged_into_text(self, tok):
        """special token 旁边的文本不应被跨 special 合并。"""
        a = tok.encode("你好<|endoftext|>世界")
        b = tok.encode("你好") + tok.encode("<|endoftext|>") + tok.encode("世界")
        assert a == b

    def test_unknown_bytes_fallback(self, tok):
        """词表外的单字节也要能编码(逐字节 id)。"""
        s = "\u0001\u0002"  # 控制字符,不会出现在训练语料里
        ids = tok.encode(s)
        assert tok.decode(ids) == s


class TestStreaming:
    def test_chunked_equals_whole(self, tok):
        whole = tok.encode(TEXT)
        for size in (1, 3, 7, 13, 100):
            chunked = list(tok.encode_iterable(TEXT[i : i + size] for i in range(0, len(TEXT), size)))
            assert chunked == whole, f"chunk size {size} 不一致"

    def test_special_across_chunk_boundary(self, tok):
        text = "开头<|endoftext|>结尾<|bos|>完"
        whole = tok.encode(text)
        for split in range(1, len(text)):
            got = list(tok.encode_iterable([text[:split], text[split:]]))
            assert got == whole, f"split at {split} 不一致"

    def test_iterable_file_like(self, tok):
        lines = TEXT.splitlines(keepends=True)
        assert list(tok.encode_iterable(iter(lines))) == tok.encode(TEXT)


class TestPersistence:
    def test_save_load_roundtrip(self, tok, tmp_path):
        vp = tmp_path / "vocab.json"
        mp = tmp_path / "merges.txt"
        tok.save(str(vp), str(mp))
        tok2 = Tokenizer.from_files(str(vp), str(mp), SPECIAL)
        assert tok2.encode(TEXT) == tok.encode(TEXT)
        assert tok2.vocab == tok.vocab
        assert tok2.merges == tok.merges

    def test_merge_rank_order(self, tok):
        """编码时按 merge 创建顺序合并,不做全局重新排序。"""
        assert len(tok.merge_ranks) == len(tok.merges)
        for pair, r in tok.merge_ranks.items():
            assert tok.merges[r] == pair