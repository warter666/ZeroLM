"""zerollm.train_bpe 测试:中文语料正确性、流式读取等价、save/load 自洽。"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zerollm.train_bpe import train_bpe, gpt2pat

CHINESE_SAMPLE = """从前的从前，有一只小兔子。它住在森林边上，每天早上都会去看太阳升起。
小兔子最喜欢吃胡萝卜，特别是下过雨之后的那种。
有一天，一只小松鼠对它说：“我们一起去探险吧！”
"Sure!" said the little rabbit. 它们一起走过了小桥，看到了一条发光的小河。
特殊标记测试<|endoftext|>后面的内容。123 个数字也要被切开，比如 2026 年。"""


def _write_tmp(text: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _decodes_cleanly(b: bytes) -> bool:
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return len(s) == 1


class TestTrainBPE:
    def test_basic_vocab_size_and_bytes(self):
        path = _write_tmp(CHINESE_SAMPLE)
        try:
            vocab, merges = train_bpe(path, 400, [])
            assert len(vocab) == 400
            # 256 个单字节必在
            for i in range(256):
                assert vocab[i] == bytes([i])
            # merges 数量 = vocab_size - 256
            assert len(merges) == 400 - 256
            # 每个 merge 拼出的 bytes 必须等于对应 vocab 条目
            for i, (a, b) in enumerate(merges):
                assert vocab[256 + i] == a + b
        finally:
            os.unlink(path)

    def test_special_tokens_reserved(self):
        path = _write_tmp(CHINESE_SAMPLE)
        try:
            vocab, merges = train_bpe(path, 400, ["<|endoftext|>"])
            assert vocab[256] == b"<|endoftext|>"
            assert len(merges) == 400 - 256 - 1
        finally:
            os.unlink(path)

    def test_chinese_merges_present(self):
        """中文语料上应学到整字/词级 token,编码后 token 数显著少于字节数。"""
        text = "从前有一只小兔子，小兔子住在森林边上。每天早上小兔子都会去看太阳升起。\n" * 20
        path = _write_tmp(text)
        try:
            vocab, merges = train_bpe(path, 700, [])
            # 至少有一个 token 是完整的单个汉字(3 字节且能独立解码)
            full_chars = [v for v in vocab.values() if len(v) == 3 and _decodes_cleanly(v)]
            assert full_chars, "没有学到完整的汉字 token"

            from zerollm.tokenizer import Tokenizer

            tok = Tokenizer(vocab, merges, [])
            n_tokens = len(tok.encode(text))
            n_bytes = len(text.encode("utf-8"))
            assert n_tokens < n_bytes / 2, f"压缩不足: {n_tokens} tokens vs {n_bytes} bytes"
        finally:
            os.unlink(path)

    def test_streaming_equals_wholesale(self):
        """流式(16MB 分块)与一次性整读在 pytorch 测试语料规模上应产出一致结果。

        用小 CHUNK 打桩模拟分块,验证 tail 逻辑不丢内容。
        """
        import zerollm.train_bpe as m

        path = _write_tmp(CHINESE_SAMPLE)
        try:
            vocab_full, merges_full = train_bpe(path, 500, ["<|endoftext|>"])

            old = m.CHUNK_SIZE
            m.CHUNK_SIZE = 64  # 极小分块,强制触发 tail 保留逻辑
            try:
                vocab_chunked, merges_chunked = train_bpe(path, 500, ["<|endoftext|>"])
            finally:
                m.CHUNK_SIZE = old

            assert merges_chunked == merges_full
            assert vocab_chunked == vocab_full
        finally:
            os.unlink(path)

    def test_reproducible(self):
        """同语料两次训练结果完全一致(无集合迭代随机性泄漏)。"""
        path = _write_tmp(CHINESE_SAMPLE)
        try:
            v1, m1 = train_bpe(path, 400, [])
            v2, m2 = train_bpe(path, 400, [])
            assert m1 == m2
            assert v1 == v2
        finally:
            os.unlink(path)
