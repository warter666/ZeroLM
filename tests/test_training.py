"""zerollm.training 测试:批采样与 checkpoint 存取。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from zerollm.loss_optimizer import AdamW
from zerollm.model import TransformerLM
from zerollm.training import get_batch, load_checkpoint, save_checkpoint


class TestGetBatch:
    def test_shapes_and_shift(self):
        data = np.arange(1000, dtype=np.uint16)
        x, y = get_batch(data, 4, 16)
        assert x.shape == (4, 16) and y.shape == (4, 16)
        assert x.dtype == torch.int64 and y.dtype == torch.int64
        # y 是 x 右移一位
        assert torch.equal(y[:, :-1], x[:, 1:])

    def test_no_out_of_range(self):
        data = np.arange(64, dtype=np.uint16)
        for _ in range(50):
            x, y = get_batch(data, 8, 16)
            assert x.min() >= 0 and y.max() < 64

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError):
            get_batch(np.arange(10), 2, 16)

    def test_memmap_input(self, tmp_path):
        p = tmp_path / "t.npy"
        np.save(p, np.arange(500, dtype=np.uint16))
        data = np.load(p, mmap_mode="r")
        x, _ = get_batch(data, 2, 8)
        assert x.shape == (2, 8)


class TestCheckpoint:
    def _make(self):
        torch.manual_seed(0)
        model = TransformerLM(32, 8, 16, 1, 2, 32, 10000.0)
        opt = AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, 32, (2, 8))
        model(x).sum().backward()
        opt.step()
        return model, opt

    def test_roundtrip(self, tmp_path):
        model, opt = self._make()
        path = tmp_path / "ck.pt"
        save_checkpoint(model, opt, 42, path)
        assert path.exists()

        torch.manual_seed(1)
        model2 = TransformerLM(32, 8, 16, 1, 2, 32, 10000.0)
        opt2 = AdamW(model2.parameters(), lr=1e-3)
        step = load_checkpoint(path, model2, opt2)
        assert step == 42
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert n1 == n2
            assert torch.equal(p1, p2)
        # optimizer 状态也要恢复
        s1 = opt.state_dict()
        s2 = opt2.state_dict()
        assert len(s1["state"]) == len(s2["state"]) == len(list(model.parameters()))

    def test_load_without_optimizer(self, tmp_path):
        model, opt = self._make()
        path = tmp_path / "ck.pt"
        save_checkpoint(model, opt, 7, path)
        model2 = TransformerLM(32, 8, 16, 1, 2, 32, 10000.0)
        assert load_checkpoint(path, model2) == 7