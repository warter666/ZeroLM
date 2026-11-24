"""自回归生成:temperature + top-p(nucleus)采样。"""

from __future__ import annotations

import torch

from zerollm.tokenizer import Tokenizer


@torch.no_grad()
def generate(
    prompts: str | list[str],
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    device: str | torch.device = "cpu",
    context_size: int = 256,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 0.95,
    eos_id: int | None = None,
) -> list[str]:
    """给一个或多个 prompt 续写;返回解码后的完整文本(含 prompt)。"""
    assert temperature > 0, "temperature 必须 > 0"
    single = isinstance(prompts, str)
    prompt_list = [prompts] if single else list(prompts)

    # 右填充到同一长度,pad id 用 0(不参与生成)
    encoded = [tokenizer.encode(p) for p in prompt_list]
    max_len = max(len(e) for e in encoded)
    pad = 0
    ids = torch.full((len(encoded), max_len), pad, dtype=torch.long, device=device)
    for i, e in enumerate(encoded):
        ids[i, : len(e)] = torch.tensor(e, dtype=torch.long, device=device)
    lengths = [len(e) for e in encoded]

    model.eval()
    for _ in range(max_new_tokens):
        ctx = ids[:, -context_size:]
        logits = model(ctx)[:, -1, :]  # [B, V]
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)

        # top-p 过滤:保留累计概率达到 p 的最小集合(含跨过阈值的那一个)
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cum - sorted_probs  # 该 token 之前的累计概率
        keep = cutoff < top_p
        sorted_probs = sorted_probs * keep
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        next_sorted = torch.multinomial(sorted_probs, num_samples=1)  # [B,1]
        next_ids = sorted_idx.gather(1, next_sorted)  # [B,1]
        ids = torch.cat([ids, next_ids], dim=1)

        if eos_id is not None and bool((next_ids == eos_id).all()):
            break

    out = []
    for i in range(len(prompt_list)):
        seq = ids[i, : lengths[i] + max_new_tokens].tolist()
        out.append(tokenizer.decode(seq))
    return out