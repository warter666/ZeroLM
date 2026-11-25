"""自回归生成:temperature + top-k + top-p(nucleus)采样。

三个旋钮的作用(实验结论见 notes/sampling.md):
- temperature 改分布**形状**:logits/temperature,<1 变尖(保守),>1 变平(发散);
- top-k 只保留概率最高的 k 个候选,**候选数量固定**,不随分布形状变化;
- top-p 保留累计概率达到 p 的最小集合,**候选数量自适应**:分布尖时自动少,平时自动多。

实现顺序:先 top-k 截断,再在剩下的集合里做 top-p(与 HuggingFace 的 top_k/top_p 叠加语义一致)。
"""

from __future__ import annotations

import torch

from zerollm.tokenizer import Tokenizer


@torch.no_grad()
def generate_ids(
    prompts: str | list[str],
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    device: str | torch.device = "cpu",
    context_size: int = 256,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int | None = None,
    eos_id: int | None = None,
    use_cache: bool = True,
) -> tuple[list[list[int]], list[int], dict[str, float]]:
    """采样续写,返回 (每个序列的完整 id 列表, 各自 prompt 的 token 长度, 统计)。

    拿到 prompt 长度就能切出"纯续写"部分做统计,不必反解字符串。
    stats 里 `avg_candidates` / `std_candidates` = 每步保留的候选 token 数的均值与标准差,
    用来实证 "top-k 固定数量 / top-p 自适应数量":top-k 的 std 恒为 0,top-p 显著大于 0。

    use_cache=True 时走 KV cache:1 次 prefill + 每步只算 1 个新 token(而非重算整段)。
    只在"所有 prompt 等长(无 pad)"且"累计长度不超过 context_size"时生效,
    否则退回整窗重算 —— 两条路径的输出逐 token 完全相同(有测试钉住)。
    """
    assert temperature > 0, "temperature 必须 > 0"
    assert 0 < top_p <= 1.0, "top_p 必须落在 (0, 1]"
    assert top_k is None or top_k > 0, "top_k 必须 > 0 或为 None"

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
    finished = torch.zeros(len(encoded), dtype=torch.bool, device=device)
    kept_counts: list[float] = []

    model.eval()

    # 何时能用 cache:
    #   1) 没有 padding —— 右填充的 pad(0) 会真的进 attention,同一批里长 prompt 会被污染;
    #   2) 绝对位置 < context_size —— RoPE 查表长度就是 context_length,超了就取越界。
    cache_ok = use_cache and max_len <= context_size and all(len(e) == max_len for e in encoded)
    kv = None
    if cache_ok:
        logits, kv = model(ids, use_cache=True)
        logits = logits[:, -1, :].clone()
    else:
        logits = model(ids[:, -context_size:])[:, -1, :]

    for _ in range(max_new_tokens):
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)

        # 降序排列一次,top-k 与 top-p 都在这上面切
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        if top_k is not None:
            sorted_probs[:, top_k:] = 0.0
        # top-p 过滤:保留累计概率达到 p 的最小集合(含跨过阈值的那一个)
        cum = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cum - sorted_probs  # 该 token 之前的累计概率
        sorted_probs = sorted_probs * (cutoff < top_p)
        kept_counts.append(float((sorted_probs > 0).sum(dim=-1).float().mean()))
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        next_sorted = torch.multinomial(sorted_probs, num_samples=1)  # [B,1]
        next_ids = sorted_idx.gather(1, next_sorted)  # [B,1]
        ids = torch.cat([ids, next_ids], dim=1)

        if eos_id is not None:
            finished |= next_ids.squeeze(1) == eos_id
            if bool(finished.all()):
                break

        # 算下一步的 logits:能续 cache 就只算 1 个 token,否则重算整窗
        total = ids.shape[1]
        if cache_ok and kv is not None and total <= context_size:
            logits, kv = model(
                next_ids,
                token_positions=torch.tensor([total - 1], device=device),
                past_kv=kv,
                use_cache=True,
            )
            logits = logits[:, -1, :].clone()
        else:
            kv = None  # 退回整窗重算:位置重新从 0 数,与 cache 路径等价
            logits = model(ids[:, -context_size:])[:, -1, :]

    out = [ids[i, : lengths[i] + max_new_tokens].tolist() for i in range(len(prompt_list))]
    n_kept = kept_counts or [0.0]
    mean_kept = sum(n_kept) / len(n_kept)
    var = sum((x - mean_kept) ** 2 for x in n_kept) / len(n_kept)
    stats = {"avg_candidates": mean_kept, "std_candidates": var**0.5}
    return out, lengths, stats


def generate(
    prompts: str | list[str],
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    device: str | torch.device = "cpu",
    context_size: int = 256,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int | None = None,
    eos_id: int | None = None,
    use_cache: bool = True,
) -> list[str]:
    """给一个或多个 prompt 续写;返回解码后的完整文本(含 prompt)。"""
    ids, _, _stats = generate_ids(
        prompts, model, tokenizer,
        device=device, context_size=context_size, max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p, top_k=top_k, eos_id=eos_id,
        use_cache=use_cache,
    )
    return [tokenizer.decode(seq) for seq in ids]
