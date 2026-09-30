"""Sampling helpers used by generation and watermark implementations."""

from __future__ import annotations

import torch


def preprocess_logits(logits: torch.Tensor, sampling_kwargs: dict | None = None) -> torch.Tensor:
    """Apply the shared decoding order before method-specific token selection."""
    kwargs = sampling_kwargs or {}
    temperature = float(kwargs.get("temperature", 1.0) or 1.0)
    filtered = logits / temperature if temperature > 0 else logits
    return top_k_top_p_filtering(filtered, top_k=int(kwargs.get("top_k", 0) or 0), top_p=float(kwargs.get("top_p", 1.0) or 1.0))


def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    filtered = logits.clone()

    if top_k and top_k > 0:
        top_k = min(top_k, filtered.size(-1))
        threshold = torch.topk(filtered, top_k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))

    if top_p and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask = torch.zeros_like(filtered, dtype=torch.bool).scatter(-1, sorted_indices, sorted_mask)
        filtered = filtered.masked_fill(mask, float("-inf"))

    return filtered


def apply_repetition_penalty(logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty <= 1.0:
        return logits
    adjusted = logits.clone()
    for token_id in set(generated_ids[0].tolist()):
        if adjusted[0, token_id] > 0:
            adjusted[0, token_id] /= penalty
        else:
            adjusted[0, token_id] *= penalty
    return adjusted


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    do_sample: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if temperature and temperature > 0:
        logits = logits / temperature

    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)

    probs = torch.softmax(logits, dim=-1)
    invalid_probs = ~torch.isfinite(probs)
    zero_sums = probs.sum(dim=-1, keepdim=True) <= 0
    if invalid_probs.any() or zero_sums.any():
        probs = torch.ones_like(probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1, generator=generator)
