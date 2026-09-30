"""Detectability statistics from final decoded text retokenization."""

from __future__ import annotations

from .watermarks import BaseWatermark, DetectionStats


def detect_token_ids(token_ids: list[int], tokenizer, watermark: BaseWatermark, threshold: float | None = None) -> dict:
    """Score an already detector-tokenized sequence."""
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        vocab_size = len(tokenizer)
    vocab_size = int(vocab_size)
    stats = watermark.detect_token_ids(token_ids, vocab_size, threshold=threshold)
    return stats.to_dict()


def detector_token_ids(text: str, tokenizer) -> list[int]:
    """Retokenize final decoded generation with the detector tokenizer."""
    return [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]


def detect_generated_text(
    text: str,
    tokenizer,
    watermark: BaseWatermark,
    threshold: float | None = None,
    model=None,
    generation_config=None,
    prompt: str | None = None,
) -> dict:
    ids = detector_token_ids(text, tokenizer)
    if hasattr(watermark, "detect_text"):
        stats = watermark.detect_text(
            text,
            tokenizer=tokenizer,
            model=model,
            threshold=threshold,
            generation_config=generation_config,
            prompt=prompt,
        )
        return stats.to_dict()
    return detect_token_ids(ids, tokenizer, watermark, threshold=threshold)
