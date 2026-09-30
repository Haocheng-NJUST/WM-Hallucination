"""Prompt construction with chat-template support."""

from __future__ import annotations

from .data_io import RAGSample


def messages_for_sample(sample: RAGSample) -> list[dict[str, str]]:
    user_content = f"Context:\n{sample.context}\n\nQuestion:\n{sample.query}"
    return [
        {"role": "system", "content": sample.system_prompt},
        {"role": "user", "content": user_content},
    ]


def build_prompt(tokenizer, sample: RAGSample) -> str:
    messages = messages_for_sample(sample)
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    system = sample.system_prompt.strip()
    context = sample.context.strip()
    query = sample.query.strip()
    return f"{system}\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
