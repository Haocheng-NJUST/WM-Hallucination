"""Context-only fact span extraction for FPTI/FPAI."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class FactSpan:
    text: str
    start: int
    end: int
    label: str
    source: str


_PATTERNS = (
    ("temporal", re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:19|20)\d{2}|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|January|February|March|April|May|June|July|August|September|October|November|December)\b", re.I)),
    ("numerical", re.compile(r"(?<!\w)(?:[$€£]\s*)?\d+(?:[.,]\d+)?%?(?:\s*(?:kg|g|ms|s|minutes?|hours?|requests?|units?))?\b", re.I)),
    ("identifier", re.compile(r"\b(?:[A-Z]{2,}[\w-]*|[A-Za-z]+[-_]\d+|[A-Z]?\d{2,}[A-Za-z]?)\b")),
)


def _deduplicate(spans: list[FactSpan]) -> list[FactSpan]:
    selected: list[FactSpan] = []
    for span in sorted(spans, key=lambda item: (item.start, -(item.end - item.start), item.label)):
        if any(span.start < other.end and other.start < span.end for other in selected):
            continue
        selected.append(span)
    return sorted(selected, key=lambda item: item.start)


def extract_context_facts(context: str, spacy_model: str | None = None) -> list[FactSpan]:
    """Extract spans only from context; evaluation answers are never consulted."""
    spans: list[FactSpan] = []
    for label, pattern in _PATTERNS:
        spans.extend(FactSpan(m.group(0), m.start(), m.end(), label, "regex") for m in pattern.finditer(context))

    if spacy_model:
        try:
            import spacy
            nlp = spacy.load(spacy_model)
        except Exception as exc:
            raise RuntimeError(f"Unable to load spaCy model {spacy_model!r}: {exc}") from exc
        for entity in nlp(context).ents:
            label = "temporal" if entity.label_ in {"DATE", "TIME"} else "numerical" if entity.label_ in {"CARDINAL", "MONEY", "QUANTITY", "PERCENT"} else "identifier" if entity.label_ in {"PRODUCT", "ORG", "GPE", "LOC", "PERSON", "NORP", "FAC"} else "qualitative"
            spans.append(FactSpan(entity.text, entity.start_char, entity.end_char, label, f"spacy:{spacy_model}"))
    return _deduplicate(spans)

