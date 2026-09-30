"""JSONL loading and validation for anonymized RAG samples."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REQUIRED_SAMPLE_FIELDS = {
    "sample_id",
    "fact_type",
    "domain",
    "system_prompt",
    "context",
    "query",
    "target_facts",
}
ALLOWED_FACT_TYPES = {"numerical", "identifier", "temporal", "qualitative"}

FORBIDDEN_PATTERNS = [
    re.compile(r"\b[A-Z]:\\"),
    re.compile(r"(?<!\w)/(?:Users|home|mnt|var|tmp)/"),
    re.compile(r"https?://(?:www\.)?github\.com", re.IGNORECASE),
    re.compile(r"openreview", re.IGNORECASE),
    re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
]


@dataclass(frozen=True)
class RAGSample:
    sample_id: str
    fact_type: str
    domain: str
    system_prompt: str
    context: str
    query: str
    target_facts: list[str]

    @classmethod
    def from_dict(cls, obj: dict[str, Any], line_number: int | None = None) -> "RAGSample":
        missing = REQUIRED_SAMPLE_FIELDS.difference(obj)
        where = f" on line {line_number}" if line_number is not None else ""
        if missing:
            raise ValueError(f"Missing required field(s){where}: {sorted(missing)}")

        target_facts = obj["target_facts"]
        if not isinstance(target_facts, list) or not all(isinstance(x, str) for x in target_facts):
            raise ValueError(f"`target_facts` must be a list of strings{where}")
        fact_type = str(obj["fact_type"])
        if fact_type not in ALLOWED_FACT_TYPES:
            raise ValueError(
                f"`fact_type` must be one of {sorted(ALLOWED_FACT_TYPES)}{where}; got {fact_type!r}"
            )

        sample = cls(
            sample_id=str(obj["sample_id"]),
            fact_type=fact_type,
            domain=str(obj["domain"]),
            system_prompt=str(obj["system_prompt"]),
            context=str(obj["context"]),
            query=str(obj["query"]),
            target_facts=target_facts,
        )
        scan_for_forbidden_text(sample.to_dict(), where=where)
        return sample

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "fact_type": self.fact_type,
            "domain": self.domain,
            "system_prompt": self.system_prompt,
            "context": self.context,
            "query": self.query,
            "target_facts": self.target_facts,
        }


def scan_for_forbidden_text(obj: Any, where: str = "") -> None:
    text = json.dumps(obj, ensure_ascii=False)
    for pattern in FORBIDDEN_PATTERNS:
        if pattern.search(text):
            raise ValueError(f"Potential deanonymizing text found{where}: {pattern.pattern}")


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc


def load_rag_samples(path: str | Path) -> list[RAGSample]:
    samples = []
    seen = set()
    for line_number, obj in enumerate(read_jsonl(path), start=1):
        sample = RAGSample.from_dict(obj, line_number=line_number)
        if sample.sample_id in seen:
            raise ValueError(f"Duplicate sample_id on line {line_number}: {sample.sample_id}")
        seen.add(sample.sample_id)
        samples.append(sample)
    return samples


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            scan_for_forbidden_text(row)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
