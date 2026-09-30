"""Watermark implementations included in the clean release."""

from __future__ import annotations

import hashlib
import inspect
import math
from dataclasses import dataclass

import numpy as np
import torch
from .sampling import preprocess_logits


def _stable_seed(parts: list[int | str], base_seed: int) -> int:
    payload = "|".join(str(p) for p in [base_seed, *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


@dataclass
class DetectionStats:
    score: float | None
    z_score: float | None
    num_tokens: int
    detected: bool | None
    threshold: float | None
    metadata: dict | None = None

    def to_dict(self) -> dict[str, float | int | bool | None]:
        result = {
            "score": self.score,
            "z_score": self.z_score,
            "num_tokens": self.num_tokens,
            "detected": self.detected,
            "threshold": self.threshold,
        }
        if self.metadata:
            result.update(self.metadata)
        return result


class BaseWatermark:
    name = "base"

    def __init__(self, gamma: float = 0.25, base_seed: int = 42):
        self.gamma = float(gamma)
        self.base_seed = int(base_seed)

    def _context_seed(self, input_ids: torch.Tensor) -> int:
        if input_ids.shape[1] == 0:
            return self.base_seed
        previous_token_id = int(input_ids[0, -1].item())
        return _stable_seed([previous_token_id], self.base_seed)

    def _greenlist(self, previous_token_id: int | None, vocab_size: int) -> np.ndarray:
        seed = _stable_seed(["green", previous_token_id if previous_token_id is not None else -1], self.base_seed)
        rng = np.random.RandomState(seed)
        green_size = max(1, min(int(vocab_size * self.gamma), vocab_size - 1))
        return rng.choice(vocab_size, size=green_size, replace=False)

    def apply(self, logits: torch.Tensor, input_ids: torch.Tensor, tokenizer=None) -> torch.Tensor:
        raise NotImplementedError

    def select_next_token(self, logits: torch.Tensor, input_ids: torch.Tensor, tokenizer=None, **sampling_kwargs):
        return None

    def detect_token_ids(self, token_ids: list[int], vocab_size: int, threshold: float | None = None) -> DetectionStats:
        if len(token_ids) < 2:
            return DetectionStats(0.0, 0.0, len(token_ids), None if threshold is None else False, threshold)

        green_count = 0
        total = 0
        for previous, current in zip(token_ids[:-1], token_ids[1:]):
            if current in set(self._greenlist(previous, vocab_size).tolist()):
                green_count += 1
            total += 1

        ratio = green_count / max(total, 1)
        std = math.sqrt(self.gamma * (1.0 - self.gamma) / max(total, 1))
        z_score = (ratio - self.gamma) / std if std > 0 else 0.0
        return DetectionStats(ratio, z_score, total, None if threshold is None else ratio >= threshold, threshold)


class KGWWatermark(BaseWatermark):
    name = "KGW"

    def __init__(self, gamma: float = 0.25, delta: float = 2.0, base_seed: int = 42):
        super().__init__(gamma=gamma, base_seed=base_seed)
        self.delta = float(delta)

    def apply(self, logits: torch.Tensor, input_ids: torch.Tensor, tokenizer=None) -> torch.Tensor:
        vocab_size = logits.shape[-1]
        previous = int(input_ids[0, -1].item()) if input_ids.shape[1] else None
        green = self._greenlist(previous, vocab_size)
        mask = torch.zeros(vocab_size, device=logits.device, dtype=logits.dtype)
        mask[torch.tensor(green, device=logits.device, dtype=torch.long)] = self.delta
        return logits + mask.unsqueeze(0)


class SWEETWatermark(KGWWatermark):
    name = "SWEET"

    def __init__(
        self,
        gamma: float = 0.25,
        delta: float = 2.0,
        entropy_threshold: float = 0.695,
        use_normalized_entropy: bool = True,
        base_seed: int = 42,
    ):
        super().__init__(gamma=gamma, delta=delta, base_seed=base_seed)
        self.entropy_threshold = float(entropy_threshold)
        self.use_normalized_entropy = bool(use_normalized_entropy)

    def apply(self, logits: torch.Tensor, input_ids: torch.Tensor, tokenizer=None) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
        if self.use_normalized_entropy:
            entropy = entropy / math.log(logits.shape[-1])
        if float(entropy.item()) < self.entropy_threshold:
            return logits
        return super().apply(logits, input_ids, tokenizer=tokenizer)


class GumbelSoftWatermark(BaseWatermark):
    name = "GumbelSoft"

    def __init__(self, ngram: int = 3, tau: float = 1.0, base_seed: int = 42, gamma: float = 0.25):
        super().__init__(gamma=gamma, base_seed=base_seed)
        self.ngram = int(ngram)
        self.tau = float(tau)
        if self.ngram < 1:
            raise ValueError("GumbelSoft requires ngram >= 1")
        if self.tau <= 0:
            raise ValueError("GumbelSoft requires tau > 0")

    def _gumbel(self, vocab_size: int, device: torch.device, context_ids: list[int]) -> torch.Tensor:
        seed = _stable_seed(["gumbel", *context_ids[-self.ngram :]], self.base_seed)
        rng = np.random.RandomState(seed)
        u = rng.uniform(low=1e-8, high=1.0 - 1e-8, size=vocab_size)
        gumbel = -np.log(-np.log(u))
        return torch.tensor(gumbel, device=device, dtype=torch.float32)

    def apply(self, logits: torch.Tensor, input_ids: torch.Tensor, tokenizer=None) -> torch.Tensor:
        return logits

    def select_next_token(self, logits: torch.Tensor, input_ids: torch.Tensor, tokenizer=None, **sampling_kwargs):
        context_ids = input_ids[0].tolist()
        filtered = preprocess_logits(logits, sampling_kwargs)
        gumbel = self._gumbel(logits.shape[-1], logits.device, context_ids).unsqueeze(0).to(logits.dtype)
        scores = torch.log_softmax(filtered, dim=-1) + self.tau * gumbel
        return torch.argmax(scores, dim=-1, keepdim=True)

    def select_next_token_with_fpti(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        factual_ids: set[int],
        tokenizer=None,
        **sampling_kwargs,
    ):
        context_ids = input_ids[0].tolist()
        gumbel = self._gumbel(logits.shape[-1], logits.device, context_ids).unsqueeze(0).to(logits.dtype)
        filtered = preprocess_logits(logits, sampling_kwargs)
        clean_scores = torch.log_softmax(filtered, dim=-1)
        gumbel_scores = clean_scores + self.tau * gumbel
        if factual_ids:
            valid_ids = [token_id for token_id in factual_ids if 0 <= token_id < logits.shape[-1]]
            if valid_ids:
                gumbel_scores[:, valid_ids] = clean_scores[:, valid_ids]
        return torch.argmax(gumbel_scores, dim=-1, keepdim=True)

    def detect_token_ids(self, token_ids: list[int], vocab_size: int, threshold: float | None = None) -> DetectionStats:
        if len(token_ids) < self.ngram + 1:
            return DetectionStats(0.0, 0.0, len(token_ids), None if threshold is None else False, threshold)

        ranks = []
        for idx in range(self.ngram, len(token_ids)):
            context = token_ids[:idx]
            current = token_ids[idx]
            gumbel = self._gumbel(vocab_size, torch.device("cpu"), context)
            rank_fraction = float((gumbel <= gumbel[current]).sum().item()) / float(vocab_size)
            ranks.append(rank_fraction)

        mean_rank = float(np.mean(ranks)) if ranks else 0.0
        expected = 0.5
        std = math.sqrt(1.0 / 12.0 / max(len(ranks), 1))
        z_score = (mean_rank - expected) / std if std > 0 else 0.0
        return DetectionStats(mean_rank, z_score, len(ranks), None if threshold is None else mean_rank >= threshold, threshold)


class GumbelMaxWatermark(BaseWatermark):
    name = "GumbelMax"

    def __init__(self, ngram: int = 3, scale: float = 1.0, base_seed: int = 42, gamma: float = 0.25):
        super().__init__(gamma=gamma, base_seed=base_seed)
        self.ngram = int(ngram)
        self.scale = float(scale)
        if self.ngram < 1:
            raise ValueError("GumbelMax requires ngram >= 1")
        if self.scale <= 0:
            raise ValueError("GumbelMax requires scale > 0")

    def _gumbel(self, vocab_size: int, device: torch.device, context_ids: list[int]) -> torch.Tensor:
        seed = _stable_seed(["gumbelmax", *context_ids[-self.ngram :]], self.base_seed)
        rng = np.random.RandomState(seed)
        u = rng.uniform(low=1e-8, high=1.0 - 1e-8, size=vocab_size)
        gumbel = -np.log(-np.log(u))
        return torch.tensor(gumbel, device=device, dtype=torch.float32)

    def apply(self, logits: torch.Tensor, input_ids: torch.Tensor, tokenizer=None) -> torch.Tensor:
        return logits

    def select_next_token(self, logits: torch.Tensor, input_ids: torch.Tensor, tokenizer=None, **sampling_kwargs):
        context_ids = input_ids[0].tolist()
        filtered = preprocess_logits(logits, sampling_kwargs)
        gumbel = self._gumbel(logits.shape[-1], logits.device, context_ids).unsqueeze(0).to(logits.dtype)
        scores = torch.log_softmax(filtered, dim=-1) + self.scale * gumbel
        return torch.argmax(scores, dim=-1, keepdim=True)

    def select_next_token_with_fpti(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        factual_ids: set[int],
        tokenizer=None,
        **sampling_kwargs,
    ):
        context_ids = input_ids[0].tolist()
        gumbel = self._gumbel(logits.shape[-1], logits.device, context_ids).unsqueeze(0).to(logits.dtype)
        filtered = preprocess_logits(logits, sampling_kwargs)
        clean_scores = torch.log_softmax(filtered, dim=-1)
        watermarked_scores = clean_scores + self.scale * gumbel
        if factual_ids:
            valid_ids = [token_id for token_id in factual_ids if 0 <= token_id < logits.shape[-1]]
            if valid_ids:
                watermarked_scores[:, valid_ids] = clean_scores[:, valid_ids]
        return torch.argmax(watermarked_scores, dim=-1, keepdim=True)

    def detect_token_ids(self, token_ids: list[int], vocab_size: int, threshold: float | None = None) -> DetectionStats:
        if len(token_ids) < self.ngram + 1:
            return DetectionStats(0.0, 0.0, len(token_ids), None if threshold is None else False, threshold)

        ranks = []
        for idx in range(self.ngram, len(token_ids)):
            context = token_ids[:idx]
            current = token_ids[idx]
            gumbel = self._gumbel(vocab_size, torch.device("cpu"), context)
            rank_fraction = float((gumbel <= gumbel[current]).sum().item()) / float(vocab_size)
            ranks.append(rank_fraction)

        mean_rank = float(np.mean(ranks)) if ranks else 0.0
        expected = 0.5
        std = math.sqrt(1.0 / 12.0 / max(len(ranks), 1))
        z_score = (mean_rank - expected) / std if std > 0 else 0.0
        return DetectionStats(mean_rank, z_score, len(ranks), None if threshold is None else mean_rank >= threshold, threshold)


class SynthIDStyleWatermark(BaseWatermark):
    name = "SynthIDStyle"

    def __init__(
        self,
        ngram: int = 3,
        strength: float = 1.0,
        key: str = "synthid-style-key",
        base_seed: int = 42,
        gamma: float = 0.25,
    ):
        super().__init__(gamma=gamma, base_seed=base_seed)
        self.ngram = int(ngram)
        self.strength = float(strength)
        self.key = str(key)
        if self.ngram < 1:
            raise ValueError("SynthIDStyle requires ngram >= 1")
        if self.strength <= 0:
            raise ValueError("SynthIDStyle requires strength > 0")

    def _keyed_scores(self, vocab_size: int, device: torch.device, context_ids: list[int]) -> torch.Tensor:
        seed = _stable_seed(["synthid", self.key, *context_ids[-self.ngram :]], self.base_seed)
        rng = np.random.RandomState(seed)
        scores = rng.uniform(low=0.0, high=1.0, size=vocab_size)
        return torch.tensor(scores, device=device, dtype=torch.float32)

    def apply(self, logits: torch.Tensor, input_ids: torch.Tensor, tokenizer=None) -> torch.Tensor:
        return logits

    def select_next_token(self, logits: torch.Tensor, input_ids: torch.Tensor, tokenizer=None, **sampling_kwargs):
        context_ids = input_ids[0].tolist()
        keyed = self._keyed_scores(logits.shape[-1], logits.device, context_ids).unsqueeze(0).to(logits.dtype)
        scores = torch.log_softmax(logits, dim=-1) + self.strength * keyed
        return torch.argmax(scores, dim=-1, keepdim=True)

    def select_next_token_with_fpti(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        factual_ids: set[int],
        tokenizer=None,
        **sampling_kwargs,
    ):
        context_ids = input_ids[0].tolist()
        keyed = self._keyed_scores(logits.shape[-1], logits.device, context_ids).unsqueeze(0).to(logits.dtype)
        clean_scores = torch.log_softmax(logits, dim=-1)
        watermarked_scores = clean_scores + self.strength * keyed
        if factual_ids:
            valid_ids = [token_id for token_id in factual_ids if 0 <= token_id < logits.shape[-1]]
            if valid_ids:
                watermarked_scores[:, valid_ids] = clean_scores[:, valid_ids]
        return torch.argmax(watermarked_scores, dim=-1, keepdim=True)

    def detect_token_ids(self, token_ids: list[int], vocab_size: int, threshold: float | None = None) -> DetectionStats:
        if len(token_ids) < self.ngram + 1:
            return DetectionStats(0.0, 0.0, len(token_ids), None if threshold is None else False, threshold)

        scores = []
        for idx in range(self.ngram, len(token_ids)):
            context = token_ids[:idx]
            current = token_ids[idx]
            keyed = self._keyed_scores(vocab_size, torch.device("cpu"), context)
            scores.append(float(keyed[current].item()))

        mean_score = float(np.mean(scores)) if scores else 0.0
        expected = 0.5
        std = math.sqrt(1.0 / 12.0 / max(len(scores), 1))
        z_score = (mean_score - expected) / std if std > 0 else 0.0
        return DetectionStats(mean_score, z_score, len(scores), None if threshold is None else mean_score >= threshold, threshold)


class SynthIDWatermark(BaseWatermark):
    name = "SynthID"

    def __init__(
        self,
        ngram_len: int = 5,
        key_num: int = 9,
        sampling_table_size: int = 65536,
        sampling_table_seed: int = 0,
        context_history_size: int = 1024,
        gamma: float = 0.25,
        base_seed: int = 42,
    ):
        super().__init__(gamma=gamma, base_seed=base_seed)
        self.ngram_len = int(ngram_len)
        self.key_num = int(key_num)
        self.sampling_table_size = int(sampling_table_size)
        self.sampling_table_seed = int(sampling_table_seed)
        self.context_history_size = int(context_history_size)
        if self.ngram_len < 1:
            raise ValueError("SynthID requires synthid_ngram_len >= 1")
        if self.key_num < 1:
            raise ValueError("SynthID requires synthid_key_num >= 1")

    def synthid_keys(self) -> list[int]:
        rng = np.random.RandomState(self.sampling_table_seed)
        return rng.randint(0, 2**31 - 1, size=self.key_num).astype(int).tolist()

    def build_config(self):
        try:
            from transformers import SynthIDTextWatermarkingConfig
        except ImportError as exc:
            raise RuntimeError(
                "Official SynthID generation requires transformers.SynthIDTextWatermarkingConfig. "
                "Install or upgrade transformers to a version that provides it."
            ) from exc

        return SynthIDTextWatermarkingConfig(
            ngram_len=self.ngram_len,
            keys=self.synthid_keys(),
            sampling_table_size=self.sampling_table_size,
            sampling_table_seed=self.sampling_table_seed,
            context_history_size=self.context_history_size,
        )

    def build_logits_processor(self, device: torch.device):
        processor_cls = get_synthid_logits_processor_class()
        if processor_cls is None:
            raise ValueError(
                "Official SynthID FPTI/FPAI requires access to the SynthID logits processor. "
                "This Transformers version does not expose a compatible processor. Use SynthIDStyle "
                "for approximate score-level intervention or standard SynthID for official generation."
            )

        kwargs = {
            "ngram_len": self.ngram_len,
            "keys": self.synthid_keys(),
            "sampling_table_size": self.sampling_table_size,
            "sampling_table_seed": self.sampling_table_seed,
            "context_history_size": self.context_history_size,
            "device": device,
        }
        try:
            signature = inspect.signature(processor_cls)
            accepted = {
                key: value
                for key, value in kwargs.items()
                if key in signature.parameters
            }
            return processor_cls(**accepted)
        except Exception as exc:
            raise ValueError(
                "Official SynthID FPTI/FPAI requires access to the SynthID logits processor. "
                "This Transformers version does not expose a compatible processor. Use SynthIDStyle "
                "for approximate score-level intervention or standard SynthID for official generation."
            ) from exc

    def apply(self, logits: torch.Tensor, input_ids: torch.Tensor, tokenizer=None) -> torch.Tensor:
        return logits

    def detect_token_ids(self, token_ids: list[int], vocab_size: int, threshold: float | None = None) -> DetectionStats:
        return DetectionStats(None, None, len(token_ids), None, threshold)


def get_synthid_logits_processor_class():
    try:
        from transformers import SynthIDTextWatermarkLogitsProcessor

        return SynthIDTextWatermarkLogitsProcessor
    except Exception:
        pass

    try:
        from transformers.generation.logits_process import SynthIDTextWatermarkLogitsProcessor

        return SynthIDTextWatermarkLogitsProcessor
    except Exception:
        return None


def official_synthid_intervention_availability() -> dict[str, str | bool]:
    try:
        from transformers import SynthIDTextWatermarkingConfig  # noqa: F401
    except Exception as exc:
        return {
            "available": False,
            "reason": f"Transformers SynthIDTextWatermarkingConfig is unavailable: {exc}",
        }

    processor_cls = get_synthid_logits_processor_class()
    if processor_cls is None:
        return {
            "available": False,
            "reason": (
                "Transformers SynthIDTextWatermarkingConfig is available, but no compatible "
                "SynthIDTextWatermarkLogitsProcessor is exposed."
            ),
        }

    return {
        "available": True,
        "reason": f"Compatible processor found: {processor_cls.__module__}.{processor_cls.__name__}",
    }


def create_watermark(method: str, **kwargs) -> BaseWatermark:
    normalized = method.strip()
    if normalized == "KGW":
        return KGWWatermark(
            gamma=kwargs.get("gamma", 0.25),
            delta=kwargs.get("delta", 2.0),
            base_seed=kwargs.get("base_seed", 42),
        )
    if normalized == "SWEET":
        return SWEETWatermark(
            gamma=kwargs.get("gamma", 0.25),
            delta=kwargs.get("delta", 2.0),
            entropy_threshold=kwargs.get("sweet_entropy_threshold", 0.695),
            use_normalized_entropy=kwargs.get("sweet_use_normalized_entropy", True),
            base_seed=kwargs.get("base_seed", 42),
        )
    if normalized == "DiPmark":
        from .official_watermarks import DiPmarkWatermark
        return DiPmarkWatermark(
            gamma=kwargs.get("dip_gamma", kwargs.get("gamma", 0.5)),
            alpha=kwargs.get("dip_alpha", 0.45),
            prefix_length=kwargs.get("dip_prefix_length", 5),
            base_seed=kwargs.get("base_seed", 42),
        )
    if normalized == "GumbelSoft":
        return GumbelSoftWatermark(
            ngram=kwargs.get("gsoft_ngram", 3),
            tau=kwargs.get("gsoft_tau", 1.0),
            gamma=kwargs.get("gamma", 0.25),
            base_seed=kwargs.get("base_seed", 42),
        )
    if normalized == "GumbelMax":
        return GumbelMaxWatermark(
            ngram=kwargs.get("gmax_ngram", 3),
            scale=kwargs.get("gmax_scale", 1.0),
            gamma=kwargs.get("gamma", 0.25),
            base_seed=kwargs.get("base_seed", 42),
        )
    if normalized == "SynthIDStyle":
        return SynthIDStyleWatermark(
            ngram=kwargs.get("synthid_ngram", 3),
            strength=kwargs.get("synthid_strength", 1.0),
            key=kwargs.get("synthid_key", "synthid-style-key"),
            gamma=kwargs.get("gamma", 0.25),
            base_seed=kwargs.get("base_seed", 42),
        )
    if normalized == "SynthID":
        return SynthIDWatermark(
            ngram_len=kwargs.get("synthid_ngram_len", 5),
            key_num=kwargs.get("synthid_key_num", 9),
            sampling_table_size=kwargs.get("synthid_sampling_table_size", 65536),
            sampling_table_seed=kwargs.get("synthid_sampling_table_seed", 0),
            context_history_size=kwargs.get("synthid_context_history_size", 1024),
            gamma=kwargs.get("gamma", 0.25),
            base_seed=kwargs.get("base_seed", 42),
        )
    if normalized == "UnbiasedWatermark":
        from .official_watermarks import UnbiasedWatermark
        return UnbiasedWatermark(prefix_length=kwargs.get("unbiased_prefix_length", 5), base_seed=kwargs.get("base_seed", 42))
    if normalized == "TextSeal":
        from .official_watermarks import TextSealWatermark
        return TextSealWatermark(ngram=kwargs.get("textseal_ngram", 2), mixing_alpha=kwargs.get("textseal_mixing_alpha", 0.5), key=kwargs.get("textseal_key", 42), scoring_method=kwargs.get("textseal_scoring_method", "v2"), base_seed=kwargs.get("base_seed", 42))
    if normalized == "MorphMark":
        from .official_watermarks import MorphMarkWatermark
        return MorphMarkWatermark(gamma=kwargs.get("morphmark_gamma", 0.5), p_0=kwargs.get("morphmark_p0", 0.15), k_exp=kwargs.get("morphmark_k_exp", 1.30), prefix_length=kwargs.get("morphmark_prefix_length", 1), hash_key=kwargs.get("morphmark_hash_key", 15485863), base_seed=kwargs.get("base_seed", 42))
    raise ValueError(f"Unknown watermark method: {method}")


SUPPORTED_METHODS = ("KGW", "SWEET", "DiPmark", "GumbelSoft", "GumbelMax", "SynthIDStyle", "SynthID", "UnbiasedWatermark", "TextSeal", "MorphMark")
