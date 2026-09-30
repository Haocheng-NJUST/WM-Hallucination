"""Small adapters derived from the supplied upstream implementations.

Sources: MarkLLM commit 0a4fe8c642b31b8a8c4993615b1c70a0c300d918 and
TextSeal commit c60d0d1da2e59f09a698438e218a07ee779b4616.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import special

from .sampling import preprocess_logits
from .watermarks import BaseWatermark, DetectionStats


def _seed(context: list[int], key: int, tag: str) -> int:
    payload = f"{tag}|{key}|".encode() + np.asarray(context, dtype=np.int64).tobytes()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def _markllm_seed(context: list[int], key: int) -> int:
    payload = np.asarray(context, dtype=np.int64).tobytes() + str(key).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big") % (2**32 - 1)


class DiPmarkWatermark(BaseWatermark):
    """DiPmark alpha-reweighting adapter based on the supplied MarkLLM DIP code."""

    name = "DiPmark"

    def __init__(self, gamma: float = 0.5, alpha: float = 0.45, prefix_length: int = 5, base_seed: int = 42, **kwargs):
        super().__init__(gamma=gamma, base_seed=base_seed)
        self.alpha = float(alpha)
        self.prefix_length = int(prefix_length)
        self.ignore_history = bool(kwargs.get("dip_ignore_history", False))
        self.cc_history: set[bytes] = set()
        self.config = {"method": self.name, "gamma": self.gamma, "alpha": self.alpha, "prefix_length": self.prefix_length, "base_seed": self.base_seed, "source_revision": "DiPmark@34abbeb527243c79bda8043313bb797a731f4ae7"}

    def _context_code(self, input_ids: torch.Tensor) -> bytes:
        context = input_ids[0].tolist()[-self.prefix_length:] if self.prefix_length else input_ids[0].tolist()
        return np.asarray(context, dtype=np.int64).tobytes()

    def reset_history(self) -> None:
        self.cc_history.clear()

    def _shuffle(self, input_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
        context = input_ids[0].tolist()[-self.prefix_length:] if self.prefix_length else input_ids[0].tolist()
        gen = torch.Generator(device=input_ids.device).manual_seed(_markllm_seed(context, self.base_seed))
        return torch.randperm(vocab_size, generator=gen, device=input_ids.device)

    def apply(self, logits: torch.Tensor, input_ids: torch.Tensor, tokenizer=None) -> torch.Tensor:
        if input_ids.shape[1] < self.prefix_length:
            return logits
        context_code = self._context_code(input_ids)
        repeated = context_code in self.cc_history
        if not self.ignore_history:
            self.cc_history.add(context_code)
        if repeated and not self.ignore_history:
            return logits
        shuffle = self._shuffle(input_ids, logits.shape[-1])
        unshuffle = torch.argsort(shuffle)
        ordered = logits[:, shuffle]
        log_cumsum = torch.logcumsumexp(ordered, dim=-1)
        log_cumsum = log_cumsum - log_cumsum[..., -1:]
        cumsum = log_cumsum.exp()
        probs = ordered.softmax(-1)
        result = torch.zeros_like(ordered)
        for boundary in (self.alpha, 1.0 - self.alpha):
            index = (cumsum > boundary).to(torch.int).argmax(-1, keepdim=True)
            p_boundary = probs.gather(-1, index).clamp_min(1e-12)
            portion = ((cumsum.gather(-1, index) - boundary) / p_boundary).clamp(0, 1)
            mask = (cumsum > boundary).to(logits.dtype)
            result = result + mask.scatter(-1, index, portion)
        result = torch.log((result / 2).clamp_min(1e-12)).gather(-1, unshuffle.expand_as(logits))
        return logits + result

    def detect_token_ids(self, token_ids: list[int], vocab_size: int, threshold: float | None = None) -> DetectionStats:
        if len(token_ids) <= self.prefix_length:
            return DetectionStats(0.0, 0.0, 0, None if threshold is None else False, threshold)
        quantiles = []
        history: set[bytes] = set()
        for index in range(self.prefix_length, len(token_ids)):
            context = torch.tensor([token_ids[:index]], dtype=torch.long)
            code = np.asarray(token_ids[max(0, index - self.prefix_length):index], dtype=np.int64).tobytes()
            if not self.ignore_history and code in history:
                continue
            history.add(code)
            shuffle = self._shuffle(context, vocab_size).cpu()
            rank = int((shuffle == token_ids[index]).nonzero(as_tuple=False)[0].item()) + 1
            quantiles.append(rank / vocab_size)
        if not quantiles:
            return DetectionStats(0.0, 0.0, 0, None if threshold is None else False, threshold)
        green_count = sum(value >= self.gamma for value in quantiles)
        green_fraction = green_count / len(quantiles)
        z_score = (green_count - (1 - self.gamma) * len(quantiles)) / math.sqrt(len(quantiles))
        return DetectionStats(z_score, z_score, len(quantiles), None if threshold is None else z_score >= threshold, threshold, {"green_fraction": green_fraction, "quantile_threshold": self.gamma, "green_count": green_count})


class UnbiasedWatermark(BaseWatermark):
    """Official delta-reweighting strategy from the supplied Unbiased Watermark source."""

    name = "UnbiasedWatermark"

    def __init__(self, prefix_length: int = 5, base_seed: int = 42, **kwargs):
        super().__init__(gamma=0.5, base_seed=base_seed)
        self.prefix_length = int(prefix_length)
        self.config = {"method": self.name, "strategy": "delta", "prefix_length": self.prefix_length, "base_seed": self.base_seed, "source_revision": "UnbiasedWatermark@050eee18b06e01c95eb175c7eed725758a9ac451"}

    def _uniform(self, input_ids: torch.Tensor) -> float:
        context = input_ids[0].tolist()[-self.prefix_length:] if self.prefix_length else input_ids[0].tolist()
        gen = torch.Generator(device=input_ids.device).manual_seed(_markllm_seed(context, self.base_seed))
        return float(torch.rand((), generator=gen, device=input_ids.device).item())

    def reset_history(self) -> None:
        """Keep the generation/detection lifecycle compatible with stateful adapters."""
        return None

    def apply(self, logits: torch.Tensor, input_ids: torch.Tensor, tokenizer=None) -> torch.Tensor:
        if input_ids.shape[1] < self.prefix_length:
            return logits
        u = self._uniform(input_ids)
        cumsum = torch.cumsum(logits.softmax(-1), dim=-1)
        index = torch.searchsorted(cumsum[0], torch.tensor(u, device=logits.device), right=True).clamp(0, logits.shape[-1] - 1)
        result = torch.full_like(logits, float("-inf"))
        result[:, index] = 0.0
        return result

    def select_next_token(self, logits: torch.Tensor, input_ids: torch.Tensor, tokenizer=None, **sampling_kwargs):
        return torch.argmax(self.apply(preprocess_logits(logits, sampling_kwargs), input_ids), dim=-1, keepdim=True)

    def detect_token_ids(self, token_ids: list[int], vocab_size: int, threshold: float | None = None) -> DetectionStats:
        raise ValueError("UnbiasedWatermark detection requires final text plus model logits; call detect_text")

    @staticmethod
    def _safe_minus(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        value = q - p
        return torch.nan_to_num(value, nan=0.0)

    @classmethod
    def _max_llr(cls, p: torch.Tensor, q: torch.Tensor, dist_p_log: float, dist_q_log: float) -> torch.Tensor:
        llr = cls._safe_minus(q, p)
        order = torch.argsort(llr, descending=True)
        p_sorted, q_sorted = p[order], q[order]
        raw_llr = cls._safe_minus(q_sorted, p_sorted)
        sum_q = torch.logcumsumexp(q_sorted, dim=-1)
        sum_p = torch.logcumsumexp(p_sorted, dim=-1)
        dq = torch.tensor(dist_q_log, device=p.device, dtype=p.dtype)
        dp = torch.tensor(dist_p_log, device=p.device, dtype=p.dtype)
        modified_q = torch.where(sum_q <= dq, torch.full_like(sum_q, -float("inf")), sum_q + torch.log(-torch.expm1(dq - sum_q)))
        modified_p = torch.logaddexp(sum_p, dp)
        modified = cls._safe_minus(modified_q, modified_p)
        padded = F.pad(modified, (1, 0), value=-float("inf"))
        cut = torch.where(torch.any(raw_llr < padded[:-1]), torch.argmax((raw_llr < padded[:-1]).to(torch.int)), torch.tensor(padded.shape[-1] - 1, device=p.device))
        return padded[cut]

    @classmethod
    def _robust_token_score(cls, p: torch.Tensor, q: torch.Tensor, token_id: int, queries: list[tuple[float, float]]) -> torch.Tensor:
        lp, lq = F.log_softmax(p, -1), F.log_softmax(q, -1)
        raw = cls._safe_minus(lq, lp)
        max_values, min_values = [], []
        for dp, dq in queries:
            max_values.append(cls._max_llr(lp, lq, dp, dq))
            min_values.append(-cls._max_llr(lq, lp, dq, dp))
        max_llr = torch.stack(max_values)
        min_llr = torch.stack(min_values)
        trivial = max_llr < min_llr
        return torch.where(trivial, torch.zeros_like(max_llr), raw[token_id].clamp(min_llr, max_llr))

    def detect_text(
        self,
        text: str,
        tokenizer,
        model=None,
        threshold: float | None = None,
        generation_config=None,
        prompt: str | None = None,
    ) -> DetectionStats:
        if model is None:
            raise ValueError("UnbiasedWatermark detection requires a model to compute p_t and q_t")
        completion_ids = [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]
        if not completion_ids:
            raise ValueError("UnbiasedWatermark detection requires a non-empty completion")
        device = next(model.parameters()).device
        if prompt:
            prompt_encoded = tokenizer(prompt, return_tensors="pt")
            prompt_ids = prompt_encoded["input_ids"]
        else:
            prompt_ids = torch.empty((1, 0), dtype=torch.long)
        prompt_ids = prompt_ids.to(device)
        completion_tensor = torch.tensor([completion_ids], dtype=torch.long, device=device)
        ids = torch.cat([prompt_ids, completion_tensor], dim=1)
        prompt_length = int(prompt_ids.shape[1])
        if ids.shape[1] <= 1:
            raise ValueError("UnbiasedWatermark detection requires a model context before the completion")
        with torch.no_grad():
            outputs = model(ids[:, :-1], attention_mask=torch.ones_like(ids[:, :-1]))
            raw_logits = outputs.logits[0]
            self.reset_history()
            q_logits = []
            p_logits = []
            repetition_penalty = float(getattr(generation_config, "repetition_penalty", 1.0) if generation_config is not None else 1.0)
            sampling_kwargs = {
                "temperature": float(getattr(generation_config, "temperature", 1.0) if generation_config is not None else 1.0),
                "top_p": float(getattr(generation_config, "top_p", 1.0) if generation_config is not None else 1.0),
                "top_k": int(getattr(generation_config, "top_k", 0) if generation_config is not None else 0),
                "do_sample": bool(getattr(generation_config, "do_sample", True) if generation_config is not None else True),
            }
            from .sampling import apply_repetition_penalty
            for index in range(raw_logits.shape[0]):
                context = ids[:, : index + 1]
                processed = apply_repetition_penalty(raw_logits[index:index + 1], context, repetition_penalty)
                processed = preprocess_logits(processed, sampling_kwargs)
                p_logits.append(processed[0])
                q_logits.append(self.apply(processed, context)[0])
            p_logits = torch.stack(p_logits)
            q_logits = torch.stack(q_logits)
            queries = [(float("-inf"), math.log(value)) for value in np.linspace(1e-6, 1.0, 10)]
            token_scores = []
            for completion_index, label in enumerate(completion_ids):
                absolute_position = prompt_length + completion_index
                logit_index = absolute_position - 1
                if logit_index < 0 or absolute_position < self.prefix_length:
                    continue
                values = self._robust_token_score(p_logits[logit_index], q_logits[logit_index], label, queries)
                token_scores.append(values)
            if not token_scores:
                raise ValueError("UnbiasedWatermark detector produced no scorable tokens")
            matrix = torch.stack(token_scores)
            totals = matrix.sum(dim=0)
            best_index = int(torch.argmax(totals).item())
            score = float(totals[best_index].item())
        return DetectionStats(score, score, len(token_scores), None if threshold is None else score >= threshold, threshold, {"detector": "official_robust_llr_v2", "query_index": best_index})


def _textseal_uniform(window: list[int], token_ids: torch.Tensor, key: int) -> torch.Tensor:
    primes = torch.tensor([10000019, 10000247, 10000439, 10000643, 10000747, 10000867, 10000993, 10001213, 10001357, 10001501][: len(window)], device=token_ids.device, dtype=torch.long)
    weighted = (torch.tensor(window, device=token_ids.device, dtype=torch.long) * primes).sum()
    x = token_ids.long()
    h = (weighted + 100000007 * x + 500001713 * key) * 15485863 * 40499
    h = h ^ (h >> 13)
    return (h % (2**13 - 1)).float() / (2**13 - 1)


class TextSealWatermark(BaseWatermark):
    """TextSeal generation-time dual-key Gumbel-max adapter."""

    name = "TextSeal"

    def __init__(self, ngram: int = 2, mixing_alpha: float = 0.5, key: int = 42, base_seed: int = 42, scoring_method: str = "v2", **kwargs):
        super().__init__(gamma=0.5, base_seed=base_seed)
        self.ngram = int(ngram)
        self.key_a = int(key)
        self.key_b = self.key_a + 12345
        self.mixing_alpha = float(mixing_alpha)
        if scoring_method not in {"v1", "v2", "none"}:
            raise ValueError("TextSeal scoring_method must be v1, v2, or none")
        self.config = {"method": self.name, "ngram": self.ngram, "key_a": self.key_a, "key_b": self.key_b, "mixing_alpha": self.mixing_alpha, "scoring_method": scoring_method, "source_revision": "TextSeal@c60d0d1da2e59f09a698438e218a07ee779b4616"}

    def _scores(self, input_ids: torch.Tensor, vocab_size: int, generator: torch.Generator | None = None) -> torch.Tensor:
        window = input_ids[0].tolist()[-self.ngram:]
        tokens = torch.arange(vocab_size, device=input_ids.device)
        a = _textseal_uniform(window, tokens, self.key_a)
        b = _textseal_uniform(window, tokens, self.key_b)
        choose_a = bool(torch.rand((), device=input_ids.device, generator=generator).item() < self.mixing_alpha)
        return a if choose_a else b

    def select_next_token(self, logits: torch.Tensor, input_ids: torch.Tensor, tokenizer=None, **sampling_kwargs):
        filtered = preprocess_logits(logits, sampling_kwargs)
        if not sampling_kwargs.get("do_sample", True):
            return torch.argmax(filtered, dim=-1, keepdim=True)
        probs = filtered.softmax(-1).clamp_min(1e-12)
        random_scores = self._scores(input_ids, logits.shape[-1], sampling_kwargs.get("generator")).unsqueeze(0).to(logits.dtype)
        scores = torch.log(random_scores.clamp_min(1e-12)) / probs
        return torch.argmax(scores, dim=-1, keepdim=True)

    def apply(self, logits: torch.Tensor, input_ids: torch.Tensor, tokenizer=None) -> torch.Tensor:
        return logits

    def detect_token_ids(self, token_ids: list[int], vocab_size: int, threshold: float | None = None) -> DetectionStats:
        if len(token_ids) <= self.ngram + 1:
            return DetectionStats(0.0, 0.0, 0, None if threshold is None else False, threshold, {"p_value": 1.0, "fused_score_sum": 0.0, "scoring_method": self.config.get("scoring_method", "v2")})
        values = []
        seen: set[tuple[int, ...]] = set()
        scoring_method = self.config.get("scoring_method", "v2")
        for index in range(self.ngram + 1, len(token_ids)):
            window = token_ids[index - self.ngram:index]
            dedup_key = tuple(window) if scoring_method == "v1" else tuple(window) + (token_ids[index],) if scoring_method == "v2" else None
            if dedup_key is not None and dedup_key in seen:
                continue
            if dedup_key is not None:
                seen.add(dedup_key)
            current = torch.tensor([token_ids[index]])
            a = float(_textseal_uniform(window, current, self.key_a)[0])
            b = float(_textseal_uniform(window, current, self.key_b)[0])
            values.append(
                self.mixing_alpha * (-math.log(max(1 - a, 1e-12)))
                + (1 - self.mixing_alpha) * (-math.log(max(1 - b, 1e-12)))
            )
        if not values:
            return DetectionStats(0.0, 0.0, 0, None if threshold is None else False, threshold, {"p_value": 1.0, "fused_score_sum": 0.0, "scoring_method": scoring_method})
        score = float(np.sum(values))
        base_var = self.mixing_alpha**2 + (1 - self.mixing_alpha)**2
        p_value = float(special.gammaincc(len(values) / base_var, score / base_var))
        log_p = -math.log(max(p_value, 1e-300))
        return DetectionStats(log_p, log_p, len(values), None if threshold is None else log_p >= threshold, threshold, {"p_value": p_value, "fused_score_sum": score, "scoring_method": scoring_method})


class MorphMarkWatermark(BaseWatermark):
    """MorphMark adaptive green-mass reweighting from MarkLLM."""

    name = "MorphMark"

    def __init__(self, gamma: float = 0.5, p_0: float = 0.15, k_exp: float = 1.30, prefix_length: int = 1, hash_key: int = 15485863, base_seed: int = 42, **kwargs):
        super().__init__(gamma=gamma, base_seed=base_seed)
        self.p_0, self.k_exp, self.prefix_length, self.hash_key = float(p_0), float(k_exp), int(prefix_length), int(hash_key)
        self.config = {"method": self.name, "gamma": self.gamma, "p_0": self.p_0, "k_exp": self.k_exp, "prefix_length": self.prefix_length, "hash_key": self.hash_key, "source_revision": "MarkLLM@0a4fe8c642b31b8a8c4993615b1c70a0c300d918"}

    def _green(self, input_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
        context = input_ids[0].tolist()[-self.prefix_length:]
        prf_gen = torch.Generator(device=input_ids.device).manual_seed(self.hash_key)
        prf = torch.randperm(vocab_size, generator=prf_gen, device=input_ids.device)
        time_result = 1
        for token_id in context:
            time_result *= int(token_id)
        previous = int(prf[time_result % vocab_size].item())
        gen = torch.Generator(device=input_ids.device).manual_seed((self.hash_key * previous) % vocab_size)
        size = max(1, int(vocab_size * self.gamma))
        mask = torch.zeros(vocab_size, device=input_ids.device, dtype=torch.bool)
        mask[torch.randperm(vocab_size, generator=gen, device=input_ids.device)[:size]] = True
        return mask

    def apply(self, logits: torch.Tensor, input_ids: torch.Tensor, tokenizer=None) -> torch.Tensor:
        if input_ids.shape[1] < self.prefix_length:
            return logits
        probs = logits.softmax(-1)
        mask = self._green(input_ids, logits.shape[-1]).unsqueeze(0)
        p_green = (probs * mask).sum(-1, keepdim=True)
        r = torch.where(
            p_green < self.p_0,
            torch.zeros_like(p_green),
            torch.exp(torch.tensor(self.k_exp, device=logits.device, dtype=logits.dtype) * p_green) - 1.0,
        )
        beta = r * (1.0 - p_green)
        green_probs = probs * mask
        green_probs = green_probs + (green_probs / green_probs.sum(-1, keepdim=True).clamp_min(1e-12)) * beta
        red_probs = probs * (~mask)
        red_probs = red_probs - (red_probs / red_probs.sum(-1, keepdim=True).clamp_min(1e-12)) * beta
        weighted = torch.nan_to_num(green_probs + red_probs, nan=0.0).clamp_min(0.0)
        weighted = weighted / weighted.sum(-1, keepdim=True).clamp_min(1e-12)
        return torch.log(weighted.clamp_min(1e-12))

    def detect_token_ids(self, token_ids: list[int], vocab_size: int, threshold: float | None = None) -> DetectionStats:
        flags = []
        for index in range(self.prefix_length, len(token_ids)):
            ids = torch.tensor([token_ids[:index]], dtype=torch.long)
            flags.append(float(self._green(ids, vocab_size)[token_ids[index]].item()))
        total = len(flags)
        green_count = float(sum(flags))
        z_score = (green_count - self.gamma * total) / math.sqrt(total * self.gamma * (1 - self.gamma)) if total else 0.0
        return DetectionStats(z_score, z_score, total, None if threshold is None else z_score >= threshold, threshold, {"green_count": green_count, "gamma": self.gamma})
