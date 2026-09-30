"""Intervention generation for watermark, FPTI, and FPTI+FPAI settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .chat import build_prompt
from .calibration import calibration_strength_fields
from .data_io import RAGSample
from .detectors import detect_generated_text, detector_token_ids
from .fact_extraction import FactSpan, extract_context_facts
from .fpai import load_fpai_calibrator
from .generation import GeneratedOutput, GenerationConfig, generate_text_with_model_generate
from .sampling import apply_repetition_penalty, sample_next_token
from .watermarks import BaseWatermark, create_watermark


SUPPORTED_INTERVENTIONS = ("watermark", "fpti", "fpti_fpai")


@dataclass(frozen=True)
class FactTokenInfo:
    token_ids: set[int]
    context_positions: list[int]
    spans: list[FactSpan]


def _find_subsequence(haystack: list[int], needle: list[int]) -> int:
    if not needle or len(needle) > len(haystack):
        return -1
    first = needle[0]
    last = len(haystack) - len(needle)
    for index in range(last + 1):
        if haystack[index] == first and haystack[index : index + len(needle)] == needle:
            return index
    return -1


def fact_token_info(sample: RAGSample, tokenizer, prompt: str, input_ids: torch.Tensor, spacy_model: str | None = None) -> FactTokenInfo:
    spans = extract_context_facts(sample.context, spacy_model=spacy_model)
    prompt_context_start = prompt.find(sample.context)
    if prompt_context_start < 0:
        raise ValueError("Context could not be located in the final prompt used for generation")
    try:
        encoded = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
        offsets = encoded["offset_mapping"]
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
            offsets = offsets[0]
        if len(offsets) != input_ids.shape[1]:
            raise ValueError("offset mapping does not align with the final prompt token IDs")
    except Exception:
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        context_ids = tokenizer(sample.context, add_special_tokens=False).input_ids
        context_start = _find_subsequence(prompt_ids, context_ids)
        if context_start < 0:
            raise ValueError("Tokenizer does not provide offsets and context token subsequence was not found")
        positions = []
        row = input_ids[0].tolist()
        for span in spans:
            context_span = sample.context[span.start:span.end]
            fact_ids = tokenizer(context_span, add_special_tokens=False).input_ids
            start = _find_subsequence(context_ids, fact_ids)
            if start < 0:
                raise ValueError(f"Fact span {span.text!r} was not found as a contiguous token span")
            positions.extend(range(context_start + start, context_start + start + len(fact_ids)))
    else:
        positions = []
        for index, (start, end) in enumerate(offsets):
            if end <= start:
                continue
            absolute_start, absolute_end = start, end
            if any(absolute_start < prompt_context_start + span.end and prompt_context_start + span.start < absolute_end for span in spans):
                positions.append(index)
    prompt_token_count = len(offsets) if "offsets" in locals() else len(tokenizer(prompt, add_special_tokens=False).input_ids)
    offset = input_ids.shape[1] - prompt_token_count
    positions = sorted({position + offset for position in positions if 0 <= position + offset < input_ids.shape[1]})
    row = input_ids[0].tolist()
    return FactTokenInfo(token_ids={row[position] for position in positions}, context_positions=positions, spans=spans)


def apply_fpti(clean_logits: torch.Tensor, input_ids: torch.Tensor, watermark: BaseWatermark, fact_ids: set[int]) -> torch.Tensor:
    watermarked_logits = watermark.apply(clean_logits.clone(), input_ids)
    if fact_ids:
        valid_ids = [token_id for token_id in fact_ids if 0 <= token_id < clean_logits.shape[-1]]
        if valid_ids:
            watermarked_logits[:, valid_ids] = clean_logits[:, valid_ids]
    return watermarked_logits


def select_with_method_specific_fpti(
    clean_logits: torch.Tensor,
    input_ids: torch.Tensor,
    watermark: BaseWatermark,
    fact_ids: set[int],
    tokenizer,
    fact_positions: list[int] | None = None,
    **sampling_kwargs,
):
    if hasattr(watermark, "select_next_token_with_fpti"):
        selected = watermark.select_next_token_with_fpti(
            clean_logits,
            input_ids,
            factual_ids=fact_ids,
            tokenizer=tokenizer,
            **sampling_kwargs,
        )
        return selected, None
    return None, apply_fpti(clean_logits, input_ids, watermark, fact_ids)


def apply_official_synthid_fpti(
    clean_logits: torch.Tensor,
    input_ids: torch.Tensor,
    watermark: BaseWatermark,
    fact_ids: set[int],
) -> torch.Tensor:
    processor = watermark.build_logits_processor(clean_logits.device)
    synthid_logits = processor(input_ids, clean_logits.clone())
    if fact_ids:
        valid_ids = [token_id for token_id in fact_ids if 0 <= token_id < clean_logits.shape[-1]]
        if valid_ids:
            synthid_logits[:, valid_ids] = clean_logits[:, valid_ids]
    return synthid_logits


def build_fpai_attention_mask(length: int, fact_positions: list[int], device: torch.device, bias: float) -> torch.Tensor:
    neg_inf = -1e9
    mask = torch.zeros((1, 1, length, length), device=device, dtype=torch.float32)
    causal = torch.triu(torch.ones((length, length), device=device, dtype=torch.bool), diagonal=1)
    mask[0, 0, causal] = neg_inf
    for pos in fact_positions:
        if 0 <= pos < length:
            mask[0, 0, length - 1, pos] += float(bias)
    return mask


def model_forward(model, input_ids: torch.Tensor, intervention: str, fact_positions: list[int], fpai_lambda: float):
    if intervention != "fpti_fpai":
        return model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

    attention_mask = build_fpai_attention_mask(
        length=input_ids.shape[1],
        fact_positions=fact_positions,
        device=input_ids.device,
        bias=fpai_lambda,
    )
    try:
        return model(input_ids=input_ids, attention_mask=attention_mask)
    except Exception as exc:
        raise RuntimeError(
            "FPAI requires model support for 4D additive attention masks. "
            "This model or transformers version rejected the mask; rerun with --intervention fpti."
        ) from exc


def generate_intervention_text(
    model,
    tokenizer,
    sample: RAGSample,
    intervention: str,
    watermark: BaseWatermark,
    config: GenerationConfig,
    pair_seed: int,
    fpai_calibrator=None,
    spacy_model: str | None = None,
    allow_experimental_synthid_intervention: bool = False,
) -> GeneratedOutput:
    if intervention not in SUPPORTED_INTERVENTIONS:
        raise ValueError(f"Unknown intervention: {intervention}")
    device = next(model.parameters()).device
    prompt = build_prompt(tokenizer, sample)
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    generated = input_ids.clone()
    if hasattr(watermark, "reset_history"):
        watermark.reset_history()
    info = fact_token_info(sample, tokenizer, prompt, input_ids, spacy_model=spacy_model)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(pair_seed))

    with torch.no_grad():
        for _ in range(config.max_new_tokens):
            if intervention == "fpti_fpai":
                completion_prefix = generated[0, input_ids.shape[1]:].tolist()
                if not completion_prefix:
                    prefix_detectability = 0.0
                else:
                    prefix_text = tokenizer.decode(completion_prefix, skip_special_tokens=True)
                    prefix_ids = detector_token_ids(prefix_text, tokenizer)
                    prefix_vocab_size = getattr(tokenizer, "vocab_size", None)
                    if prefix_vocab_size is None:
                        prefix_vocab_size = len(tokenizer)
                    prefix_stats = watermark.detect_token_ids(prefix_ids, int(prefix_vocab_size))
                    if prefix_stats.score is None or not torch.isfinite(torch.tensor(prefix_stats.score)):
                        raise ValueError(f"FPAI requires a finite prefix detector score for {watermark.name}")
                    prefix_detectability = float(prefix_stats.score)
            else:
                prefix_detectability = None
            relative_position = min(1.0, max(0.0, (generated.shape[1] - input_ids.shape[1]) / max(config.max_new_tokens, 1)))
            fpai_lambda = fpai_calibrator(prefix_detectability, relative_position, generated.shape[1] - input_ids.shape[1], {"sample_id": sample.sample_id}) if intervention == "fpti_fpai" else 0.0
            outputs = model_forward(model, generated, intervention, info.context_positions, fpai_lambda)
            clean_logits = outputs.logits[:, -1, :].clone()
            clean_logits = apply_repetition_penalty(clean_logits, generated, config.repetition_penalty)

            selected = None
            if intervention == "watermark":
                selected = watermark.select_next_token(
                    clean_logits,
                    generated,
                    tokenizer=tokenizer,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    do_sample=config.do_sample,
                    generator=generator,
                )
                logits = clean_logits if selected is not None else watermark.apply(clean_logits, generated, tokenizer=tokenizer)
            elif watermark.name == "SynthID":
                if not allow_experimental_synthid_intervention:
                    raise ValueError(
                        "Official SynthID FPTI/FPAI is experimental and disabled by default. "
                        "Pass --allow_experimental_synthid_intervention to attempt it, or use SynthIDStyle "
                        "for approximate score-level intervention."
                    )
                logits = apply_official_synthid_fpti(clean_logits, generated, watermark, info.token_ids)
            else:
                selected, logits = select_with_method_specific_fpti(
                    clean_logits,
                    generated,
                    watermark,
                    info.token_ids,
                    tokenizer,
                    fact_positions=info.context_positions,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    do_sample=config.do_sample,
                    generator=generator,
                )

            if selected is None:
                selected = sample_next_token(
                    logits,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    do_sample=config.do_sample,
                    generator=generator,
                )

            generated = torch.cat([generated, selected], dim=-1)
            if selected.item() == tokenizer.eos_token_id:
                break

    new_ids = generated[0, input_ids.shape[1] :]
    return GeneratedOutput(tokenizer.decode(new_ids, skip_special_tokens=True).strip(), [int(value) for value in new_ids.tolist()])


def generate_intervention_rows(
    samples: Iterable[RAGSample],
    model,
    tokenizer,
    intervention: str,
    method: str,
    config: GenerationConfig,
    watermark_kwargs: dict,
    seed: int = 42,
    num_iterations: int = 1,
    fpai_calibration_table: str | None = None,
    fpai_calibrator: str | None = None,
    spacy_model: str | None = None,
    allow_experimental_synthid_intervention: bool = False,
) -> list[dict]:
    if intervention not in SUPPORTED_INTERVENTIONS:
        raise ValueError(f"Unknown intervention: {intervention}")
    if method in {"UnbiasedWatermark", "TextSeal", "MorphMark"} and intervention != "watermark":
        raise ValueError(f"{method} currently supports standard watermark generation/detection only; {intervention} is not an implemented method-specific intervention")
    if method == "SynthID" and intervention in {"fpti", "fpti_fpai"} and not allow_experimental_synthid_intervention:
        raise ValueError(
            "Official SynthID FPTI/FPAI is experimental and disabled by default. "
            "Official SynthID FPTI/FPAI requires access to the SynthID logits processor. "
            "Pass --allow_experimental_synthid_intervention to attempt it, use SynthIDStyle "
            "for approximate score-level intervention, or use standard SynthID for official generation."
        )

    calibrator = None
    calibration_metadata = None
    if intervention == "fpti_fpai":
        calibrator, calibration_metadata = load_fpai_calibrator(fpai_calibration_table, fpai_calibrator)
    rows = []
    watermark = create_watermark(method, **watermark_kwargs)
    watermark_config = {"method": method, **watermark_kwargs, **getattr(watermark, "config", {})}
    calibration_strength_name, calibration_strength_value = calibration_strength_fields(method, watermark_config)

    for sample_index, sample in enumerate(samples):
        for iteration in range(int(num_iterations)):
            pair_seed = int(seed) + sample_index * 1000 + iteration
            if method == "SynthID" and intervention == "watermark":
                prompt = build_prompt(tokenizer, sample)
                text = generate_text_with_model_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    config=config,
                    pair_seed=pair_seed,
                    watermarking_config=watermark.build_config(),
                )
            else:
                text = generate_intervention_text(
                    model=model,
                    tokenizer=tokenizer,
                    sample=sample,
                    intervention=intervention,
                    watermark=watermark,
                    config=config,
                    pair_seed=pair_seed,
                    fpai_calibrator=calibrator,
                    spacy_model=spacy_model,
                    allow_experimental_synthid_intervention=allow_experimental_synthid_intervention,
                )
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "fact_type": sample.fact_type,
                    "domain": sample.domain,
                    "context": sample.context,
                    "query": sample.query,
                    "target_facts": sample.target_facts,
                    "intervention": intervention,
                    "watermark_method": method,
                    "iteration": iteration,
                    "pair_seed": pair_seed,
                    "generation": text.text,
                    "generated_token_ids": text.token_ids,
                    "generated_token_count": len(text.token_ids),
                    "detector_token_ids": detector_token_ids(text.text, tokenizer),
                    "detectability": detect_generated_text(text.text, tokenizer, watermark, model=model, generation_config=config, prompt=build_prompt(tokenizer, sample)),
                    "watermark_config": watermark_config,
                    "calibration_strength_name": calibration_strength_name,
                    "calibration_strength_value": calibration_strength_value,
                    "fpai_calibration": calibration_metadata,
                }
            )

    return rows
