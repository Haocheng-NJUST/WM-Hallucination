"""Paired unwatermarked/watermarked generation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .chat import build_prompt
from .calibration import calibration_strength_fields
from .data_io import RAGSample
from .detectors import detect_generated_text, detector_token_ids
from .sampling import apply_repetition_penalty, sample_next_token
from .watermarks import BaseWatermark, create_watermark


@dataclass
class GenerationConfig:
    max_new_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    do_sample: bool = True


@dataclass(frozen=True)
class GeneratedOutput:
    text: str
    token_ids: list[int]


def load_model_and_tokenizer(model_name_or_path: str, device: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        device_map="auto" if device is None and torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    config: GenerationConfig,
    pair_seed: int,
    watermark: BaseWatermark | None = None,
) -> GeneratedOutput:
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    generated = input_ids.clone()
    if watermark is not None and hasattr(watermark, "reset_history"):
        watermark.reset_history()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(pair_seed))

    with torch.no_grad():
        for _ in range(config.max_new_tokens):
            outputs = model(input_ids=generated, attention_mask=None if attention_mask is None else torch.ones_like(generated))
            logits = outputs.logits[:, -1, :].clone()
            logits = apply_repetition_penalty(logits, generated, config.repetition_penalty)

            selected = None
            if watermark is not None:
                selected = watermark.select_next_token(
                    logits,
                    generated,
                    tokenizer=tokenizer,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    do_sample=config.do_sample,
                    generator=generator,
                )
                if selected is None:
                    logits = watermark.apply(logits, generated, tokenizer=tokenizer)

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
    token_ids = [int(value) for value in new_ids.tolist()]
    return GeneratedOutput(tokenizer.decode(new_ids, skip_special_tokens=True).strip(), token_ids)


def generate_text_with_model_generate(
    model,
    tokenizer,
    prompt: str,
    config: GenerationConfig,
    pair_seed: int,
    watermarking_config=None,
) -> GeneratedOutput:
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    torch.manual_seed(int(pair_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(pair_seed))

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "repetition_penalty": config.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if config.temperature and config.temperature > 0:
        generate_kwargs["temperature"] = config.temperature
    if watermarking_config is not None:
        generate_kwargs["watermarking_config"] = watermarking_config

    with torch.no_grad():
        output_ids = model.generate(**generate_kwargs)

    new_ids = output_ids[0, input_ids.shape[1] :]
    token_ids = [int(value) for value in new_ids.tolist()]
    return GeneratedOutput(tokenizer.decode(new_ids, skip_special_tokens=True).strip(), token_ids)


def generate_paired_rows(
    samples: Iterable[RAGSample],
    model,
    tokenizer,
    method: str,
    config: GenerationConfig,
    watermark_kwargs: dict,
    seed: int = 42,
    num_iterations: int = 1,
    calibration_strength_name: str | None = None,
    calibration_strength_value: str | float | int | None = None,
) -> list[dict]:
    rows = []
    watermark = create_watermark(method, **watermark_kwargs)
    watermark_config = {"method": method, **watermark_kwargs, **getattr(watermark, "config", {})}
    inferred_name, inferred_value = calibration_strength_fields(method, watermark_config)
    calibration_strength_name = calibration_strength_name or inferred_name
    calibration_strength_value = calibration_strength_value if calibration_strength_value is not None else inferred_value

    for index, sample in enumerate(samples):
        prompt = build_prompt(tokenizer, sample)
        for iteration in range(int(num_iterations)):
            pair_seed = int(seed) + index * 1000 + iteration
            started = time.time()

            if method == "SynthID":
                baseline_text = generate_text_with_model_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    config=config,
                    pair_seed=pair_seed,
                    watermarking_config=None,
                )
                watermarked_text = generate_text_with_model_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    config=config,
                    pair_seed=pair_seed,
                    watermarking_config=watermark.build_config(),
                )
            else:
                baseline_text = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    config=config,
                    pair_seed=pair_seed,
                    watermark=None,
                )
                watermarked_text = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    config=config,
                    pair_seed=pair_seed,
                    watermark=watermark,
                )

            common = {
                "sample_id": sample.sample_id,
                "fact_type": sample.fact_type,
                "domain": sample.domain,
                "context": sample.context,
                "query": sample.query,
                "prompt": prompt,
                "target_facts": sample.target_facts,
                "watermark_method": method,
                "watermark_config": watermark_config,
                "calibration_strength_name": calibration_strength_name,
                "calibration_strength_value": calibration_strength_value,
                "iteration": iteration,
                "pair_seed": pair_seed,
                "elapsed_seconds_pair": round(time.time() - started, 3),
            }
            rows.append(
                {
                    **common,
                    "variant": "unwatermarked",
                    "generation": baseline_text.text,
                    "generated_token_ids": baseline_text.token_ids,
                    "generated_token_count": len(baseline_text.token_ids),
                    "detector_token_ids": detector_token_ids(baseline_text.text, tokenizer),
                    "detectability": detect_generated_text(baseline_text.text, tokenizer, watermark, model=model, generation_config=config, prompt=prompt),
                }
            )
            rows.append(
                {
                    **common,
                    "variant": "watermarked",
                    "generation": watermarked_text.text,
                    "generated_token_ids": watermarked_text.token_ids,
                    "generated_token_count": len(watermarked_text.token_ids),
                    "detector_token_ids": detector_token_ids(watermarked_text.text, tokenizer),
                    "detectability": detect_generated_text(watermarked_text.text, tokenizer, watermark, model=model, generation_config=config, prompt=prompt),
                }
            )

    return rows
