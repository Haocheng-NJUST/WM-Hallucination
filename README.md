# Watermark Supplement

Minimal code for paired RAG generation, watermark interventions, token-level detection, and empirical detectability calibration.

## Install

```bash
python -m pip install -r requirements.txt
```

Use a Transformers-compatible causal language model. The four added methods are exposed through lightweight adapters that preserve the supplied upstream algorithms and configuration semantics.

## Input schema

Each JSONL row contains `sample_id`, `fact_type`, `domain`, `system_prompt`, `context`, `query`, and `target_facts`. `fact_type` is one of `numerical`, `identifier`, `temporal`, or `qualitative`. `target_facts` is retained for evaluation records only.

## Methods

The factory supports `KGW`, `SWEET`, `DiPmark`, `GumbelSoft`, `GumbelMax`, official `SynthID`, `UnbiasedWatermark`, `TextSeal`, and `MorphMark`. The added adapters track DiPmark (https://github.com/yihwu/DiPmark, revision `34abbeb527243c79bda8043313bb797a731f4ae7`), Unbiased Watermark (https://github.com/xiaoniu-578fa6bff964d005/UnbiasedWatermark, revision `050eee18b06e01c95eb175c7eed725758a9ac451`), TextSeal (https://github.com/facebookresearch/textseal, commit `c60d0d1da2e59f09a698438e218a07ee779b4616`), and MorphMark in MarkLLM (https://github.com/THU-BPM/MarkLLM, commit `0a4fe8c642b31b8a8c4993615b1c70a0c300d918`).

These four methods are lightweight adapters of the supplied upstream revisions. DiPmark and MorphMark retain upstream history/greenlist processing and z-score detection; TextSeal retains random dual-key generation, deduplication, fused score, and p-value calculation; Unbiased retains delta generation and the model-dependent robust-LLR detector.

Gumbel methods apply repetition penalty, temperature scaling, top-k filtering, and top-p filtering before keyed selection. Clean and watermarked generation share the same `GenerationConfig`.

## Paired generation

```bash
python scripts/generate_paired_outputs.py \
  --model_name_or_path <model> \
  --input_jsonl examples/rag_samples.example.jsonl \
  --output_jsonl outputs/paired_generations.jsonl \
  --method KGW
```

Every output row keeps the rendered `generation`, the generation-loop audit field `generated_token_ids`, and `detector_token_ids`. The latter is produced by the detector tokenizer with `add_special_tokens=False` from the final decoded text, and all detectability scores use it. `generated_token_count` counts the generation-loop IDs.

Unbiased Watermark uses the upstream delta-reweighting generator and model-dependent robust-LLR detector. Detection requires the model used to compute the original distribution `p_t` and delta-reweighted distribution `q_t`; missing model inputs raise an explicit error.

When a prompt is available, the Unbiased detector reconstructs model distributions from the same prompt token IDs plus the final-text detector token IDs, and reports scores only for completion tokens.

## FPTI and FPTI+FPAI

```bash
python scripts/run_interventions.py \
  --model_name_or_path <model> \
  --input_jsonl examples/rag_samples.example.jsonl \
  --output_jsonl outputs/fpti_fpai.jsonl \
  --method KGW --intervention fpti_fpai \
  --fpai_calibration_table configs/fpai_calibration.example.json
```

FPTI facts come only from automatic extraction over `context` (regex spans, or an optional spaCy model in the intervention API). FPAI maps character spans to final-prompt token positions with fast-tokenizer offsets; the strict contiguous-subsequence fallback is used only when offsets are unavailable. FPAI requires either a JSON calibration table or a callback `module:function` with signature `calibrate_lambda(prefix_detectability, relative_position, step, metadata)`. The selected calibration type and path are written to `fpai_calibration` metadata.

## Detectability calibration

Clean and watermarked score JSONL files can be calibrated without model inference:

```bash
python scripts/calibrate_detectability.py \
  --clean_jsonl outputs/clean.jsonl \
  --watermarked_jsonl outputs/kgw_strength_sweep.jsonl \
  --target_fpr 0.01 --target_tpr 0.90 \
  --output_json outputs/kgw_tpr90_calibration.json
```

The record contains the empirical clean-score threshold, measured FPR/TPR, selected strength, sample counts, method, and score direction. Detector rows keep `score`, `z_score`, and use `detected: null` until an empirical threshold is supplied.

Each sweep row records explicit `calibration_strength_name` and `calibration_strength_value`. KGW/SWEET use `delta`, DiPmark uses `alpha`, MorphMark uses `k_exp`, and Gumbel methods use their configured scale parameter. TextSeal and Unbiased use a configuration label unless a sweep explicitly supplies a calibration name/value. A compact sweep reader and sequential candidate generator are available as `scripts/sweep_detectability.py`.

TextSeal `score` is `-log(p_value)` from the official Gamma-tail calculation; the raw fused statistic is retained as `fused_score_sum` metadata.

TextSeal, Unbiased Watermark, and MorphMark expose their standard generation and detection paths. Their unsupported FPTI/FPAI combinations are rejected by the intervention CLI.

## Annotation aggregation

```bash
python scripts/aggregate_annotations.py \
  --generations_jsonl outputs/paired_generations.jsonl \
  --annotations_jsonl examples/annotations.example.jsonl \
  --output_jsonl outputs/paired_with_annotations.jsonl
```
