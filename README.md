# Invisible Ink, Visible Lies

This is the official code for the NeurIPS 2026 paper **“Invisible Ink, Visible Lies: How Production Watermarking Causes LLMs to Hallucinate.”**

This repository provides the implementation for studying factual errors induced by text watermarking in retrieval-augmented generation. It supports paired clean and watermarked generation, empirical detectability calibration at matched TPR@1%FPR, and the proposed Fact-Preserving Token Intervention (FPTI) and Fact-Preserving Attention Intervention (FPAI). The implemented watermarking methods include KGW, SWEET, DiPmark, GumbelSoft, Gumbel-Max, SynthID, Unbiased Watermark, TextSeal, and MorphMark.

## Installation

```bash
python -m pip install -r requirements.txt
```

The code uses a Transformers-compatible causal language model.

## Data Format

The input is a JSONL file in which each row contains:

```json
{
  "sample_id": "sample-001",
  "fact_type": "numerical",
  "domain": "example-domain",
  "system_prompt": "Answer using the provided context.",
  "context": "The retrieved context containing the supporting evidence.",
  "query": "A question about the context.",
  "target_facts": ["The reference facts used for evaluation."]
}
```

`fact_type` must be one of `numerical`, `identifier`, `temporal`, or `qualitative`. `target_facts` is used for evaluation and is not used by FPTI or FPAI.

Example inputs are provided in `examples/`.

## Paired Generation

```bash
python scripts/generate_paired_outputs.py \
  --model_name_or_path <model> \
  --input_jsonl examples/rag_samples.example.jsonl \
  --output_jsonl outputs/paired_generations.jsonl \
  --method KGW
```

Clean and watermarked generations use the same decoding configuration and paired random seeds.

## FPTI and FPAI

Run FPTI:

```bash
python scripts/run_interventions.py \
  --model_name_or_path <model> \
  --input_jsonl examples/rag_samples.example.jsonl \
  --output_jsonl outputs/fpti.jsonl \
  --method KGW \
  --intervention fpti
```

Run FPTI together with FPAI:

```bash
python scripts/run_interventions.py \
  --model_name_or_path <model> \
  --input_jsonl examples/rag_samples.example.jsonl \
  --output_jsonl outputs/fpti_fpai.jsonl \
  --method KGW \
  --intervention fpti_fpai \
  --fpai_calibration_table configs/fpai_calibration.example.json
```

FPAI accepts a calibration table or calibration callback.

## Detectability Calibration

```bash
python scripts/calibrate_detectability.py \
  --clean_jsonl outputs/clean.jsonl \
  --watermarked_jsonl outputs/watermarked_strength_sweep.jsonl \
  --target_fpr 0.01 \
  --target_tpr 0.90 \
  --output_json outputs/calibration.json
```

The calibration estimates an empirical detection threshold at the specified FPR and selects the watermark configuration closest to the target TPR.
