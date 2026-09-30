"""CLI for paired unwatermarked and watermarked generation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paired unwatermarked and watermarked RAG outputs with detectability statistics."
    )
    parser.add_argument("--model_name_or_path", required=True, help="Model name or path understood by transformers.")
    parser.add_argument("--input_jsonl", required=True, help="Input RAG samples JSONL.")
    parser.add_argument("--output_jsonl", required=True, help="Output paired generations JSONL.")
    parser.add_argument(
        "--method",
        default="KGW",
        choices=["KGW", "SWEET", "DiPmark", "GumbelSoft", "GumbelMax", "SynthIDStyle", "SynthID", "UnbiasedWatermark", "TextSeal", "MorphMark"],
        help="Watermark method.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed. Each pair uses seed + sample_index * 1000 + iteration.",
    )
    parser.add_argument("--num_iterations", type=int, default=1, help="Number of paired generations per sample.")
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding for non-GumbelSoft sampling.")
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--strength", type=float, default=2.0, help="Watermark strength where supported.")
    parser.add_argument("--base_seed", type=int, default=42, help="Watermark key seed.")
    parser.add_argument("--sweet_entropy_threshold", type=float, default=0.695)
    parser.add_argument("--sweet_use_normalized_entropy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gsoft_ngram", type=int, default=3)
    parser.add_argument("--gsoft_tau", type=float, default=1.0)
    parser.add_argument("--gmax_ngram", type=int, default=3)
    parser.add_argument("--gmax_scale", type=float, default=1.0)
    parser.add_argument("--synthid_ngram", type=int, default=3)
    parser.add_argument("--synthid_strength", type=float, default=1.0)
    parser.add_argument("--synthid_key", default="synthid-style-key")
    parser.add_argument("--synthid_ngram_len", type=int, default=5)
    parser.add_argument("--synthid_key_num", type=int, default=9)
    parser.add_argument("--synthid_sampling_table_size", type=int, default=65536)
    parser.add_argument("--synthid_sampling_table_seed", type=int, default=0)
    parser.add_argument("--synthid_context_history_size", type=int, default=1024)
    parser.add_argument("--dip_alpha", type=float, default=0.45)
    parser.add_argument("--dip_prefix_length", type=int, default=5)
    parser.add_argument("--unbiased_prefix_length", type=int, default=5)
    parser.add_argument("--textseal_ngram", type=int, default=2)
    parser.add_argument("--textseal_mixing_alpha", type=float, default=0.5)
    parser.add_argument("--textseal_key", type=int, default=42)
    parser.add_argument("--textseal_scoring_method", choices=["v1", "v2", "none"], default="v2")
    parser.add_argument("--morphmark_gamma", type=float, default=0.5)
    parser.add_argument("--morphmark_p0", type=float, default=0.15)
    parser.add_argument("--morphmark_k_exp", type=float, default=1.30)
    parser.add_argument("--morphmark_prefix_length", type=int, default=1)
    parser.add_argument("--morphmark_hash_key", type=int, default=15485863)
    parser.add_argument("--calibration_strength_name", default=None, help="Explicit calibration grouping name.")
    parser.add_argument("--calibration_strength_value", default=None, help="Explicit calibration grouping value.")
    parser.add_argument("--device", default=None, help="Optional explicit device, for example cpu or cuda:0.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from supplement.data_io import load_rag_samples, write_jsonl
    from supplement.generation import GenerationConfig, generate_paired_rows, load_model_and_tokenizer

    samples = load_rag_samples(args.input_jsonl)
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, device=args.device)
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        do_sample=not args.greedy,
    )
    watermark_kwargs = {
        "gamma": args.gamma,
        "delta": args.delta,
        "strength": args.strength,
        "base_seed": args.base_seed,
        "sweet_entropy_threshold": args.sweet_entropy_threshold,
        "sweet_use_normalized_entropy": args.sweet_use_normalized_entropy,
        "gsoft_ngram": args.gsoft_ngram,
        "gsoft_tau": args.gsoft_tau,
        "gmax_ngram": args.gmax_ngram,
        "gmax_scale": args.gmax_scale,
        "synthid_ngram": args.synthid_ngram,
        "synthid_strength": args.synthid_strength,
        "synthid_key": args.synthid_key,
        "synthid_ngram_len": args.synthid_ngram_len,
        "synthid_key_num": args.synthid_key_num,
        "synthid_sampling_table_size": args.synthid_sampling_table_size,
        "synthid_sampling_table_seed": args.synthid_sampling_table_seed,
        "synthid_context_history_size": args.synthid_context_history_size,
        "dip_alpha": args.dip_alpha,
        "dip_prefix_length": args.dip_prefix_length,
        "unbiased_prefix_length": args.unbiased_prefix_length,
        "textseal_ngram": args.textseal_ngram,
        "textseal_mixing_alpha": args.textseal_mixing_alpha,
        "textseal_key": args.textseal_key,
        "textseal_scoring_method": args.textseal_scoring_method,
        "morphmark_gamma": args.morphmark_gamma,
        "morphmark_p0": args.morphmark_p0,
        "morphmark_k_exp": args.morphmark_k_exp,
        "morphmark_prefix_length": args.morphmark_prefix_length,
        "morphmark_hash_key": args.morphmark_hash_key,
    }
    rows = generate_paired_rows(
        samples=samples,
        model=model,
        tokenizer=tokenizer,
        method=args.method,
        config=generation_config,
        watermark_kwargs=watermark_kwargs,
        seed=args.seed,
        num_iterations=args.num_iterations,
        calibration_strength_name=args.calibration_strength_name,
        calibration_strength_value=args.calibration_strength_value,
    )
    write_jsonl(args.output_jsonl, rows)
    print(f"Wrote {len(rows)} paired generation rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
