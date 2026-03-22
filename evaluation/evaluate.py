"""
Main evaluation driver.
Runs inference with all prompt strategies across all tracks and computes metrics.

Usage: python evaluation/evaluate.py --results-dir ./results --data-dir ./data
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.preprocess import load_and_preprocess_spider
from data.prompt_templates import apply_prompt_template, get_all_strategies
from evaluation.metrics import compute_metrics, compute_inference_metrics, format_results_table, save_results


def load_causal_model(checkpoint_dir: str | Path, quantize: bool = True):
    """Load a fine-tuned causal model for inference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    checkpoint_dir = Path(checkpoint_dir)
    adapter_dir = checkpoint_dir / "final_adapter"
    if not adapter_dir.exists():
        adapter_dir = checkpoint_dir

    # Load metrics to find base model name
    metrics_files = list(checkpoint_dir.parent.glob("*_metrics.json"))
    base_model_name = None
    for mf in metrics_files:
        with open(mf) as f:
            data = json.load(f)
        base_model_name = data.get("config", {}).get("model", {}).get("name")
        if base_model_name:
            break

    if not base_model_name:
        raise ValueError(f"Cannot determine base model from {checkpoint_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    return model, tokenizer


def load_t5_model(checkpoint_dir: str | Path):
    """Load a fine-tuned T5 model for inference."""
    from transformers import T5ForConditionalGeneration, AutoTokenizer

    checkpoint_dir = Path(checkpoint_dir)
    model_dir = checkpoint_dir / "final_model"
    if not model_dir.exists():
        model_dir = checkpoint_dir / "finetune" / "final"

    model = T5ForConditionalGeneration.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    return model, tokenizer


def generate_sql_causal(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
) -> tuple[str, float, int]:
    """
    Generate SQL from a causal model.
    Returns: (generated_sql, inference_time_ms, input_token_count)
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
    input_ids = inputs["input_ids"].to(model.device)
    input_token_count = input_ids.shape[1]

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Decode only the generated portion
    generated_ids = outputs[0][input_ids.shape[1]:]
    generated_sql = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Clean up — extract just the SQL
    if "SELECT" in generated_sql.upper():
        # Find the SQL query
        lines = generated_sql.split("\n")
        sql_lines = []
        capturing = False
        for line in lines:
            if "SELECT" in line.upper() or capturing:
                capturing = True
                sql_lines.append(line)
                if ";" in line:
                    break
        if sql_lines:
            generated_sql = " ".join(sql_lines).strip().rstrip(";")

    return generated_sql, elapsed_ms, input_token_count


def generate_sql_t5(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 256,
) -> tuple[str, float, int]:
    """
    Generate SQL from a T5 model.
    Returns: (generated_sql, inference_time_ms, input_token_count)
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(model.device)
    input_token_count = input_ids.shape[1]

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=max_length)
    elapsed_ms = (time.perf_counter() - start) * 1000

    generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return generated_sql, elapsed_ms, input_token_count


def evaluate_condition(
    model,
    tokenizer,
    eval_data: list[dict],
    track: str,
    strategy: str,
    databases_dir: Path,
    generate_fn,
) -> dict:
    """Evaluate a single experimental condition."""
    condition_name = f"{track}_{strategy}"
    print(f"\n  Evaluating: {condition_name} ({len(eval_data)} samples)...")

    predictions = []
    inference_times = []
    token_counts = []

    for sample in tqdm(eval_data, desc=condition_name, leave=False):
        prompt = apply_prompt_template(
            schema=sample["schema"],
            question=sample["question"],
            track=track,
            strategy=strategy,
        )

        try:
            pred_sql, inf_time, n_tokens = generate_fn(model, tokenizer, prompt)
        except Exception as e:
            pred_sql = f"ERROR: {e}"
            inf_time = 0
            n_tokens = 0

        predictions.append({
            "pred_sql": pred_sql,
            "gold_sql": sample["query"],
            "db_id": sample["db_id"],
            "difficulty": sample.get("difficulty", "unknown"),
            "question": sample["question"],
        })
        inference_times.append(inf_time)
        token_counts.append(n_tokens)

    # Compute metrics
    eval_metrics = compute_metrics(predictions, databases_dir)
    inf_metrics = compute_inference_metrics(inference_times, token_counts)

    result = {
        "condition": condition_name,
        "track": track,
        "strategy": strategy,
        "summary": eval_metrics["summary"],
        "inference": inf_metrics,
        "per_sample": eval_metrics["per_sample"],
    }

    return result


def run_full_evaluation(
    results_dir: Path,
    data_dir: Path,
    databases_dir: Path,
    tracks_to_eval: list[str] = None,
):
    """Run evaluation for all trained models across all strategies."""
    if tracks_to_eval is None:
        tracks_to_eval = ["track_a", "track_b", "track_c"]

    eval_data = load_and_preprocess_spider(data_dir, split="validation")
    print(f"  Loaded {len(eval_data)} evaluation samples")

    all_results = {}
    strategies = get_all_strategies()

    for track in tracks_to_eval:
        # Determine checkpoint dir and model type
        if track == "track_c":
            ckpt_dir = Path(f"./checkpoints/{track}_t5_scratch")
            if not ckpt_dir.exists():
                print(f"  [SKIP] No checkpoint for {track}")
                continue
            model, tokenizer = load_t5_model(ckpt_dir)
            gen_fn = generate_sql_t5
        else:
            suffix = "qwen3_no_think" if track == "track_a" else "qwen3_think"
            ckpt_dir = Path(f"./checkpoints/{track}_{suffix}")
            if not ckpt_dir.exists():
                print(f"  [SKIP] No checkpoint for {track}")
                continue
            model, tokenizer = load_causal_model(ckpt_dir)
            gen_fn = generate_sql_causal

        for strategy in strategies:
            condition_name = f"{track}_{strategy}"
            result = evaluate_condition(
                model, tokenizer, eval_data,
                track, strategy, databases_dir, gen_fn,
            )
            all_results[condition_name] = result

            # Save per-condition results
            save_results(result, results_dir / f"{condition_name}_results.json")

        # Cleanup model between tracks
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save combined results
    combined = {k: {
        "summary": v["summary"],
        "inference": v.get("inference", {}),
    } for k, v in all_results.items()}

    save_results(combined, results_dir / "all_results_summary.json")

    # Print table
    print(f"\n{'='*60}")
    print("RESULTS TABLE")
    print(f"{'='*60}")
    table_data = {k: v for k, v in all_results.items()}
    print(format_results_table(table_data))

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate all models and strategies")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--databases-dir", type=str, default="./data/databases")
    parser.add_argument("--tracks", nargs="+", default=None, help="Tracks to evaluate")
    args = parser.parse_args()

    run_full_evaluation(
        results_dir=Path(args.results_dir),
        data_dir=Path(args.data_dir),
        databases_dir=Path(args.databases_dir),
        tracks_to_eval=args.tracks,
    )


if __name__ == "__main__":
    main()
