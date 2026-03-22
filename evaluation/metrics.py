"""
Evaluation metrics computation for Text-to-SQL.
Computes EX (execution accuracy), EM (exact match), and stratified results.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Support both package import and direct execution
try:
    from evaluation.sql_executor import evaluate_single, find_database_path
except ImportError:
    from sql_executor import evaluate_single, find_database_path


def compute_metrics(
    predictions: list[dict],
    databases_dir: str | Path,
) -> dict:
    """
    Compute all metrics for a list of predictions.

    Each prediction dict must have:
        - pred_sql: str
        - gold_sql: str
        - db_id: str
        - difficulty: str (optional)

    Returns comprehensive metrics dict.
    """
    db_dir = Path(databases_dir)
    results = []
    errors = []

    # Per-difficulty tracking
    difficulty_buckets = defaultdict(lambda: {"total": 0, "ex_correct": 0, "em_correct": 0})
    total_ex = 0
    total_em = 0
    total_pred_errors = 0

    for i, pred in enumerate(predictions):
        db_path = find_database_path(pred["db_id"], db_dir)

        if db_path is None:
            # Can't evaluate without DB — log and skip
            result = {
                "index": i,
                "db_id": pred["db_id"],
                "execution_match": False,
                "exact_match": False,
                "error": f"Database not found for {pred['db_id']}",
                "difficulty": pred.get("difficulty", "unknown"),
            }
            errors.append(result)
            results.append(result)
            continue

        eval_result = evaluate_single(pred["pred_sql"], pred["gold_sql"], db_path)
        eval_result["index"] = i
        eval_result["db_id"] = pred["db_id"]
        eval_result["difficulty"] = pred.get("difficulty", "unknown")
        eval_result["question"] = pred.get("question", "")
        results.append(eval_result)

        # Aggregate
        if eval_result["execution_match"]:
            total_ex += 1
        if eval_result["exact_match"]:
            total_em += 1
        if eval_result.get("pred_error"):
            total_pred_errors += 1

        # Per-difficulty
        diff = eval_result["difficulty"]
        difficulty_buckets[diff]["total"] += 1
        if eval_result["execution_match"]:
            difficulty_buckets[diff]["ex_correct"] += 1
        if eval_result["exact_match"]:
            difficulty_buckets[diff]["em_correct"] += 1

    total = len(predictions)

    # Compute rates
    metrics = {
        "total_samples": total,
        "execution_accuracy": round(total_ex / total * 100, 2) if total > 0 else 0,
        "exact_match_accuracy": round(total_em / total * 100, 2) if total > 0 else 0,
        "execution_correct": total_ex,
        "exact_match_correct": total_em,
        "prediction_errors": total_pred_errors,
        "error_rate": round(total_pred_errors / total * 100, 2) if total > 0 else 0,
        "db_not_found": len(errors),
    }

    # Per-difficulty metrics
    metrics["by_difficulty"] = {}
    for diff, counts in sorted(difficulty_buckets.items()):
        t = counts["total"]
        metrics["by_difficulty"][diff] = {
            "total": t,
            "execution_accuracy": round(counts["ex_correct"] / t * 100, 2) if t > 0 else 0,
            "exact_match_accuracy": round(counts["em_correct"] / t * 100, 2) if t > 0 else 0,
        }

    return {
        "summary": metrics,
        "per_sample": results,
    }


def compute_inference_metrics(
    inference_times: list[float],
    token_counts: list[int],
) -> dict:
    """Compute inference time and token statistics."""
    import statistics

    if not inference_times:
        return {}

    return {
        "inference_time_ms": {
            "mean": round(statistics.mean(inference_times), 2),
            "median": round(statistics.median(inference_times), 2),
            "std": round(statistics.stdev(inference_times), 2) if len(inference_times) > 1 else 0,
            "min": round(min(inference_times), 2),
            "max": round(max(inference_times), 2),
            "p95": round(sorted(inference_times)[int(len(inference_times) * 0.95)], 2),
        },
        "token_counts": {
            "mean": round(statistics.mean(token_counts), 1) if token_counts else 0,
            "max": max(token_counts) if token_counts else 0,
            "total": sum(token_counts),
        },
    }


def format_results_table(all_results: dict[str, dict]) -> str:
    """
    Format results across all conditions into a readable table.
    all_results: {condition_name: metrics_summary}
    """
    lines = []
    header = f"{'Condition':<35} {'EX %':>7} {'EM %':>7} {'Errors':>7} {'Inf ms':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for condition, metrics in sorted(all_results.items()):
        summary = metrics.get("summary", metrics)
        inf = metrics.get("inference", {}).get("inference_time_ms", {})
        lines.append(
            f"{condition:<35} "
            f"{summary.get('execution_accuracy', 0):>6.1f}% "
            f"{summary.get('exact_match_accuracy', 0):>6.1f}% "
            f"{summary.get('prediction_errors', 0):>7} "
            f"{inf.get('mean', 0):>7.1f}"
        )

    return "\n".join(lines)


def save_results(results: dict, output_path: str | Path):
    """Save results to JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {path}")
