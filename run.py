"""
Main orchestrator script — run the full pipeline step by step.

Usage:
  python run.py download        # Download datasets
  python run.py train-a         # Train Track A (Qwen3 /no_think)
  python run.py train-b         # Train Track B (Qwen3 /think)
  python run.py train-c         # Train Track C (T5 from scratch)
  python run.py evaluate        # Evaluate all trained models
  python run.py visualize       # Generate presentation charts
  python run.py all             # Run everything end-to-end
  python run.py status          # Show training status for all tracks
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def cmd_download(args):
    """Download all datasets."""
    from data.download_data import download_spider, download_wikisql, download_spider_databases, verify_data

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Text-to-SQL Dataset Downloader")
    print("=" * 60)

    download_spider(data_dir)

    if not args.skip_wikisql:
        download_wikisql(data_dir, subset_size=10000)

    download_spider_databases(data_dir)
    stats = verify_data(data_dir)

    with open(data_dir / "download_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("All datasets ready!")
    print("=" * 60)


def cmd_train_a(args):
    """Train Track A: Qwen3 /no_think (non-reasoning)."""
    from training.train_qlora import load_config, train_qlora

    config = load_config("configs/track_a_qlora.yaml")
    summary = train_qlora(config, Path(args.data_dir), Path(args.results_dir))

    print(f"\n{'='*60}")
    print("Training Summary:")
    print(json.dumps(summary, indent=2))
    print(f"{'='*60}")


def cmd_train_b(args):
    """Train Track B: Qwen3 /think (reasoning)."""
    from training.train_qlora import load_config, train_qlora

    config = load_config("configs/track_b_qlora.yaml")
    summary = train_qlora(config, Path(args.data_dir), Path(args.results_dir))

    print(f"\n{'='*60}")
    print("Training Summary:")
    print(json.dumps(summary, indent=2))
    print(f"{'='*60}")


def cmd_train_c(args):
    """Train Track C: T5-50M from scratch."""
    from training.train_t5_scratch import load_config, train_t5_scratch

    config = load_config("configs/track_c_t5_scratch.yaml")
    summary = train_t5_scratch(config, Path(args.data_dir), Path(args.results_dir))

    print(f"\n{'='*60}")
    print("Training Summary:")
    print(json.dumps(summary, indent=2))
    print(f"{'='*60}")


def cmd_evaluate(args):
    """Evaluate all trained models."""
    from evaluation.evaluate import run_full_evaluation

    run_full_evaluation(
        results_dir=Path(args.results_dir),
        data_dir=Path(args.data_dir),
        databases_dir=Path(args.data_dir) / "databases",
        tracks_to_eval=None,
    )


def cmd_visualize(args):
    """Generate all charts."""
    from visualization.generate_charts import generate_all_charts

    generate_all_charts(
        results_dir=Path(args.results_dir),
        output_dir=Path(args.results_dir) / "charts",
    )


def cmd_status(args):
    """Show status of all training runs."""
    results_dir = Path(args.results_dir)
    print(f"\n{'='*60}")
    print("Training Status")
    print(f"{'='*60}")

    metrics_files = sorted(results_dir.glob("*_metrics.json"))
    if not metrics_files:
        print("  No training metrics found yet.")
        return

    for mf in metrics_files:
        with open(mf) as f:
            data = json.load(f)

        name = data.get("experiment_name", mf.stem)
        status = data.get("status", "unknown")
        epochs = len(data.get("training", {}).get("epochs", []))
        total_time_s = data.get("training", {}).get("total_training_time_s", 0)
        total_time = data.get("training", {}).get("total_training_time_min", round(total_time_s / 60, 1))
        errors = len(data.get("errors", []))
        best_loss = data.get("training", {}).get("best_eval_loss", "N/A")

        params = data.get("model_info", {}).get("total_params_m", "?")
        trainable = data.get("model_info", {}).get("trainable_params_m", "?")

        eval_strategies = list(data.get("evaluation", {}).keys())

        status_icons = {
            "training_complete": "[DONE]",
            "training": "[RUNNING]",
            "error": "[ERROR]",
            "interrupted": "[PAUSED]",
            "initialized": "[INIT]",
        }
        icon = status_icons.get(status, "[?]")

        print(f"\n  {icon} {name}")
        print(f"     Status:     {status}")
        print(f"     Params:     {params}M total / {trainable}M trainable")
        print(f"     Epochs:     {epochs}")
        print(f"     Time:       {total_time} min")
        print(f"     Best Loss:  {best_loss}")
        print(f"     Errors:     {errors}")
        if eval_strategies:
            print(f"     Evaluated:  {', '.join(eval_strategies)}")

    print()


def cmd_all(args):
    """Run everything end-to-end."""
    print("=" * 60)
    print("FULL PIPELINE: Download > Train > Evaluate > Visualize")
    print("=" * 60)

    print("\n[1/6] Downloading datasets...")
    cmd_download(args)

    print("\n[2/6] Training Track A (Qwen3 /no_think)...")
    try:
        cmd_train_a(args)
    except Exception as e:
        print(f"  [ERROR] Track A failed: {e}")

    print("\n[3/6] Training Track B (Qwen3 /think)...")
    try:
        cmd_train_b(args)
    except Exception as e:
        print(f"  [ERROR] Track B failed: {e}")

    print("\n[4/6] Training Track C (T5-50M scratch)...")
    try:
        cmd_train_c(args)
    except Exception as e:
        print(f"  [ERROR] Track C failed: {e}")

    print("\n[5/6] Evaluating all models...")
    try:
        cmd_evaluate(args)
    except Exception as e:
        print(f"  [ERROR] Evaluation failed: {e}")

    print("\n[6/6] Generating visualizations...")
    try:
        cmd_visualize(args)
    except Exception as e:
        print(f"  [ERROR] Visualization failed: {e}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    cmd_status(args)


def main():
    parser = argparse.ArgumentParser(
        description="Text-to-SQL SLM Optimization - Main Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  download    Download Spider + WikiSQL datasets
  train-a     Train Track A: Qwen3-1.7B /no_think (non-reasoning)
  train-b     Train Track B: Qwen3-1.7B /think (reasoning)
  train-c     Train Track C: T5-50M from scratch
  evaluate    Run evaluation on all trained models
  visualize   Generate presentation charts
  status      Show training status
  all         Run entire pipeline end-to-end
        """,
    )
    parser.add_argument("step", choices=[
        "download", "train-a", "train-b", "train-c",
        "evaluate", "visualize", "status", "all",
    ])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--skip-wikisql", action="store_true")
    args = parser.parse_args()

    commands = {
        "download": cmd_download,
        "train-a": cmd_train_a,
        "train-b": cmd_train_b,
        "train-c": cmd_train_c,
        "evaluate": cmd_evaluate,
        "visualize": cmd_visualize,
        "status": cmd_status,
        "all": cmd_all,
    }

    commands[args.step](args)


if __name__ == "__main__":
    main()
