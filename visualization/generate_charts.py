"""
Publication-quality visualization for research findings.
Generates charts for presentation to a panel.

Usage: python visualization/generate_charts.py --results-dir ./results --output-dir ./results/charts
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Kaggle
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns


# ============================================================
# Style Configuration — premium, presentation-ready
# ============================================================

def setup_style():
    """Set up publication-quality plot style."""
    plt.rcParams.update({
        "figure.figsize": (12, 7),
        "figure.dpi": 150,
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#e6edf3",
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "axes.grid": True,
        "grid.color": "#21262d",
        "grid.alpha": 0.6,
        "text.color": "#e6edf3",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.facecolor": "#161b22",
        "legend.edgecolor": "#30363d",
        "font.family": "sans-serif",
        "savefig.bbox": "tight",
        "savefig.facecolor": "#0d1117",
        "savefig.pad_inches": 0.3,
    })


STRATEGY_COLORS = {
    "baseline": "#8b949e",
    "prompt_repetition": "#58a6ff",
    "re2_rereading": "#3fb950",
    "combined": "#d2a8ff",
}

STRATEGY_LABELS = {
    "baseline": "Baseline",
    "prompt_repetition": "Prompt Repetition (PR)",
    "re2_rereading": "RE2 Re-Reading",
    "combined": "PR + RE2 Combined",
}

TRACK_LABELS = {
    "track_a": "Track A: Qwen3 /no_think\n(Non-Reasoning)",
    "track_b": "Track B: Qwen3 /think\n(Reasoning)",
    "track_c": "Track C: T5-50M\n(From Scratch)",
}

TRACK_SHORT = {
    "track_a": "Qwen3 /no_think",
    "track_b": "Qwen3 /think",
    "track_c": "T5-50M Scratch",
}


# ============================================================
# Chart Generators
# ============================================================

def chart_accuracy_comparison(results: dict, output_dir: Path):
    """
    Main chart: Execution Accuracy across all 12 conditions.
    Grouped bar chart — tracks on x-axis, strategies as bar groups.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 8))

    tracks = ["track_a", "track_b", "track_c"]
    strategies = ["baseline", "prompt_repetition", "re2_rereading", "combined"]
    x = np.arange(len(tracks))
    width = 0.18

    for i, strategy in enumerate(strategies):
        values = []
        for track in tracks:
            key = f"{track}_{strategy}"
            acc = results.get(key, {}).get("summary", {}).get("execution_accuracy", 0)
            values.append(acc)

        bars = ax.bar(
            x + i * width - 1.5 * width,
            values,
            width,
            label=STRATEGY_LABELS[strategy],
            color=STRATEGY_COLORS[strategy],
            edgecolor="#30363d",
            linewidth=0.8,
            zorder=3,
        )
        # Value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}%",
                    ha="center", va="bottom",
                    fontsize=9, color="#e6edf3", fontweight="bold",
                )

    ax.set_xlabel("Model Track", fontsize=14, fontweight="bold")
    ax.set_ylabel("Execution Accuracy (%)", fontsize=14, fontweight="bold")
    ax.set_title("Execution Accuracy: All Strategies × All Tracks", fontsize=18, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([TRACK_LABELS[t] for t in tracks], fontsize=11)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())

    plt.savefig(output_dir / "accuracy_comparison.png", dpi=200)
    plt.close()
    print(f"  Saved: accuracy_comparison.png")


def chart_difficulty_breakdown(results: dict, output_dir: Path):
    """
    Heatmap: Execution accuracy by difficulty level across conditions.
    """
    setup_style()

    difficulties = ["easy", "medium", "hard", "extra"]
    tracks = ["track_a", "track_b", "track_c"]
    strategies = ["baseline", "prompt_repetition", "re2_rereading", "combined"]

    conditions = [f"{t}_{s}" for t in tracks for s in strategies]
    condition_labels = [f"{TRACK_SHORT[t]}\n{STRATEGY_LABELS[s]}" for t in tracks for s in strategies]

    matrix = []
    for condition in conditions:
        row = []
        by_diff = results.get(condition, {}).get("summary", {}).get("by_difficulty", {})
        for diff in difficulties:
            row.append(by_diff.get(diff, {}).get("execution_accuracy", 0))
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(10, 16))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=[d.capitalize() for d in difficulties],
        yticklabels=condition_labels,
        ax=ax,
        linewidths=0.5,
        linecolor="#30363d",
        cbar_kws={"label": "Execution Accuracy (%)"},
    )
    ax.set_title("Accuracy by SQL Difficulty", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Query Difficulty", fontsize=13)

    plt.savefig(output_dir / "difficulty_heatmap.png", dpi=200)
    plt.close()
    print(f"  Saved: difficulty_heatmap.png")


def chart_inference_latency(results: dict, output_dir: Path):
    """
    Inference latency comparison — bar chart with error bars.
    Shows the cost of PR (double tokens) vs RE2 (semantic overhead).
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    strategies = ["baseline", "prompt_repetition", "re2_rereading", "combined"]
    tracks = ["track_a", "track_b", "track_c"]

    # Plot 1: Latency
    x = np.arange(len(strategies))
    width = 0.25
    for i, track in enumerate(tracks):
        values = []
        for strategy in strategies:
            key = f"{track}_{strategy}"
            lat = results.get(key, {}).get("inference", {}).get("inference_time_ms", {}).get("mean", 0)
            values.append(lat)
        ax1.bar(
            x + i * width - width,
            values, width,
            label=TRACK_SHORT[track],
            color=list(STRATEGY_COLORS.values())[i],
            edgecolor="#30363d",
            zorder=3,
        )

    ax1.set_xlabel("Strategy", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Mean Inference Time (ms)", fontsize=13, fontweight="bold")
    ax1.set_title("Inference Latency", fontsize=16, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([STRATEGY_LABELS[s] for s in strategies], rotation=20, ha="right")
    ax1.legend()

    # Plot 2: Token overhead
    for i, track in enumerate(tracks):
        values = []
        for strategy in strategies:
            key = f"{track}_{strategy}"
            tokens = results.get(key, {}).get("inference", {}).get("token_counts", {}).get("mean", 0)
            values.append(tokens)
        ax2.bar(
            x + i * width - width,
            values, width,
            label=TRACK_SHORT[track],
            color=list(STRATEGY_COLORS.values())[i],
            edgecolor="#30363d",
            zorder=3,
        )

    ax2.set_xlabel("Strategy", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Mean Input Tokens", fontsize=13, fontweight="bold")
    ax2.set_title("Token Overhead", fontsize=16, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([STRATEGY_LABELS[s] for s in strategies], rotation=20, ha="right")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "inference_latency.png", dpi=200)
    plt.close()
    print(f"  Saved: inference_latency.png")


def chart_training_metrics(results_dir: Path, output_dir: Path):
    """
    Training curves: loss over time for all tracks.
    """
    setup_style()

    metrics_files = list(results_dir.glob("*_metrics.json"))
    if not metrics_files:
        print("  [SKIP] No training metrics files found.")
        return

    fig, axes = plt.subplots(1, len(metrics_files), figsize=(7 * len(metrics_files), 6))
    if len(metrics_files) == 1:
        axes = [axes]

    colors = ["#58a6ff", "#3fb950", "#d2a8ff"]

    for idx, mf in enumerate(sorted(metrics_files)):
        with open(mf) as f:
            data = json.load(f)

        ax = axes[idx]
        name = data.get("experiment_name", mf.stem)
        steps_data = data.get("training", {}).get("steps", [])

        if steps_data:
            steps = [s["step"] for s in steps_data if "loss" in s]
            losses = [s["loss"] for s in steps_data if "loss" in s]
            if steps and losses:
                ax.plot(steps, losses, color=colors[idx % len(colors)],
                        linewidth=2, alpha=0.8, label="Training Loss")
                ax.set_xlabel("Steps", fontsize=12)
                ax.set_ylabel("Loss", fontsize=12)

        ax.set_title(name.replace("_", " ").title(), fontsize=14, fontweight="bold")
        ax.legend()

    fig.suptitle("Training Loss Curves", fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=200)
    plt.close()
    print(f"  Saved: training_curves.png")


def chart_model_comparison_radar(results: dict, output_dir: Path):
    """
    Radar chart: Compare best strategy per track across multiple dimensions.
    """
    setup_style()

    categories = ["EX Accuracy", "EM Accuracy", "Low Error Rate", "Low Latency", "Low Token Cost"]
    tracks = ["track_a", "track_b", "track_c"]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    colors = ["#58a6ff", "#3fb950", "#d2a8ff"]

    for idx, track in enumerate(tracks):
        # Find best strategy by EX accuracy
        best_key = None
        best_ex = -1
        for strategy in ["baseline", "prompt_repetition", "re2_rereading", "combined"]:
            key = f"{track}_{strategy}"
            ex = results.get(key, {}).get("summary", {}).get("execution_accuracy", 0)
            if ex > best_ex:
                best_ex = ex
                best_key = key

        if best_key is None:
            continue

        r = results[best_key]
        summary = r.get("summary", {})
        inf = r.get("inference", {}).get("inference_time_ms", {})

        values = [
            summary.get("execution_accuracy", 0),
            summary.get("exact_match_accuracy", 0),
            100 - summary.get("error_rate", 0),
            max(0, 100 - inf.get("mean", 0) / 10),  # Normalize latency
            max(0, 100 - r.get("inference", {}).get("token_counts", {}).get("mean", 0) / 20),
        ]
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=2, color=colors[idx], label=TRACK_SHORT[track])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1))
    ax.set_title("Best Strategy Per Track — Multi-Dimension", fontsize=16, fontweight="bold", pad=30)

    plt.savefig(output_dir / "radar_comparison.png", dpi=200)
    plt.close()
    print(f"  Saved: radar_comparison.png")


def chart_pr_vs_re2_delta(results: dict, output_dir: Path):
    """
    Key finding chart: Show the accuracy DELTA (improvement over baseline)
    for PR and RE2 on their intended vs cross-test targets.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    comparisons = {
        "PR on Non-Reasoning\n(Intended)": ("track_a_baseline", "track_a_prompt_repetition"),
        "PR on Reasoning\n(Cross-test)": ("track_b_baseline", "track_b_prompt_repetition"),
        "RE2 on Reasoning\n(Intended)": ("track_b_baseline", "track_b_re2_rereading"),
        "RE2 on Non-Reasoning\n(Cross-test)": ("track_a_baseline", "track_a_re2_rereading"),
        "PR on T5\n(Bidi Control)": ("track_c_baseline", "track_c_prompt_repetition"),
        "RE2 on T5\n(Scratch Control)": ("track_c_baseline", "track_c_re2_rereading"),
    }

    labels = list(comparisons.keys())
    deltas = []
    colors = []
    for label, (baseline_key, test_key) in comparisons.items():
        base_acc = results.get(baseline_key, {}).get("summary", {}).get("execution_accuracy", 0)
        test_acc = results.get(test_key, {}).get("summary", {}).get("execution_accuracy", 0)
        delta = test_acc - base_acc
        deltas.append(delta)
        colors.append("#3fb950" if delta > 0 else "#f85149")

    bars = ax.barh(labels, deltas, color=colors, edgecolor="#30363d", height=0.6, zorder=3)

    # Value labels
    for bar, delta in zip(bars, deltas):
        x_pos = bar.get_width() + (0.3 if delta >= 0 else -0.3)
        ha = "left" if delta >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{delta:+.1f}%", va="center", ha=ha,
                fontsize=12, fontweight="bold", color="#e6edf3")

    ax.axvline(x=0, color="#8b949e", linewidth=1.5, linestyle="--", zorder=2)
    ax.set_xlabel("Accuracy Delta vs Baseline (%)", fontsize=13, fontweight="bold")
    ax.set_title("Key Finding: Intended vs Cross-Test Accuracy Deltas",
                 fontsize=16, fontweight="bold", pad=15)
    ax.invert_yaxis()

    plt.savefig(output_dir / "pr_vs_re2_delta.png", dpi=200)
    plt.close()
    print(f"  Saved: pr_vs_re2_delta.png")


def chart_memory_comparison(results_dir: Path, output_dir: Path):
    """
    GPU memory usage comparison across models.
    """
    setup_style()

    metrics_files = list(results_dir.glob("*_metrics.json"))
    if not metrics_files:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    names = []
    allocated = []
    total_params = []

    for mf in sorted(metrics_files):
        with open(mf) as f:
            data = json.load(f)
        name = data.get("experiment_name", mf.stem)
        mem = data.get("memory_report", {})

        # Try different nested paths
        alloc = None
        for key in ["after_lora_injection", "after_load", "after_training"]:
            if key in mem and "allocated_gb" in mem[key]:
                alloc = mem[key]["allocated_gb"]
                break
        if isinstance(mem, dict) and "gpu_memory" in mem:
            for key in ["after_lora_injection", "after_load"]:
                if key in mem["gpu_memory"] and "allocated_gb" in mem["gpu_memory"][key]:
                    alloc = mem["gpu_memory"][key]["allocated_gb"]
                    break

        params_m = data.get("model_info", {}).get("total_params_m", 0)
        if data.get("parameters"):
            params_m = data["parameters"].get("total_params_m", params_m)

        names.append(name.replace("_", " ").title())
        allocated.append(alloc or 0)
        total_params.append(params_m)

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, allocated, width, label="GPU Memory (GB)",
                   color="#58a6ff", edgecolor="#30363d", zorder=3)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, total_params, width, label="Total Params (M)",
                    color="#d2a8ff", edgecolor="#30363d", alpha=0.7, zorder=3)

    ax.set_xlabel("Model", fontsize=13, fontweight="bold")
    ax.set_ylabel("GPU Memory Allocated (GB)", fontsize=12, color="#58a6ff")
    ax2.set_ylabel("Total Parameters (M)", fontsize=12, color="#d2a8ff")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_title("Model Size & Memory Footprint", fontsize=16, fontweight="bold")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.savefig(output_dir / "memory_comparison.png", dpi=200)
    plt.close()
    print(f"  Saved: memory_comparison.png")


# ============================================================
# Main
# ============================================================

def generate_all_charts(results_dir: Path, output_dir: Path):
    """Generate all visualization charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print("Generating Visualizations")
    print(f"{'='*60}")

    # Load combined results
    summary_file = results_dir / "all_results_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            results = json.load(f)

        chart_accuracy_comparison(results, output_dir)
        chart_difficulty_breakdown(results, output_dir)
        chart_inference_latency(results, output_dir)
        chart_pr_vs_re2_delta(results, output_dir)
        chart_model_comparison_radar(results, output_dir)
    else:
        print(f"  [WARN] {summary_file} not found. Run evaluation first.")

    # Training metrics charts (independent of eval results)
    chart_training_metrics(results_dir, output_dir)
    chart_memory_comparison(results_dir, output_dir)

    print(f"\n  All charts saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate visualization charts")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--output-dir", type=str, default="./results/charts")
    args = parser.parse_args()

    generate_all_charts(Path(args.results_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
