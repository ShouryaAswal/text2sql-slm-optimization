"""
QLoRA fine-tuning script for Tracks A and B.
- Track A: Qwen3-1.7B /no_think (non-reasoning)
- Track B: Qwen3-1.7B /think (reasoning)

Features: checkpointing, memory tracking, error recovery, metric persistence.
Usage: python training/train_qlora.py --config configs/track_a_qlora.yaml
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer, SFTConfig

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.model_loader import load_qlora_model, get_gpu_memory_stats, count_parameters
from data.preprocess import load_and_preprocess_spider
from data.prompt_templates import apply_prompt_template
from training.metrics_tracker import MetricsTracker


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_dataset(
    data_dir: Path,
    track: str,
    tokenizer,
    max_seq_length: int,
    split: str = "train",
) -> Dataset:
    """
    Load Spider data, apply prompt templates, and tokenize.
    For training, we use "baseline" prompts (PR/RE2 are inference-time only).
    """
    print(f"\n  Preparing {split} dataset for {track}...")

    raw_data = load_and_preprocess_spider(data_dir, split=split)
    prompt_strategy = "baseline"  # Train on baseline; test others at inference

    formatted = []
    for sample in raw_data:
        prompt = apply_prompt_template(
            schema=sample["schema"],
            question=sample["question"],
            track=track,
            strategy=prompt_strategy,
        )
        # Format as instruction → completion
        text = f"{prompt}\n{sample['query']}"
        formatted.append({"text": text})

    ds = Dataset.from_list(formatted)
    print(f"  {split}: {len(ds)} samples prepared")
    return ds


class CheckpointCallback:
    """Custom callback-like logic for checkpoint tracking."""

    def __init__(self, tracker: MetricsTracker):
        self.tracker = tracker
        self.step_count = 0

    def on_step(self, step: int, logs: dict):
        """Called periodically during training."""
        self.step_count = step
        if logs:
            self.tracker.log_step(step, {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in logs.items()
            })

    def on_save(self, path: str, step: int, metrics: dict = None):
        """Called when a checkpoint is saved."""
        self.tracker.log_checkpoint(path, step, metrics)


def train_qlora(config: dict, data_dir: Path, results_dir: Path):
    """Main QLoRA training function with full error handling."""
    model_config = config["model"]
    quant_config = config["quantization"]
    lora_config = config["lora"]
    train_config = config["training"]
    data_config = config["data"]

    track = "track_a" if not model_config.get("thinking_mode", False) else "track_b"
    experiment_name = f"{track}_qwen3"

    # Initialize metrics tracker
    tracker = MetricsTracker(experiment_name, results_dir)
    tracker.data["config"] = config

    print(f"\n{'='*60}")
    print(f"QLoRA Training: {experiment_name}")
    print(f"Track: {track} ({'reasoning /think' if 'b' in track else 'non-reasoning /no_think'})")
    print(f"{'='*60}")

    model = None
    try:
        # ---- Step 1: Load model ----
        model, tokenizer, memory_report = load_qlora_model(
            model_name=model_config["name"],
            lora_config=lora_config,
            quantization_config=quant_config,
            trust_remote_code=model_config.get("trust_remote_code", True),
        )
        tracker.set_model_info(count_parameters(model))
        tracker.set_memory_report(memory_report)

        # ---- Step 2: Prepare data ----
        train_dataset = prepare_dataset(
            data_dir, track, tokenizer,
            max_seq_length=train_config["max_seq_length"],
            split="train",
        )
        eval_dataset = prepare_dataset(
            data_dir, track, tokenizer,
            max_seq_length=train_config["max_seq_length"],
            split="validation",
        )

        # ---- Step 3: Training arguments ----
        output_dir = Path(train_config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        sft_config = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=train_config["num_train_epochs"],
            per_device_train_batch_size=train_config["per_device_train_batch_size"],
            gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
            learning_rate=train_config["learning_rate"],
            lr_scheduler_type=train_config["lr_scheduler_type"],
            warmup_ratio=train_config["warmup_ratio"],
            weight_decay=train_config["weight_decay"],
            max_grad_norm=train_config["max_grad_norm"],
            max_seq_length=train_config["max_seq_length"],
            fp16=train_config.get("fp16", False),
            bf16=train_config.get("bf16", True),
            logging_steps=train_config.get("logging_steps", 50),
            save_steps=train_config.get("save_steps", 500),
            save_total_limit=train_config.get("save_total_limit", 2),
            seed=train_config.get("seed", 42),
            optim=train_config.get("optim", "paged_adamw_8bit"),
            gradient_checkpointing=train_config.get("gradient_checkpointing", True),
            eval_strategy="epoch",
            save_strategy="steps",
            logging_dir=str(output_dir / "logs"),
            report_to="none",
            load_best_model_at_end=False,
            dataset_text_field="text",
        )

        # ---- Step 4: Resume check ----
        resume_from = None
        if (output_dir / "trainer_state.json").exists():
            print("\n  [RESUME] Found existing checkpoint, attempting resume...")
            resume_from = str(output_dir)

        # ---- Step 5: Train ----
        print(f"\n  Starting training ({'resuming' if resume_from else 'from scratch'})...")
        tracker.start_training()

        mem_before_train = get_gpu_memory_stats()
        tracker.data["memory_report"]["before_training"] = mem_before_train

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_config,
        )

        # Train
        train_result = trainer.train(resume_from_checkpoint=resume_from)

        # ---- Step 6: Save & record ----
        mem_after_train = get_gpu_memory_stats()
        tracker.data["memory_report"]["after_training"] = mem_after_train

        final_metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime_s": round(train_result.metrics.get("train_runtime", 0), 2),
            "train_samples_per_second": round(train_result.metrics.get("train_samples_per_second", 0), 2),
            "train_steps_per_second": round(train_result.metrics.get("train_steps_per_second", 0), 2),
            "total_steps": train_result.global_step,
        }
        tracker.end_training(final_metrics)

        # Save final adapter
        final_path = output_dir / "final_adapter"
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        tracker.log_checkpoint(
            str(final_path), train_result.global_step,
            {"train_loss": train_result.training_loss}
        )

        # Run final eval
        print("\n  Running final evaluation...")
        eval_results = trainer.evaluate()
        tracker.end_epoch(
            epoch=train_config["num_train_epochs"],
            metrics={
                "train_loss": train_result.training_loss,
                "eval_loss": eval_results.get("eval_loss", None),
            },
        )

        print(f"\n  [SUCCESS] Training complete!")
        print(f"  Final train loss: {train_result.training_loss:.4f}")
        print(f"  Final eval loss:  {eval_results.get('eval_loss', 'N/A')}")
        print(f"  Model saved to:   {final_path}")
        print(f"  Metrics saved to: {tracker.metrics_file}")

    except torch.cuda.OutOfMemoryError as e:
        error_msg = f"CUDA OOM: {e}"
        print(f"\n  [ERROR] {error_msg}")
        tracker.log_error(error_msg, {"type": "OOM", "traceback": traceback.format_exc()})
        gc.collect()
        torch.cuda.empty_cache()
        raise

    except KeyboardInterrupt:
        print("\n  [INTERRUPTED] Training interrupted. Metrics saved so far.")
        tracker.log_error("Training interrupted by user", {"type": "interrupt"})
        tracker.data["status"] = "interrupted"
        tracker._save()

    except Exception as e:
        error_msg = f"Training error: {e}"
        print(f"\n  [ERROR] {error_msg}")
        tracker.log_error(error_msg, {"type": type(e).__name__, "traceback": traceback.format_exc()})
        raise

    finally:
        # Always clean up GPU
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return tracker.get_summary()


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Tracks A/B")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--results-dir", type=str, default="./results", help="Results directory")
    args = parser.parse_args()

    config = load_config(args.config)
    summary = train_qlora(config, Path(args.data_dir), Path(args.results_dir))

    print(f"\n{'='*60}")
    print("Training Summary:")
    print(json.dumps(summary, indent=2))
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
