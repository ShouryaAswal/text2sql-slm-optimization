"""
T5 from-scratch training for Track C.
Two-phase: WikiSQL pre-warm → Spider fine-tune.

Features: checkpointing, memory tracking, error recovery, metric persistence.
Usage: python training/train_t5_scratch.py --config configs/track_c_t5_scratch.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.model_loader import load_t5_from_scratch, get_gpu_memory_stats, count_parameters
from data.preprocess import load_and_preprocess_spider, load_and_preprocess_wikisql
from data.prompt_templates import apply_prompt_template
from training.metrics_tracker import MetricsTracker


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_seq2seq_dataset(
    samples: list[dict],
    tokenizer,
    max_source_length: int = 512,
    max_target_length: int = 256,
    prompt_strategy: str = "baseline",
) -> Dataset:
    """Tokenize samples for T5 seq2seq format."""
    inputs = []
    targets = []

    for sample in samples:
        prompt = apply_prompt_template(
            schema=sample["schema"],
            question=sample["question"],
            track="track_c",
            strategy=prompt_strategy,
        )
        inputs.append(prompt)
        targets.append(sample["query"])

    # Tokenize
    model_inputs = tokenizer(
        inputs, max_length=max_source_length, truncation=True, padding="max_length",
    )
    labels = tokenizer(
        targets, max_length=max_target_length, truncation=True, padding="max_length",
    )

    # Replace padding token id with -100 for labels
    label_ids = []
    for label in labels["input_ids"]:
        label_ids.append([
            l if l != tokenizer.pad_token_id else -100 for l in label
        ])

    model_inputs["labels"] = label_ids
    return Dataset.from_dict(model_inputs)


def train_phase(
    phase_name: str,
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    config: dict,
    output_dir: Path,
    tracker: MetricsTracker,
):
    """Run a single training phase (pre-warm or fine-tune)."""
    print(f"\n{'='*60}")
    print(f"  Phase: {phase_name}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"{'='*60}")

    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 2),
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        weight_decay=config.get("weight_decay", 0.01),
        fp16=config.get("fp16", True),
        logging_steps=config.get("logging_steps", 100),
        save_steps=config.get("save_steps", 1000),
        save_total_limit=config.get("save_total_limit", 2),
        eval_strategy="epoch",
        predict_with_generate=False,
        report_to="none",
        seed=config.get("seed", 42),
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Check for resume
    resume_from = None
    if (output_dir / "trainer_state.json").exists():
        print(f"  [RESUME] Found checkpoint in {output_dir}")
        resume_from = str(output_dir)

    tracker.start_epoch(0)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    result = trainer.train(resume_from_checkpoint=resume_from)

    # Log phase metrics
    phase_metrics = {
        "phase": phase_name,
        "train_loss": result.training_loss,
        "runtime_s": round(result.metrics.get("train_runtime", 0), 2),
        "samples_per_second": round(result.metrics.get("train_samples_per_second", 0), 2),
    }
    tracker.end_epoch(0, phase_metrics)

    # Eval
    eval_results = trainer.evaluate()
    phase_metrics["eval_loss"] = eval_results.get("eval_loss")

    # Save
    trainer.save_model(str(output_dir / "final"))
    tracker.log_checkpoint(str(output_dir / "final"), result.global_step, phase_metrics)

    return model, phase_metrics


def train_t5_scratch(config: dict, data_dir: Path, results_dir: Path):
    """Main T5 from-scratch training with both phases."""
    experiment_name = "track_c_t5_scratch"
    tracker = MetricsTracker(experiment_name, results_dir)
    tracker.data["config"] = config

    model_config = config.get("model", {}).get("config", {})
    train_config = config["training"]
    output_base = Path(train_config["output_dir"])

    print(f"\n{'='*60}")
    print(f"T5 From-Scratch Training (Track C)")
    print(f"{'='*60}")

    model = None
    try:
        # ---- Load model ----
        model, tokenizer, memory_report = load_t5_from_scratch(
            config_overrides=model_config if model_config else None,
        )
        tracker.set_model_info(count_parameters(model))
        tracker.set_memory_report(memory_report)
        tracker.start_training()

        # ---- Phase 1: WikiSQL Pre-warm ----
        prewarm_config = train_config.get("prewarm")
        if prewarm_config:
            print("\n  Loading WikiSQL for pre-warming...")
            wiki_data = load_and_preprocess_wikisql(
                data_dir, max_samples=prewarm_config.get("num_samples", 10000),
            )
            if wiki_data:
                max_src = train_config.get("max_source_length", 512)
                max_tgt = train_config.get("max_target_length", 256)

                wiki_train = prepare_seq2seq_dataset(
                    wiki_data[:int(len(wiki_data)*0.9)],
                    tokenizer, max_src, max_tgt,
                )
                wiki_eval = prepare_seq2seq_dataset(
                    wiki_data[int(len(wiki_data)*0.9):],
                    tokenizer, max_src, max_tgt,
                )
                model, prewarm_metrics = train_phase(
                    "WikiSQL Pre-warm", model, tokenizer,
                    wiki_train, wiki_eval,
                    prewarm_config, output_base / "prewarm",
                    tracker,
                )
                print(f"  Pre-warm loss: {prewarm_metrics['train_loss']:.4f}")
            else:
                print("  [SKIP] No WikiSQL data available, skipping pre-warm.")
        else:
            print("  [SKIP] No pre-warm phase configured.")

        # ---- Phase 2: Spider Fine-tune ----
        print("\n  Loading Spider for fine-tuning...")
        finetune_config = train_config.get("finetune", train_config)
        spider_train = load_and_preprocess_spider(data_dir, "train")
        spider_eval = load_and_preprocess_spider(data_dir, "validation")

        max_src = train_config.get("max_source_length", 512)
        max_tgt = train_config.get("max_target_length", 256)

        train_dataset = prepare_seq2seq_dataset(
            spider_train, tokenizer, max_src, max_tgt,
        )
        eval_dataset = prepare_seq2seq_dataset(
            spider_eval, tokenizer, max_src, max_tgt,
        )

        model, finetune_metrics = train_phase(
            "Spider Fine-tune", model, tokenizer,
            train_dataset, eval_dataset,
            finetune_config, output_base / "finetune",
            tracker,
        )

        # ---- Finalize ----
        tracker.end_training({
            "prewarm": prewarm_config is not None,
            "finetune_loss": finetune_metrics["train_loss"],
            "finetune_eval_loss": finetune_metrics.get("eval_loss"),
        })

        # Save final model
        final_path = output_base / "final_model"
        model.save_pretrained(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        print(f"\n  [SUCCESS] T5 training complete!")
        print(f"  Model saved to: {final_path}")

    except torch.cuda.OutOfMemoryError as e:
        tracker.log_error(f"CUDA OOM: {e}", {"traceback": traceback.format_exc()})
        gc.collect()
        torch.cuda.empty_cache()
        raise

    except KeyboardInterrupt:
        tracker.log_error("Interrupted by user", {"type": "interrupt"})
        tracker.data["status"] = "interrupted"
        tracker._save()

    except Exception as e:
        tracker.log_error(str(e), {"type": type(e).__name__, "traceback": traceback.format_exc()})
        raise

    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return tracker.get_summary()


def main():
    parser = argparse.ArgumentParser(description="T5 from-scratch training (Track C)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--results-dir", type=str, default="./results", help="Results directory")
    args = parser.parse_args()

    config = load_config(args.config)
    summary = train_t5_scratch(config, Path(args.data_dir), Path(args.results_dir))

    print(f"\n{'='*60}")
    print("Training Summary:")
    print(json.dumps(summary, indent=2))
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
