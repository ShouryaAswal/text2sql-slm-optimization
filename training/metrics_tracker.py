"""
Persistent metrics tracker for training and evaluation.
Saves all metrics to JSON after every update so training can resume if interrupted.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch


class MetricsTracker:
    """
    Tracks training metrics, checkpoints, timing, and GPU memory.
    Persists all data to a JSON file on every update.
    """

    def __init__(self, experiment_name: str, output_dir: str | Path):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.output_dir / f"{experiment_name}_metrics.json"

        # Initialize or load existing metrics
        if self.metrics_file.exists():
            with open(self.metrics_file, "r") as f:
                self.data = json.load(f)
            print(f"  [MetricsTracker] Resumed from {self.metrics_file}")
        else:
            self.data = {
                "experiment_name": experiment_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "initialized",
                "model_info": {},
                "memory_report": {},
                "training": {
                    "epochs": [],
                    "steps": [],
                    "total_training_time_s": 0,
                    "best_eval_loss": float("inf"),
                    "best_checkpoint": None,
                },
                "evaluation": {},
                "checkpoints": [],
                "errors": [],
            }

        self._train_start_time: Optional[float] = None
        self._epoch_start_time: Optional[float] = None
        self._step_start_time: Optional[float] = None

    def _save(self):
        """Persist metrics to disk."""
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
        # Use a temp file then rename for atomicity
        tmp_file = self.metrics_file.with_suffix(".tmp")
        with open(tmp_file, "w") as f:
            json.dump(self.data, f, indent=2, default=str)
        tmp_file.replace(self.metrics_file)

    def set_model_info(self, info: dict):
        """Record model metadata."""
        self.data["model_info"] = info
        self._save()

    def set_memory_report(self, report: dict):
        """Record GPU memory layout."""
        self.data["memory_report"] = report
        self._save()

    def start_training(self):
        """Mark training start."""
        self._train_start_time = time.time()
        self.data["status"] = "training"
        self.data["training"]["started_at"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def start_epoch(self, epoch: int):
        """Mark epoch start."""
        self._epoch_start_time = time.time()
        self.data["training"]["current_epoch"] = epoch
        self._save()

    def end_epoch(self, epoch: int, metrics: dict):
        """Record epoch completion with metrics."""
        epoch_time = time.time() - (self._epoch_start_time or time.time())

        epoch_data = {
            "epoch": epoch,
            "duration_s": round(epoch_time, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metrics,
        }

        # Add GPU memory snapshot
        if torch.cuda.is_available():
            epoch_data["gpu_memory"] = {
                "allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
                "max_allocated_gb": round(torch.cuda.max_memory_allocated(0) / 1e9, 2),
            }

        self.data["training"]["epochs"].append(epoch_data)
        self.data["training"]["total_training_time_s"] += epoch_time

        # Track best eval loss
        eval_loss = metrics.get("eval_loss")
        if eval_loss and eval_loss < self.data["training"]["best_eval_loss"]:
            self.data["training"]["best_eval_loss"] = eval_loss

        self._save()
        print(f"  [Epoch {epoch}] {epoch_time:.0f}s | loss={metrics.get('train_loss', 'N/A'):.4f}")

    def log_step(self, step: int, metrics: dict):
        """Log a training step (called periodically, not every step)."""
        step_data = {
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metrics,
        }
        self.data["training"]["steps"].append(step_data)

        # Don't save on every step — batch saves at epoch end
        # But save periodically (every 100 logged steps)
        if len(self.data["training"]["steps"]) % 100 == 0:
            self._save()

    def log_checkpoint(self, path: str, step: int, metrics: Optional[dict] = None):
        """Record checkpoint save."""
        checkpoint_data = {
            "path": str(path),
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics or {},
        }
        self.data["checkpoints"].append(checkpoint_data)

        # Update best checkpoint if relevant
        if metrics and metrics.get("eval_loss"):
            if metrics["eval_loss"] < self.data["training"]["best_eval_loss"]:
                self.data["training"]["best_eval_loss"] = metrics["eval_loss"]
                self.data["training"]["best_checkpoint"] = str(path)

        self._save()
        print(f"  [Checkpoint] Saved at step {step}: {path}")

    def end_training(self, final_metrics: Optional[dict] = None):
        """Mark training complete."""
        total_time = time.time() - (self._train_start_time or time.time())
        self.data["status"] = "training_complete"
        self.data["training"]["total_training_time_s"] = round(total_time, 2)
        self.data["training"]["total_training_time_min"] = round(total_time / 60, 1)
        self.data["training"]["finished_at"] = datetime.now(timezone.utc).isoformat()
        if final_metrics:
            self.data["training"]["final_metrics"] = final_metrics
        self._save()
        print(f"\n  [Training Complete] Total time: {total_time/60:.1f} minutes")

    def log_evaluation(self, strategy: str, metrics: dict):
        """Record evaluation results for a given strategy."""
        eval_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metrics,
        }
        self.data["evaluation"][strategy] = eval_data
        self._save()

    def log_error(self, error_msg: str, context: Optional[dict] = None):
        """Record an error that occurred during training."""
        error_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": error_msg,
            "context": context or {},
        }
        if torch.cuda.is_available():
            error_data["gpu_memory_at_error"] = {
                "allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
                "max_allocated_gb": round(torch.cuda.max_memory_allocated(0) / 1e9, 2),
            }
        self.data["errors"].append(error_data)
        self.data["status"] = "error"
        self._save()

    def get_summary(self) -> dict:
        """Return a concise summary for display."""
        training = self.data["training"]
        return {
            "experiment": self.experiment_name,
            "status": self.data["status"],
            "epochs_completed": len(training["epochs"]),
            "total_time_min": training.get("total_training_time_min",
                                           round(training.get("total_training_time_s", 0) / 60, 1)),
            "best_eval_loss": training.get("best_eval_loss"),
            "best_checkpoint": training.get("best_checkpoint"),
            "num_errors": len(self.data.get("errors", [])),
            "strategies_evaluated": list(self.data.get("evaluation", {}).keys()),
        }
