"""
Model loading utilities for QLoRA (Tracks A/B) and T5 from scratch (Track C).
Handles quantization, LoRA injection, and memory reporting.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    T5Config,
    T5ForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def get_gpu_memory_stats() -> dict:
    """Get detailed GPU memory statistics."""
    if not torch.cuda.is_available():
        return {"gpu_available": False}

    stats = {
        "gpu_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "total_memory_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2),
        "allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
        "reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2),
        "free_gb": round((torch.cuda.get_device_properties(0).total_mem - torch.cuda.memory_allocated(0)) / 1e9, 2),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated(0) / 1e9, 2),
    }
    return stats


def count_parameters(model) -> dict:
    """Count total, trainable, and frozen parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": frozen,
        "total_params_m": round(total / 1e6, 2),
        "trainable_params_m": round(trainable / 1e6, 2),
        "trainable_pct": round(100 * trainable / total, 2) if total > 0 else 0,
    }


def load_qlora_model(
    model_name: str,
    lora_config: dict,
    quantization_config: dict,
    trust_remote_code: bool = True,
) -> tuple:
    """
    Load a model with 4-bit quantization and apply LoRA adapters.

    Returns:
        (model, tokenizer, memory_report)
    """
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"{'='*60}")

    # Pre-loading memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    mem_before = get_gpu_memory_stats()
    load_start = time.time()

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quantization_config.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=getattr(torch, quantization_config.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_quant_type=quantization_config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=quantization_config.get("bnb_4bit_use_double_quant", True),
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16,
    )

    mem_after_load = get_gpu_memory_stats()

    # Prepare for kbit training
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_cfg = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "CAUSAL_LM"),
        target_modules=lora_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
    )
    model = get_peft_model(model, lora_cfg)

    load_time = time.time() - load_start
    mem_after_lora = get_gpu_memory_stats()
    param_info = count_parameters(model)

    # Build memory report
    memory_report = {
        "model_name": model_name,
        "load_time_seconds": round(load_time, 2),
        "parameters": param_info,
        "gpu_memory": {
            "before_load": mem_before,
            "after_quantized_load": mem_after_load,
            "after_lora_injection": mem_after_lora,
        },
        "quantization": quantization_config,
        "lora": {k: v for k, v in lora_config.items() if k != "target_modules"},
    }

    # Print report
    print(f"\n  Model loaded in {load_time:.1f}s")
    print(f"  Total params:     {param_info['total_params_m']}M")
    print(f"  Trainable params: {param_info['trainable_params_m']}M ({param_info['trainable_pct']}%)")
    print(f"  GPU allocated:    {mem_after_lora.get('allocated_gb', 'N/A')} GB")
    print(f"  GPU free:         {mem_after_lora.get('free_gb', 'N/A')} GB")

    return model, tokenizer, memory_report


def load_t5_from_scratch(
    config_overrides: Optional[dict] = None,
) -> tuple:
    """
    Initialize a T5 model from scratch (~50M params).

    Returns:
        (model, tokenizer, memory_report)
    """
    print(f"\n{'='*60}")
    print(f"Initializing T5 from scratch")
    print(f"{'='*60}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    mem_before = get_gpu_memory_stats()
    load_start = time.time()

    # Base T5-small config (~60M params)
    default_config = {
        "vocab_size": 32128,
        "d_model": 512,
        "d_ff": 2048,
        "d_kv": 64,
        "num_heads": 8,
        "num_layers": 6,
        "num_decoder_layers": 6,
    }
    if config_overrides:
        default_config.update(config_overrides)

    t5_config = T5Config(**default_config)
    model = T5ForConditionalGeneration(t5_config)

    # Use standard T5 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    load_time = time.time() - load_start
    mem_after = get_gpu_memory_stats()
    param_info = count_parameters(model)

    memory_report = {
        "model_name": "T5-Custom-Scratch",
        "load_time_seconds": round(load_time, 2),
        "parameters": param_info,
        "gpu_memory": {
            "before_load": mem_before,
            "after_load": mem_after,
        },
        "architecture": default_config,
    }

    print(f"\n  T5 initialized in {load_time:.1f}s")
    print(f"  Total params:     {param_info['total_params_m']}M (all trainable)")
    print(f"  GPU allocated:    {mem_after.get('allocated_gb', 'N/A')} GB")

    return model, tokenizer, memory_report
