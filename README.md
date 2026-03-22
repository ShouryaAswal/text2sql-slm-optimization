# Text-to-SQL SLM Optimization

**Research project** implementing and comparing two prompting techniques on Small Language Models for Text-to-SQL generation:

1. **Prompt Repetition** (Google, 2025) — for non-reasoning / 0-shot models
2. **RE2 Re-Reading** (Xu et al., EMNLP 2024) — for reasoning models with Chain-of-Thought

## Key Insight

These two papers look similar (both repeat the input) but target completely different mechanisms:
- **PR** exploits *causal attention* — fakes bidirectional attention via structural duplication
- **RE2** exploits *semantic context* — forces attention re-prioritization via natural language

## Experiment Design

Three tracks using **Qwen3-1.7B** (same model, two modes) + **T5-50M from scratch**:

| Track | Model | Mode | Purpose |
|-------|-------|------|---------|
| A | Qwen3-1.7B | `/no_think` (non-reasoning) | Prompt Repetition testing |
| B | Qwen3-1.7B | `/think` (reasoning) | RE2 Re-Reading testing |
| C | T5-50M | from scratch | Controlled baseline |

12 experimental conditions × Spider dev set evaluation.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/text2sql-slm-optimization.git
cd text2sql-slm-optimization
cp .env.example .env
# Edit .env with your HF_TOKEN

# Install (use Kaggle notebook for GPU training)
pip install -r requirements.txt

# Download data
python data/download_data.py

# Train (see notebooks/02_training.ipynb for Kaggle)
python training/train_qlora.py --config configs/track_a_qlora.yaml
python training/train_t5_scratch.py --config configs/track_c_t5_scratch.yaml

# Evaluate
python evaluation/evaluate.py --all-strategies
```

## Project Structure

```
├── configs/           # Training configs (YAML)
├── data/              # Data download, preprocessing, prompt templates
├── models/            # Model loading and T5 architecture
├── training/          # Training scripts (QLoRA + T5 scratch)
├── evaluation/        # SQL execution and metrics
├── inference/         # Generation and strategy comparison
├── notebooks/         # Kaggle-ready training notebooks
├── papers/            # Research paper PDFs (add manually)
└── results/           # Evaluation outputs
```

## Hardware Requirements

- **Training**: Kaggle T4 16GB (free tier) — ~26h total across all tracks
- **Inference**: Any GPU with ≥4GB VRAM or CPU (slower)

## Papers

- [Prompt Repetition (Google, 2025)](https://arxiv.org/abs/2504.XXXXX) — Non-reasoning
- [RE2 Re-Reading (Xu et al., EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.XXX/) — Reasoning

## License

MIT
