"""
Prompt templates for all experimental conditions.

Track A (non-reasoning /no_think): baseline, PR, RE2, PR+RE2  
Track B (reasoning /think):        baseline_cot, PR, RE2, PR+RE2  
Track C (T5 seq2seq):              baseline, PR, RE2, PR+RE2  
"""

from typing import Literal

PromptStrategy = Literal["baseline", "prompt_repetition", "re2_rereading", "combined"]
ModelTrack = Literal["track_a", "track_b", "track_c"]


# ============================================================
# Track A: Non-Reasoning (Qwen3 /no_think) — 0-shot prompts
# ============================================================

def track_a_baseline(schema: str, question: str) -> str:
    """Standard 0-shot prompt for non-reasoning model."""
    return (
        f"<schema>\n{schema}\n</schema>\n"
        f"<question>{question}</question>\n"
        f"Generate the SQL query:"
    )


def track_a_prompt_repetition(schema: str, question: str) -> str:
    """PR: raw concatenation of the full prompt (Google's approach)."""
    base = track_a_baseline(schema, question)
    return f"{base}\n{base}"


def track_a_re2_rereading(schema: str, question: str) -> str:
    """RE2 on non-reasoning model (cross-test)."""
    return (
        f"<schema>\n{schema}\n</schema>\n"
        f"<question>{question}</question>\n"
        f"Read the question again: {question}\n"
        f"Generate the SQL query:"
    )


def track_a_combined(schema: str, question: str) -> str:
    """PR + RE2 combined on non-reasoning model."""
    base = (
        f"<schema>\n{schema}\n</schema>\n"
        f"<question>{question}</question>\n"
    )
    return (
        f"{base}"
        f"{base}"
        f"Read the question again: {question}\n"
        f"Generate the SQL query:"
    )


# ============================================================
# Track B: Reasoning (Qwen3 /think) — CoT prompts
# ============================================================

def track_b_baseline(schema: str, question: str) -> str:
    """CoT baseline for reasoning model."""
    return (
        f"<schema>\n{schema}\n</schema>\n"
        f"<question>{question}</question>\n"
        f"Think step by step about which tables, columns, and joins are needed, "
        f"then generate the SQL query."
    )


def track_b_prompt_repetition(schema: str, question: str) -> str:
    """PR on reasoning model (cross-test)."""
    base = track_b_baseline(schema, question)
    return f"{base}\n{base}"


def track_b_re2_rereading(schema: str, question: str) -> str:
    """RE2 on reasoning model (Xu et al.'s intended use)."""
    return (
        f"<schema>\n{schema}\n</schema>\n"
        f"<question>{question}</question>\n"
        f"Read the question again: {question}\n"
        f"Think step by step about which tables, columns, and joins are needed, "
        f"then generate the SQL query."
    )


def track_b_combined(schema: str, question: str) -> str:
    """PR + RE2 combined on reasoning model."""
    base = (
        f"<schema>\n{schema}\n</schema>\n"
        f"<question>{question}</question>\n"
    )
    return (
        f"{base}"
        f"{base}"
        f"Read the question again: {question}\n"
        f"Think step by step about which tables, columns, and joins are needed, "
        f"then generate the SQL query."
    )


# ============================================================
# Track C: T5 Seq2Seq — encoder-decoder prompts
# ============================================================

def track_c_baseline(schema: str, question: str) -> str:
    """Seq2seq format for T5."""
    return f"translate to SQL: {schema} | {question}"


def track_c_prompt_repetition(schema: str, question: str) -> str:
    """PR for T5 — repeated input."""
    base = f"{schema} | {question}"
    return f"translate to SQL: {base} | {base}"


def track_c_re2_rereading(schema: str, question: str) -> str:
    """RE2 for T5."""
    return f"translate to SQL: {schema} | {question} | Read the question again: {question}"


def track_c_combined(schema: str, question: str) -> str:
    """PR + RE2 for T5."""
    base = f"{schema} | {question}"
    return f"translate to SQL: {base} | {base} | Read the question again: {question}"


# ============================================================
# Dispatcher
# ============================================================

TEMPLATE_REGISTRY = {
    "track_a": {
        "baseline": track_a_baseline,
        "prompt_repetition": track_a_prompt_repetition,
        "re2_rereading": track_a_re2_rereading,
        "combined": track_a_combined,
    },
    "track_b": {
        "baseline": track_b_baseline,
        "prompt_repetition": track_b_prompt_repetition,
        "re2_rereading": track_b_re2_rereading,
        "combined": track_b_combined,
    },
    "track_c": {
        "baseline": track_c_baseline,
        "prompt_repetition": track_c_prompt_repetition,
        "re2_rereading": track_c_re2_rereading,
        "combined": track_c_combined,
    },
}


def apply_prompt_template(
    schema: str,
    question: str,
    track: ModelTrack,
    strategy: PromptStrategy,
) -> str:
    """
    Apply the correct prompt template for a given track and strategy.

    Args:
        schema: Linearized database schema
        question: Natural language question
        track: "track_a", "track_b", or "track_c"
        strategy: "baseline", "prompt_repetition", "re2_rereading", or "combined"

    Returns:
        Formatted prompt string.
    """
    if track not in TEMPLATE_REGISTRY:
        raise ValueError(f"Unknown track: {track}. Choose from {list(TEMPLATE_REGISTRY.keys())}")
    if strategy not in TEMPLATE_REGISTRY[track]:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(TEMPLATE_REGISTRY[track].keys())}")

    return TEMPLATE_REGISTRY[track][strategy](schema, question)


def get_all_strategies() -> list[str]:
    """Return list of all strategy names."""
    return ["baseline", "prompt_repetition", "re2_rereading", "combined"]


def get_all_tracks() -> list[str]:
    """Return list of all track names."""
    return ["track_a", "track_b", "track_c"]
