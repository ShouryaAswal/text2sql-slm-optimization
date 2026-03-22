"""
SQL execution engine for evaluation.
Executes generated SQL against Spider's SQLite databases and compares results.
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Optional


def normalize_sql(sql: str) -> str:
    """Normalize SQL for exact-match comparison."""
    sql = sql.strip().rstrip(";")
    # Collapse whitespace
    sql = re.sub(r"\s+", " ", sql)
    # Lowercase keywords but preserve string literals
    parts = re.split(r"('[^']*')", sql)
    normalized = []
    for i, part in enumerate(parts):
        if i % 2 == 0:  # Not inside quotes
            normalized.append(part.lower())
        else:
            normalized.append(part)
    return " ".join("".join(normalized).split())


def execute_sql(
    sql: str,
    db_path: str | Path,
    timeout: float = 30.0,
) -> tuple[Optional[list], Optional[str], float]:
    """
    Execute SQL against a SQLite database.

    Returns:
        (results, error_msg, execution_time_ms)
    """
    start = time.perf_counter()
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        elapsed = (time.perf_counter() - start) * 1000
        return results, None, round(elapsed, 2)
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return None, str(e), round(elapsed, 2)


def compare_results(
    pred_results: Optional[list],
    gold_results: Optional[list],
) -> bool:
    """
    Compare query results for execution accuracy.
    Results match if they contain the same rows (order-independent).
    """
    if pred_results is None or gold_results is None:
        return False

    # Convert to sets of tuples for order-independent comparison
    try:
        pred_set = set(tuple(row) for row in pred_results)
        gold_set = set(tuple(row) for row in gold_results)
        return pred_set == gold_set
    except (TypeError, ValueError):
        # Unhashable types — fall back to sorted comparison
        try:
            return sorted(pred_results) == sorted(gold_results)
        except TypeError:
            return pred_results == gold_results


def evaluate_single(
    pred_sql: str,
    gold_sql: str,
    db_path: str | Path,
) -> dict:
    """
    Evaluate a single predicted SQL against the gold SQL.

    Returns dict with:
        - exact_match: bool
        - execution_match: bool
        - pred_error: Optional[str]
        - gold_error: Optional[str]
        - pred_time_ms: float
        - gold_time_ms: float
    """
    # Exact match (normalized)
    exact_match = normalize_sql(pred_sql) == normalize_sql(gold_sql)

    # Execution match
    pred_results, pred_error, pred_time = execute_sql(pred_sql, db_path)
    gold_results, gold_error, gold_time = execute_sql(gold_sql, db_path)

    execution_match = False
    if pred_error is None and gold_error is None:
        execution_match = compare_results(pred_results, gold_results)

    return {
        "exact_match": exact_match,
        "execution_match": execution_match,
        "pred_error": pred_error,
        "gold_error": gold_error,
        "pred_time_ms": pred_time,
        "gold_time_ms": gold_time,
        "pred_sql_normalized": normalize_sql(pred_sql),
        "gold_sql_normalized": normalize_sql(gold_sql),
    }


def find_database_path(db_id: str, databases_dir: str | Path) -> Optional[Path]:
    """Find the SQLite database file for a given db_id."""
    db_dir = Path(databases_dir)

    # Common Spider layout: databases/{db_id}/{db_id}.sqlite
    candidates = [
        db_dir / db_id / f"{db_id}.sqlite",
        db_dir / db_id / f"{db_id}.db",
        db_dir / f"{db_id}.sqlite",
        db_dir / f"{db_id}.db",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None
