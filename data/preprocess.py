"""
Schema linearization and data preprocessing for Spider/WikiSQL.
Converts raw dataset records into model-ready format.
"""

import json
import re
from pathlib import Path
from typing import Optional


def linearize_schema(tables: dict, db_id: str) -> str:
    """
    Convert Spider schema dict into a compact linearized string.

    Format:
        Database: {db_id}
        Tables:
        - table_name (col1 TYPE PK, col2 TYPE, col3 TYPE FK→other_table)
    """
    if not tables or db_id not in _build_schema_lookup(tables):
        return f"Database: {db_id}\nTables: (schema unavailable)"

    schema = _build_schema_lookup(tables)[db_id]
    lines = [f"Database: {db_id}", "Tables:"]

    for table_name, columns in schema["tables"].items():
        col_strs = []
        for col in columns:
            parts = [col["name"], col["type"]]
            if col.get("is_primary"):
                parts.append("PK")
            if col.get("fk_ref"):
                parts.append(f"FK→{col['fk_ref']}")
            col_strs.append(" ".join(parts))
        lines.append(f"- {table_name} ({', '.join(col_strs)})")

    return "\n".join(lines)


def _build_schema_lookup(tables_data: dict) -> dict:
    """Build a lookup dict from Spider tables.json format."""
    lookup = {}

    if isinstance(tables_data, list):
        for db in tables_data:
            db_id = db["db_id"]
            schema = {"tables": {}}

            table_names = db.get("table_names_original", db.get("table_names", []))
            col_names = db.get("column_names_original", db.get("column_names", []))
            col_types = db.get("column_types", [])
            pk_ids = set(db.get("primary_keys", []))
            fk_pairs = db.get("foreign_keys", [])

            # Build FK lookup: col_id → referenced_table
            fk_lookup = {}
            for fk_col, ref_col in fk_pairs:
                if ref_col < len(col_names):
                    ref_table_idx = col_names[ref_col][0]
                    if ref_table_idx < len(table_names):
                        fk_lookup[fk_col] = table_names[ref_table_idx]

            for i, tname in enumerate(table_names):
                cols = []
                for col_idx, (table_idx, col_name) in enumerate(col_names):
                    if table_idx == i:
                        col_info = {
                            "name": col_name,
                            "type": col_types[col_idx] if col_idx < len(col_types) else "TEXT",
                            "is_primary": col_idx in pk_ids,
                            "fk_ref": fk_lookup.get(col_idx),
                        }
                        cols.append(col_info)
                schema["tables"][tname] = cols

            lookup[db_id] = schema

    return lookup


def linearize_schema_from_db_id(db_id: str, tables_file: Path) -> str:
    """Load tables.json and linearize schema for a given db_id."""
    with open(tables_file, "r", encoding="utf-8") as f:
        tables_data = json.load(f)
    return linearize_schema(tables_data, db_id)


def preprocess_spider_sample(
    sample: dict,
    schema_lookup: dict,
) -> dict:
    """
    Preprocess a single Spider sample into a standard format.

    Returns:
        {
            "db_id": str,
            "question": str,
            "query": str (gold SQL),
            "schema": str (linearized),
            "difficulty": str,
        }
    """
    db_id = sample.get("db_id", "unknown")
    question = sample.get("question", "")
    query = sample.get("query", "")
    difficulty = sample.get("difficulty", sample.get("hardness", "unknown"))

    # Linearize schema
    schema_str = ""
    if db_id in schema_lookup:
        db_info = schema_lookup[db_id]
        lines = [f"Database: {db_id}", "Tables:"]
        for table_name, columns in db_info["tables"].items():
            col_strs = []
            for col in columns:
                parts = [col["name"], col["type"]]
                if col.get("is_primary"):
                    parts.append("PK")
                if col.get("fk_ref"):
                    parts.append(f"FK→{col['fk_ref']}")
                col_strs.append(" ".join(parts))
            lines.append(f"- {table_name} ({', '.join(col_strs)})")
        schema_str = "\n".join(lines)
    else:
        schema_str = f"Database: {db_id}\nTables: (schema unavailable)"

    return {
        "db_id": db_id,
        "question": question.strip(),
        "query": query.strip(),
        "schema": schema_str,
        "difficulty": difficulty,
    }


def preprocess_wikisql_sample(sample: dict) -> dict:
    """Preprocess a WikiSQL sample into the standard format."""
    return {
        "db_id": "wikisql",
        "question": sample.get("input", sample.get("question", "")).strip(),
        "query": sample.get("output", sample.get("sql", sample.get("query", ""))).strip(),
        "schema": sample.get("instruction", ""),
        "difficulty": "simple",
    }


def load_and_preprocess_spider(data_dir: Path, split: str = "train") -> list[dict]:
    """Load and preprocess an entire Spider split."""
    spider_dir = data_dir / "spider"
    fname = "train_spider.json" if split == "train" else "dev.json"
    data_file = spider_dir / fname
    tables_file = spider_dir / "tables.json"

    if not data_file.exists():
        raise FileNotFoundError(f"Spider data not found at {data_file}. Run download_data.py first.")

    with open(data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Build schema lookup
    schema_lookup = {}
    if tables_file.exists():
        with open(tables_file, "r", encoding="utf-8") as f:
            tables_data = json.load(f)
        schema_lookup = _build_schema_lookup(tables_data)

    processed = []
    for sample in raw_data:
        processed.append(preprocess_spider_sample(sample, schema_lookup))

    return processed


def load_and_preprocess_wikisql(data_dir: Path, max_samples: int = 10000) -> list[dict]:
    """Load and preprocess WikiSQL subset."""
    wiki_file = data_dir / "wikisql" / "wikisql_subset.json"

    if not wiki_file.exists():
        print("[WARN] WikiSQL not found. Run download_data.py first.")
        return []

    with open(wiki_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    processed = []
    for sample in raw_data[:max_samples]:
        processed.append(preprocess_wikisql_sample(sample))

    return processed
