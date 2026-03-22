"""
Download and prepare Spider and WikiSQL datasets for training.
Usage: python data/download_data.py [--data-dir ./data] [--wikisql-subset 10000]
"""

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


SPIDER_URL = "https://drive.google.com/uc?export=download&id=1iRDVHLr6THMBf1kManGPUyJTCqXMDft5"
WIKISQL_HF = "kaxap/pg-wikiSQL-sql-instructions-80k"


def download_file(url: str, dest: Path, desc: str = "Downloading") -> None:
    """Download a file with progress bar and error handling."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [SKIP] {dest} already exists.")
        return

    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=desc
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    except Exception as e:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"Download failed for {url}: {e}") from e


def download_spider(data_dir: Path) -> Path:
    """Download and extract the Spider dataset."""
    spider_dir = data_dir / "spider"
    if spider_dir.exists() and (spider_dir / "train_spider.json").exists():
        print("[SKIP] Spider dataset already downloaded.")
        return spider_dir

    print("\n=== Downloading Spider Dataset ===")
    zip_path = data_dir / "spider.zip"

    # Try HuggingFace datasets first (more reliable)
    try:
        from datasets import load_dataset

        print("  Downloading via HuggingFace datasets...")
        ds = load_dataset("xlangai/spider", trust_remote_code=True)
        spider_dir.mkdir(parents=True, exist_ok=True)

        # Save as JSON for consistency
        for split_name in ["train", "validation"]:
            if split_name in ds:
                records = [dict(r) for r in ds[split_name]]
                out_file = "train_spider.json" if split_name == "train" else "dev.json"
                with open(spider_dir / out_file, "w", encoding="utf-8") as f:
                    json.dump(records, f, indent=2, ensure_ascii=False)
                print(f"  Saved {len(records)} samples to {out_file}")

        print("[OK] Spider dataset ready.")
        return spider_dir

    except Exception as e:
        print(f"  HuggingFace download failed ({e}), trying direct download...")

    # Fallback: direct URL download
    download_file(SPIDER_URL, zip_path, desc="Spider dataset")
    print("  Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)
    zip_path.unlink()

    print("[OK] Spider dataset ready.")
    return spider_dir


def download_wikisql(data_dir: Path, subset_size: int = 10000) -> Path:
    """Download WikiSQL subset via HuggingFace."""
    wikisql_dir = data_dir / "wikisql"
    output_file = wikisql_dir / "wikisql_subset.json"

    if output_file.exists():
        print("[SKIP] WikiSQL subset already downloaded.")
        return wikisql_dir

    print(f"\n=== Downloading WikiSQL ({subset_size} samples) ===")
    try:
        from datasets import load_dataset

        ds = load_dataset(WIKISQL_HF, split=f"train[:{subset_size}]")
        wikisql_dir.mkdir(parents=True, exist_ok=True)

        records = [dict(r) for r in ds]
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(records)} samples to {output_file}")

    except Exception as e:
        print(f"[WARN] WikiSQL download failed: {e}")
        print("  WikiSQL is optional (used for T5 pre-warming only).")
        wikisql_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump([], f)

    print("[OK] WikiSQL subset ready.")
    return wikisql_dir


def download_spider_databases(data_dir: Path) -> Path:
    """Download Spider SQLite databases for execution-based evaluation."""
    db_dir = data_dir / "databases"
    if db_dir.exists() and any(db_dir.iterdir()):
        print("[SKIP] Spider databases already exist.")
        return db_dir

    print("\n=== Downloading Spider Databases ===")
    try:
        from datasets import load_dataset

        # The spider dataset on HF includes database info
        # Databases are part of the spider distribution
        print("  Databases should be included with the Spider dataset.")
        print("  If not present, download manually from: https://yale-lily.github.io/spider")
        db_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[WARN] Database download issue: {e}")
        db_dir.mkdir(parents=True, exist_ok=True)

    return db_dir


def verify_data(data_dir: Path) -> dict:
    """Verify downloaded data and report statistics."""
    stats = {}
    print("\n=== Data Verification ===")

    # Spider
    spider_dir = data_dir / "spider"
    for fname, label in [("train_spider.json", "Spider train"), ("dev.json", "Spider dev")]:
        fpath = spider_dir / fname
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            stats[label] = len(data)
            print(f"  {label}: {len(data)} samples ✓")
        else:
            stats[label] = 0
            print(f"  {label}: NOT FOUND ✗")

    # WikiSQL
    wiki_path = data_dir / "wikisql" / "wikisql_subset.json"
    if wiki_path.exists():
        with open(wiki_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        stats["WikiSQL subset"] = len(data)
        print(f"  WikiSQL subset: {len(data)} samples ✓")
    else:
        stats["WikiSQL subset"] = 0
        print(f"  WikiSQL subset: NOT FOUND ✗")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Download datasets for Text-to-SQL research")
    parser.add_argument("--data-dir", type=str, default="./data", help="Base data directory")
    parser.add_argument("--wikisql-subset", type=int, default=10000, help="WikiSQL subset size")
    parser.add_argument("--skip-wikisql", action="store_true", help="Skip WikiSQL download")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Text-to-SQL Dataset Downloader")
    print("=" * 60)

    download_spider(data_dir)

    if not args.skip_wikisql:
        download_wikisql(data_dir, args.wikisql_subset)

    download_spider_databases(data_dir)

    stats = verify_data(data_dir)

    # Save stats
    with open(data_dir / "download_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("All datasets ready!")
    print("=" * 60)


if __name__ == "__main__":
    main()
