import json
import os
import logging
import pandas as pd

def setup_logging():
    """Centralized logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_jsonl(filepath):
    """General utility to load JSONL files into a list or DataFrame."""
    data = []
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        return pd.DataFrame()
    with open(filepath, 'r', encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(data)

def save_jsonl(df, filepath, columns=None):
    """General utility to save DataFrames to JSONL."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if columns:
        df = df[columns]
    df.to_json(filepath, orient='records', lines=True)
    logging.info(f"Saved {len(df)} records to {filepath}")

def ensure_dir(path):
    """Ensures a directory exists for a given file path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)