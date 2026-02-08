import argparse
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
import logging

# Configuration for system logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------
# 1) CONFIGURATION AND PATHS
# -----------------------------
INPUT_JSONL = "data/02_generated/eli5_dataset_rewritten_complex.jsonl"
RANDOM_SEED = 42
TEST_SIZE = 0.1  # 10% allocation for validation

# -----------------------------
# 2) FILE UTILITIES
# -----------------------------
def load_jsonl(file_path):
    """Loads records from a JSONL file into a pandas DataFrame."""
    data = []
    if not os.path.exists(file_path):
        logging.error(f"Input file not found: {file_path}")
        return pd.DataFrame()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logging.error(f"Error reading JSONL file: {e}")
        return pd.DataFrame()
    return pd.DataFrame(data)

def save_jsonl(df, filepath):
    """Persists specified columns of a DataFrame to a JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Preservation of 'input' and 'output' columns only
    df[['input', 'output']].to_json(filepath, orient='records', lines=True)
    logging.info(f"Saved {len(df)} records to {filepath}")

# -----------------------------
# 3) CORE EXECUTION LOGIC
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Consolidated Data Preparation Tool")
    parser.add_argument("--mode", choices=["full", "prototype"], required=True, help="Execution mode")
    args = parser.parse_args()

    # Mode-specific directory and size configuration
    if args.mode == "full":
        output_dir = "data/04_processed/full_revised"
        prototype_size = None
        logging.info("Starting FULL data preparation...")
    else:
        output_dir = "data/04_processed/smaller_dataset"
        prototype_size = 5000
        logging.info("Starting PROTOTYPE data preparation...")

    # Data loading phase
    df_raw = load_jsonl(INPUT_JSONL)
    if df_raw.empty: 
        logging.error("No data loaded. Execution terminated.")
        return
    
    total_records = len(df_raw)
    logging.info(f"Loaded {total_records} total records.")

    # Implementation of stratified sampling logic
    try:
        stratify_key = df_raw['subject_area'] + "_" + df_raw.get('complexity', '')
        
        if args.mode == "prototype":
            # Downsampling for prototype mode using stratified split
            target_size = min(prototype_size, total_records)
            df_working, _ = train_test_split(
                df_raw, train_size=target_size, random_state=RANDOM_SEED, stratify=stratify_key
            )
        else:
            # Utilization of full dataset for full mode
            df_working = df_raw

    except (ValueError, KeyError) as e:
        logging.warning(f"Stratification failed ({e}). Fallback to random sampling applied.")
        if args.mode == "prototype":
            df_working = df_raw.sample(n=min(prototype_size, total_records), random_state=RANDOM_SEED)
        else:
            df_working = df_raw

    # Final split into training and validation sets
    # Stratification applied to the final split if possible
    try:
        final_stratify = df_working['subject_area'] + "_" + df_working.get('complexity', '')
        df_train, df_val = train_test_split(
            df_working, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=final_stratify
        )
    except (ValueError, KeyError):
        logging.warning("Final split stratification failed. Fallback to random split applied.")
        df_train, df_val = train_test_split(
            df_working, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )

    # Persistence of processed datasets
    save_jsonl(df_train, os.path.join(output_dir, "train.jsonl"))
    save_jsonl(df_val, os.path.join(output_dir, "validation.jsonl"))
    
    logging.info(f"Data preparation complete.")
    logging.info(f"Training Samples: {len(df_train)}")
    logging.info(f"Validation Samples: {len(df_val)}")
    logging.info(f"Data saved to: {output_dir}")

if __name__ == "__main__":
    main()