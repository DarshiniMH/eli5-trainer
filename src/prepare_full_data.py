import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION ---
INPUT_JSONL = "data/02_generated/eli5_dataset_raw.jsonl"
OUTPUT_DIR = "data/04_processed/full"  # Changed folder name to 'full'
RANDOM_SEED = 42
TEST_SIZE = 0.1  # 10% for validation (approx 800 examples)

def load_jsonl(file_path):
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
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Ensure we only save the columns we need
    df[['input', 'output']].to_json(filepath, orient='records', lines=True)
    logging.info(f"Saved {len(df)} records to {filepath}")

def main():
    logging.info("Starting FULL data preparation...")
    
    # 1. Load the Data
    df_raw = load_jsonl(INPUT_JSONL)
    if df_raw.empty: 
        logging.error("No data loaded. Exiting.")
        return
    
    total_records = len(df_raw)
    logging.info(f"Loaded {total_records} total records.")

    # 2. Stratified Split (Train vs Validation)
    # We try to stratify by subject & complexity to ensure the val set is representative
    try:
        stratify_key = df_raw['subject_area'] + "_" + df_raw.get('complexity', '')
        
        # NOTE: We split df_raw directly now (no downsampling step)
        df_train, df_val = train_test_split(
            df_raw, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_SEED, 
            stratify=stratify_key
        )
    except (ValueError, KeyError) as e:
        logging.warning(f"Stratification failed ({e}). Falling back to random sampling.")
        df_train, df_val = train_test_split(
            df_raw, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_SEED
        )

    # 3. Save the Splits
    save_jsonl(df_train, os.path.join(OUTPUT_DIR, "train.jsonl"))
    save_jsonl(df_val, os.path.join(OUTPUT_DIR, "validation.jsonl"))
    
    logging.info(f"Full data preparation complete.")
    logging.info(f"Training Samples: {len(df_train)}")
    logging.info(f"Validation Samples: {len(df_val)}")
    logging.info(f"Data saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()