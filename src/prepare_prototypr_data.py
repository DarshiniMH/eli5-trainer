import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
import logging


logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
INPUT_JSONL = "data/02_generated/eli5_dataset_raw.jsonl"
OUTPUT_DIR = "data/04_processed/prototype"
PROTOTYPE_SIZE = 1000
RANDOM_SEED = 42

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
    df[['input', 'output']].to_json(filepath, orient='records', lines=True)
    logging.info(f"Saved {len(df)} records to {filepath}")

def main():
    logging.info("Starting prototype data preparation...")
    df_raw = load_jsonl(INPUT_JSONL)
    if df_raw.empty: 
        logging.error("No data loaded. Exiting.")
        return
    
    target_size = min(PROTOTYPE_SIZE, len(df_raw))

    try:
        stratify_key = df_raw['subject_area'] + "_" + df_raw.get('complexity', '')

        df_prototype, _ = train_test_split(
            df_raw, train_size = target_size, random_state = RANDOM_SEED, stratify = stratify_key
        )
    except (ValueError, KeyError):
        logging.warning("Stratification failed (e.g., too few samples in a category). Falling back to random sampling.")
        df_prototype = df_raw.sample(n=target_size, random_state = RANDOM_SEED)
    
    df_train, df_val =  train_test_split(df_prototype, test_size = 0.1, random_state = RANDOM_SEED)

    save_jsonl(df_train, os.path.join(OUTPUT_DIR, "train.jsonl"))
    save_jsonl(df_val, os.path.join(OUTPUT_DIR, "validation.jsonl"))
    
    logging.info(f"Prototype data preparation complete. Data saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()


    
