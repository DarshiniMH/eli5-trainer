import pandas as pd
import logging
import os
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_JSONL = "data/02_generated/eli5_dataset_raw.jsonl"
OUTPUT_REVIEW_CSV = "data/03_curated/review_sample.csv"

SAMPLE_RATE = 0.1

SAFETY_CATEGORY = "Safety/Refusal"
RANDOM_SEED = 42

def load_jsonl(filepath):
    data = []
    if not os.path.exists(filepath):
        logging.error(f"Input filepath not found: {filepath}")
        return pd.Dataframe()
    
    logging.info(f"Loading data from {filepath}...")
    try:
        with open(filepath, 'r', encoding="utf-8") as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    logging.warning("Skipping corrupted line in JSONL file.")
                    continue
    except Exception as e:
        logging.error(f"Error reading JSONL file: {e}")
        return pd.Dataframe()
    return pd.DataFrame(data)

def main():
    logging.info("Starting data curation sampling process...")

    df_raw = load_jsonl(INPUT_JSONL)
    if df_raw.empty:
        return
    
    logging.info(f"Loaded {len(df_raw)} total examples.")

    if "subject_area" not in df_raw.columns:
        logging.error("Error: 'subject_area' column not found. Cannot perform stratified sampling.")
        return
    
    def sample_group(group):
        if group.name == SAFETY_CATEGORY:
            return group
        else:
            n_samples = int(round(len(group) * SAMPLE_RATE))
            if n_samples == 0 and len(group) > 0:
                n_samples = 1

        return group.sample(n = n_samples, random_state = RANDOM_SEED)
    
    logging.info("Performing stratified sampling...")
    df_samples = df_raw.groupby("subject_area", group_keys = False).apply(sample_group)

    df_samples = df_samples.copy()
    df_samples["review_status"] = "Pending"

    if SAFETY_CATEGORY in df_samples["subject_area"].unique():
        df_samples.loc[df_samples["subject_area"] == SAFETY_CATEGORY, "review_status"] = "Pending-safety"

    df_sample = df_samples.sample(frac = 1, random_state = RANDOM_SEED).reset_index(drop = True)

    logging.info(f"\n--- Sampling Complete ---")
    logging.info(f"Total records selected for review: {len(df_sample)}")
    logging.info(f"Percentage of total dataset: {len(df_sample)/len(df_raw)*100:.2f}%")

    if SAFETY_CATEGORY in df_raw['subject_area'].unique():
        safety_total = len(df_raw[df_raw['subject_area'] == SAFETY_CATEGORY])
        safety_reviewed = len(df_sample[df_sample['subject_area'] == SAFETY_CATEGORY])
        logging.info(f"Safety/Refusal records: {safety_reviewed}/{safety_total} (100% coverage)")

    if OUTPUT_REVIEW_CSV:
        os.makedirs(os.path.dirname(OUTPUT_REVIEW_CSV), exist_ok=True)
    
    # Select and reorder columns relevant for review
    columns_for_review = [
        'review_status', 'subject_area', 'complexity', 'input', 'output', 
        'teacher_reflection', 'original_index', 'generating_model'
    ]
    # Ensure all selected columns exist before saving
    final_columns = [col for col in columns_for_review if col in df_sample.columns]
    
    # Save as CSV for easy review in Excel, Google Sheets, or VS Code
    df_sample[final_columns].to_csv(OUTPUT_REVIEW_CSV, index=False)
    logging.info(f"\nSaved review sample to {OUTPUT_REVIEW_CSV}")

if __name__ == "__main__":
    main()