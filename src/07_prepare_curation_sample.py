import pandas as pd
import logging
import os
import json

# Configuration for system logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File path definitions for curation
INPUT_JSONL = "data/02_generated/eli5_dataset_rewritten_complex.jsonl"
OUTPUT_REVIEW_CSV = "data/03_curated/review_sample.csv"

# Sampling ratio for general categories
SAMPLE_RATE = 0.1

# Specific category requiring comprehensive audit
SAFETY_CATEGORY = "Safety/Refusal"
RANDOM_SEED = 42

def load_jsonl(filepath):
    """Loads records from a JSONL file into a structured DataFrame."""
    data = []
    if not os.path.exists(filepath):
        logging.error(f"Input filepath not found: {filepath}")
        return pd.DataFrame()
    
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
        return pd.DataFrame()
    return pd.DataFrame(data)

def main():
    """Executes stratified sampling to create a representative subset for manual review."""
    logging.info("Starting data curation sampling process...")

    df_raw = load_jsonl(INPUT_JSONL)
    if df_raw.empty:
        return
    
    logging.info(f"Loaded {len(df_raw)} total examples.")

    # Verification of subject_area column for grouping
    if "subject_area" not in df_raw.columns:
        logging.error("Column 'subject_area' not found. Stratified sampling cannot proceed.")
        return
    
    def sample_group(group):
        """Applies 100% sampling to safety topics and proportional sampling to others."""
        if group.name == SAFETY_CATEGORY:
            return group
        else:
            n_samples = int(round(len(group) * SAMPLE_RATE))
            # Ensures at least one sample is taken if the group is not empty
            if n_samples == 0 and len(group) > 0:
                n_samples = 1
        return group.sample(n=n_samples, random_state=RANDOM_SEED)
    
    logging.info("Performing stratified sampling by subject area...")
    df_samples = df_raw.groupby("subject_area", group_keys=False).apply(sample_group)

    # Assignment of review status metadata
    df_samples = df_samples.copy()
    df_samples["review_status"] = "Pending"

    # Specific status tag for safety-critical rows
    if SAFETY_CATEGORY in df_samples["subject_area"].unique():
        df_samples.loc[df_samples["subject_area"] == SAFETY_CATEGORY, "review_status"] = "Pending-safety"

    # Shuffling to ensure diverse topics during sequential review
    df_sample = df_samples.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    logging.info(f"Sampling complete. {len(df_sample)} records selected ({len(df_sample)/len(df_raw)*100:.2f}% coverage).")

    # Storage of review set to CSV for spreadsheet accessibility
    if OUTPUT_REVIEW_CSV:
        os.makedirs(os.path.dirname(OUTPUT_REVIEW_CSV), exist_ok=True)
    
    # Column selection relevant for manual pedagogical and safety auditing
    columns_for_review = [
        'review_status', 'subject_area', 'complexity', 'input', 'output', 
        'teacher_reflection', 'original_index', 'generating_model'
    ]
    final_columns = [col for col in columns_for_review if col in df_sample.columns]
    
    df_sample[final_columns].to_csv(OUTPUT_REVIEW_CSV, index=False)
    logging.info(f"Review sample persisted to {OUTPUT_REVIEW_CSV}")

if __name__ == "__main__":
    main()