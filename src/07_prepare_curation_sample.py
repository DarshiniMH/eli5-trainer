import pandas as pd
import os
import json

# Import centralized configurations and shared utilities
from src.config import ELI5_REWRITTEN_JSONL, CURATION_SAMPLE_CSV
from src.utils import setup_logging, ensure_dir, load_jsonl, logging

# -----------------------------
# 1) CONFIGURATION
# -----------------------------
setup_logging() # Replaces standard basicConfig

# Sampling ratio for general categories
SAMPLE_RATE = 0.1

# Specific category requiring comprehensive audit
SAFETY_CATEGORY = "Safety/Refusal"
RANDOM_SEED = 42

# -----------------------------
# 2) CORE EXECUTION LOGIC
# -----------------------------
def main():
    """Executes stratified sampling to create a representative subset for manual review."""
    logging.info("Starting data curation sampling process...")

    # Utilize centralized utility and config for loading
    df_raw = load_jsonl(ELI5_REWRITTEN_JSONL)
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
    # Original sampling logic strictly preserved
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

    # Storage utilizing config path and utility directory management
    ensure_dir(CURATION_SAMPLE_CSV)
    
    # Column selection relevant for manual pedagogical and safety auditing preserved
    columns_for_review = [
        'review_status', 'subject_area', 'complexity', 'input', 'output', 
        'teacher_reflection', 'original_index', 'generating_model'
    ]
    final_columns = [col for col in columns_for_review if col in df_sample.columns]
    
    df_sample[final_columns].to_csv(CURATION_SAMPLE_CSV, index=False)
    logging.info(f"Review sample persisted to {CURATION_SAMPLE_CSV}")

if __name__ == "__main__":
    main()