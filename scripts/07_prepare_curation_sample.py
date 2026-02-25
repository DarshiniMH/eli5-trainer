import pandas as pd
import os
import json
import hydra
from omegaconf import DictConfig

# Import shared utilities
from src.utils import setup_logging, ensure_dir, load_jsonl, logging

# Initialize standard logging
setup_logging() 

# Use Hydra to manage configurations
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Executes stratified sampling to create a representative subset for manual review."""
    logging.info("Starting data curation sampling process...")

    # Extract file paths from the config
    input_file = cfg.files.eli5_rewritten_jsonl
    output_file = cfg.files.curation_sample_csv
    
    # Load the dataset and exit early if it is empty
    df_raw = load_jsonl(input_file)
    if df_raw.empty:
        return
    
    logging.info(f"Loaded {len(df_raw)} total examples.")

    # Verifying the subject area column exists before grouping
    if "subject_area" not in df_raw.columns:
        logging.error("Column 'subject_area' not found. Stratified sampling cannot proceed.")
        return
    
    def sample_group(group):
        # Retain all safety topics; otherwise, take a proportional sample
        if group.name == cfg.curation.safety_category:
            return group
        else:
            n_samples = int(round(len(group) * cfg.curation.sample_rate))
            # Ensure at least one sample is taken from non-empty groups
            if n_samples == 0 and len(group) > 0:
                n_samples = 1
        return group.sample(n=n_samples, random_state=cfg.generation.random_seed)
    
    logging.info("Performing stratified sampling by subject area...")
    
    # Group the data by subject and apply the sampling logic
    df_samples = df_raw.groupby("subject_area", group_keys=False).apply(sample_group)

    # Set the default review status
    df_samples = df_samples.copy()
    df_samples["review_status"] = "Pending"

    # Flag safety-critical topics specifically for the review queue
    if cfg.curation.safety_category in df_samples["subject_area"].unique():
        df_samples.loc[df_samples["subject_area"] == cfg.curation.safety_category, "review_status"] = "Pending-safety"

    # Shuffle the dataset to mix up topics for reviewers
    df_sample = df_samples.sample(frac=1, random_state=cfg.generation.random_seed).reset_index(drop=True)

    logging.info(f"Sampling complete. {len(df_sample)} records selected ({len(df_sample)/len(df_raw)*100:.2f}% coverage).")

    ensure_dir(output_file)
    
    # Filter down to the specific columns needed for auditing
    columns_for_review = [
        'review_status', 'subject_area', 'complexity', 'input', 'output', 
        'teacher_reflection', 'original_index', 'generating_model'
    ]
    final_columns = [col for col in columns_for_review if col in df_sample.columns]
    
    # Save the finalized sample to a CSV file
    df_sample[final_columns].to_csv(output_file, index=False)
    logging.info(f"Review sample persisted to {output_file}")

if __name__ == "__main__":
    main()