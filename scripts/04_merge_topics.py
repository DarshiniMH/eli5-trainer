import pandas as pd
import os
import hydra
from omegaconf import DictConfig

# Import centralized shared utilities only
from src.utils import setup_logging, ensure_dir, logging

# -----------------------------
# LOGGING INITIALIZATION
# -----------------------------
setup_logging() # Standardized logging format

def load_csv(file_path):
    """Loads a CSV file with original error handling logic preserved."""
    if os.path.exists(file_path):
        logging.info(f"Loading {os.path.basename(file_path)}...")
        return pd.read_csv(file_path, on_bad_lines='skip')
    else:
        logging.warning(f"File not found: {file_path}. Skipping.")
        return pd.DataFrame()
    
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main execution loop for dataset merging and consolidating."""
    logging.info("Starting dataset merging and consolidating...")

    # Utilize centralized paths directly from the Hydra config object
    df_mini = load_csv(cfg.files.topic_simple_mini_csv)
    df_4o = load_csv(cfg.files.topic_simple_4o_csv)
    df_complex = load_csv(cfg.files.topic_complex_csv)

    if df_mini.empty and df_4o.empty:
        logging.error("Error: Simple question datasets (Mini and 4o) not found or empty.")
        return
    
    # Original metadata assignment logic preserved
    df_mini['generating_model'] = "GPT-4o-mini"
    df_4o['generating_model'] = "GPT-4o"
    df_complex['generating_model'] = "GPT-4o-mini"
    
    df_mini["complexity"] = "simple"
    df_4o["complexity"] = "simple"
    df_complex["complexity"] = "complex"

    if df_complex.empty:
        logging.error("Error: Complex question dataset not found or empty.")
        return

    # Consolidating datasets
    combined_df = pd.concat([df_mini, df_4o, df_complex], ignore_index=True)

    logging.info(f"simple questions combined length: {combined_df[combined_df['complexity'] == 'simple'].shape[0]}")

    if 'question' not in combined_df.columns:
        logging.error("Error: 'question' column missing in the combined dataset.")
        return
    
    initial_count = len(combined_df)
    logging.info(f"Initial combined dataset length: {initial_count}")
    
    # Original deduplication logic preserved
    df_final = combined_df.drop_duplicates(subset=["question"], keep='first')
    duplicates_removed = initial_count - len(df_final)

    # Shuffling logic updated to use the centralized random seed from Hydra
    df_final = df_final.sample(frac=1, random_state=cfg.generation.random_seed).reset_index(drop=True)

    logging.info(f"\n--- Merging Complete ---")
    logging.info(f"Total unique inputs (Simple + Complex): {len(df_final)}")
    logging.info(f"Total duplicates removed: {duplicates_removed}")
    logging.info("\nFinal Source Distribution Summary:")
    print(df_final['generating_model'].value_counts())

    # Persistent storage using Hydra configuration
    ensure_dir(cfg.files.merged_topics_csv)
    df_final.to_csv(cfg.files.merged_topics_csv, index=False)
    logging.info(f"Merged dataset saved to {cfg.files.merged_topics_csv}")

if __name__ == "__main__":
    main()