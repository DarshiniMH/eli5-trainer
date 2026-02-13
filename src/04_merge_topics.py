import pandas as pd
import os
# Import centralized configurations and shared utilities
from src.config import (
    TOPIC_SIMPLE_MINI_CSV, 
    TOPIC_SIMPLE_4O_CSV, 
    TOPIC_COMPLEX_CSV, 
    MERGED_TOPICS_CSV
)
from src.utils import setup_logging, ensure_dir, logging

# -----------------------------
# 1) LOGGING INITIALIZATION
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
    
def main():
    """Main execution loop for dataset merging and consolidating."""
    logging.info("Starting dataset merging and consolidating...")

    # Utilize centralized paths from config.py
    df_mini = load_csv(TOPIC_SIMPLE_MINI_CSV)
    df_4o = load_csv(TOPIC_SIMPLE_4O_CSV)
    df_complex = load_csv(TOPIC_COMPLEX_CSV)

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
    
    # Original deduplication and shuffling logic preserved
    df_final = combined_df.drop_duplicates(subset=["question"], keep='first')
    duplicates_removed = initial_count - len(df_final)

    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    logging.info(f"\n--- Merging Complete ---")
    logging.info(f"Total unique inputs (Simple + Complex): {len(df_final)}")
    logging.info(f"Total duplicates removed: {duplicates_removed}")
    logging.info("\nFinal Source Distribution Summary:")
    print(df_final['generating_model'].value_counts())

    # Persistent storage using config and utility path management
    ensure_dir(MERGED_TOPICS_CSV)
    df_final.to_csv(MERGED_TOPICS_CSV, index=False)
    logging.info(f"Merged dataset saved to {MERGED_TOPICS_CSV}")

if __name__ == "__main__":
    main()