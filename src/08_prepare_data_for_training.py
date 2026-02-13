import argparse
import os
from sklearn.model_selection import train_test_split

# Import centralized configurations and shared utilities
from src.config import ELI5_REWRITTEN_JSONL, PRO_DIR
from src.utils import setup_logging, load_jsonl, save_jsonl, logging

# -----------------------------
# 1) CONFIGURATION AND PATHS
# -----------------------------
setup_logging()  # Replaces standard basicConfig

# Configuration constants preserved
RANDOM_SEED = 42
TEST_SIZE = 0.1  # 10% allocation for validation

# -----------------------------
# 2) CORE EXECUTION LOGIC
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Consolidated Data Preparation Tool")
    parser.add_argument("--mode", choices=["full", "prototype"], required=True, help="Execution mode")
    args = parser.parse_args()

    # Mode-specific directory and size configuration utilizing config.py paths
    if args.mode == "full":
        # Paths now relative to centralized PRO_DIR
        output_dir = os.path.join(PRO_DIR, "full_revised")
        prototype_size = None
        logging.info("Starting FULL data preparation...")
    else:
        output_dir = os.path.join(PRO_DIR, "smaller_dataset")
        prototype_size = 5000
        logging.info("Starting PROTOTYPE data preparation...")

    # Data loading phase utilizing centralized utils and config
    df_raw = load_jsonl(ELI5_REWRITTEN_JSONL)
    if df_raw.empty: 
        logging.error("No data loaded. Execution terminated.")
        return
    
    total_records = len(df_raw)
    logging.info(f"Loaded {total_records} total records.")

    # Implementation of stratified sampling logic preserved exactly
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

    # Final split into training and validation sets preserved
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

    # Persistence utilizing centralized save utility
    save_jsonl(df_train, os.path.join(output_dir, "train.jsonl"))
    save_jsonl(df_val, os.path.join(output_dir, "validation.jsonl"))
    
    logging.info(f"Data preparation complete.")
    logging.info(f"Training Samples: {len(df_train)}")
    logging.info(f"Validation Samples: {len(df_val)}")
    logging.info(f"Data saved to: {output_dir}")

if __name__ == "__main__":
    main()