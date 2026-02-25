import os
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

# Import shared utilities
from src.utils import setup_logging, load_jsonl, save_jsonl, logging

# Initialize standard logging
setup_logging()

# Use Hydra to manage configurations and execution
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Prepares and splits the dataset into training and validation sets."""
    
    # Determine the output directory and size limit based on the execution mode
    if cfg.prep.mode == "full":
        output_dir = os.path.join(cfg.files.pro_dir, "full_revised")
        prototype_size = None
        logging.info("Starting FULL data preparation...")
    else:
        output_dir = os.path.join(cfg.files.pro_dir, "smaller_dataset")
        prototype_size = cfg.prep.prototype_size
        logging.info("Starting PROTOTYPE data preparation...")

    # Load the raw dataset and exit if it is empty
    df_raw = load_jsonl(cfg.files.eli5_rewritten_jsonl)
    if df_raw.empty: 
        logging.error("No data loaded. Execution terminated.")
        return
    
    total_records = len(df_raw)
    logging.info(f"Loaded {total_records} total records.")

    # Attempt to create a stratified sample based on subject area and complexity
    try:
        stratify_key = df_raw['subject_area'] + "_" + df_raw.get('complexity', '')
        
        if cfg.prep.mode == "prototype":
            # Downsample the data for a prototype run while maintaining category distribution
            target_size = min(prototype_size, total_records)
            df_working, _ = train_test_split(
                df_raw, train_size=target_size, random_state=cfg.generation.random_seed, stratify=stratify_key
            )
        else:
            df_working = df_raw

    except (ValueError, KeyError) as e:
        logging.warning(f"Stratification failed ({e}). Fallback to random sampling applied.")
        if cfg.prep.mode == "prototype":
            # Fall back to a random sample if stratification fails
            df_working = df_raw.sample(n=min(prototype_size, total_records), random_state=cfg.generation.random_seed)
        else:
            df_working = df_raw

    # Split the working dataset into final training and validation sets
    try:
        final_stratify = df_working['subject_area'] + "_" + df_working.get('complexity', '')
        df_train, df_val = train_test_split(
            df_working, test_size=cfg.prep.test_size, random_state=cfg.generation.random_seed, stratify=final_stratify
        )
    except (ValueError, KeyError):
        logging.warning("Final split stratification failed. Fallback to random split applied.")
        # Fall back to a basic random split if the final stratification fails
        df_train, df_val = train_test_split(
            df_working, test_size=cfg.prep.test_size, random_state=cfg.generation.random_seed
        )

    # Save the resulting datasets to the designated output directory
    save_jsonl(df_train, os.path.join(output_dir, "train.jsonl"))
    save_jsonl(df_val, os.path.join(output_dir, "validation.jsonl"))
    
    logging.info("Data preparation complete.")
    logging.info(f"Training Samples: {len(df_train)}")
    logging.info(f"Validation Samples: {len(df_val)}")
    logging.info(f"Data saved to: {output_dir}")

if __name__ == "__main__":
    main()