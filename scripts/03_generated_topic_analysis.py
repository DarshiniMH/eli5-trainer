import pandas as pd
import hydra
from omegaconf import DictConfig

# Import centralized shared utilities only (configs are now handled by Hydra)
from src.utils import setup_logging, logging

# -----------------------------
# 1) LOGGING AND DATA LOADING
# -----------------------------
setup_logging() # Standardized logging format

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Performs comparative analysis between GPT-4o and GPT-4o-mini topic sets."""
    try:
        # Utilizing paths directly from the Hydra config object
        df_mini = pd.read_csv(cfg.files.topic_simple_mini_csv)
        df_4o = pd.read_csv(cfg.files.topic_simple_4o_csv)
        logging.info("Successfully loaded topic datasets for analysis.")

    except FileNotFoundError as e:
        logging.error(f"Error: CSV file not found. Ensure generation scripts were successful: {e}")
        return

    # -----------------------------
    # 2) OVERLAP ANALYSIS
    # -----------------------------
    # Original shape reporting logic preserved
    print(f"Mini shape: {df_mini.shape}")
    print(f"4o shape:{df_4o.shape}\n")

    set_mini = set(df_mini['question'])
    set_4o = set(df_4o['question'])

    # Standard set mathematics for overlap detection
    overlap = set_mini.intersection(set_4o)
    overlap_count = len(overlap)
    union_count = len(set_mini.union(set_4o))

    print(f"Total Unique Questions (if combined): {union_count}")
    print(f"Number of Identical Questions (Overlap): {overlap_count}")
    print(f"Overlap Percentage (vs Mini): {overlap_count / len(set_mini) * 100:.2f}%")

    # -----------------------------
    # 3) SAMPLE COMPARISON UTILITY
    # -----------------------------
    def compare_samples(subject, num_samples=10):
        """Original stratified sampling and comparison logic preserved."""
        print(f"\n---- comparing samples: {subject}-----\n")

        try:
            # Deterministic sampling using the random_seed from Hydra config
            seed = cfg.generation.random_seed
            samples_4o = df_4o[df_4o["subject_area"] == subject].sample(n=num_samples, random_state=seed)
            samples_mini = df_mini[df_mini["subject_area"] == subject].sample(n=num_samples, random_state=seed)
        except ValueError as e:
            print(f"Not enough samples for subject {subject}. Skipping sample comparison.")
            samples_4o = df_4o[df_4o["subject_area"] == subject]
            samples_mini = df_mini[df_mini["subject_area"] == subject]
        
        comparison_df = pd.DataFrame({
            "GPT-4o-mini": samples_mini["question"].reset_index(drop=True),
            "GPT-4o": samples_4o["question"].reset_index(drop=True)
        })

        pd.set_option('display.max_colwidth', None)
        print(comparison_df)

    # Execute comparisons for original subject areas
    compare_samples("Math/Logic")
    compare_samples("Physics")
    compare_samples("Genetics")
    compare_samples("Safety/Refusal")

if __name__ == "__main__":
    main()