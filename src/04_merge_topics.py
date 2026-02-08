import pandas as pd
import logging
import os

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')

DATA_DIR = "../data/01_raw"

FILE_MINI = os.path.join(DATA_DIR, "master_topic_list_tagged_4o_mini.csv")
FILE_4O = os.path.join(DATA_DIR, "master_topic_list_tagged_4o.csv")
FILE_COMPLEX = os.path.join(DATA_DIR, "master_topic_list_complex_mini.csv")

OUTPUT_FILE = os.path.join(DATA_DIR, "merged_master_topic_list.csv")

def load_csv(file_path):
    if os.path.exists(file_path):
        logging.info(f"Loading {os.path.basename(file_path)}...")
        return pd.read_csv(file_path, on_bad_lines='skip')
    else:
        logging.warning(f"File not found:{file_path}. Skipping.")
        return pd.DataFrame()
    
def main():
    logging.info("Starting dataset merging and consolidating...")

    df_mini = load_csv(FILE_MINI)
    df_4o = load_csv(FILE_4O)
    df_complex = load_csv(FILE_COMPLEX)

    if df_mini.empty and df_4o.empty:
        logging.error("Error: Simple question datasets (Mini and 4o) not found or empty.")
        return
    
    df_mini['generating_model'] = "GPT-4o-mini"
    df_4o['generating_model'] = "GPT-4o"
    df_complex['generating_model'] = "GPT-4o-mini"
    df_mini["complexity"] = "simple"
    df_4o["complexity"] = "simple"
    df_complex["complexity"] = "complex"

    if df_complex.empty:
        logging.error("Error: Complex question dataset not found or empty.")
        return

    combined_df = pd.concat([df_mini, df_4o, df_complex], ignore_index = True)

    logging.info(f"simple questions combined length: {combined_df[combined_df["complexity"] == 'simple'].shape[0]}")

    if 'question' not in combined_df.columns:
        logging.error("Error: 'question' column missing in the combined dataset.")
        return
    
    initial_count = len(combined_df)
    logging.info(f"Initial combined dataset length: {initial_count}")
    df_final = combined_df.drop_duplicates(subset = ["question"], keep = 'first')
    duplicates_removed = initial_count - len(df_final)

    df_final = df_final.sample(frac = 1, random_state = 42).reset_index(drop=True)

    logging.info(f"\n--- Merging Complete ---")
    logging.info(f"Total unique inputs (Simple + Complex): {len(df_final)}")
    logging.info(f"Total duplicates removed: {duplicates_removed}")
    logging.info("\nFinal Source Distribution Summary:")
    print(df_final['generating_model'].value_counts())

    df_final.to_csv(OUTPUT_FILE, index = False)
    logging.info(f"Merged dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()