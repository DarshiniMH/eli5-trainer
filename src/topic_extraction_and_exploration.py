import pandas as pd
from datasets import load_dataset
import  logging
import os

logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_mmlu_questions(num_questions=2500):
    logging.info("Starting MMLU extraction")

    stem_subjects = [
        'high_school_biology', 'high_school_chemistry', 'high_school_physics',
        'high_school_computer_science', 'astronomy', 'college_mathematics'
    ]
    humanities_subjects = [
        'high_school_macroeconomics', 'high_school_psychology', 
        'high_school_world_history', 'philosophy', 'sociology',
        'global_facts'
    ]
    professional_subjects = [
        'professional_law', 'moral_scenarios', 'management', 'professional_medicine'
    ]

    all_subjects = stem_subjects + humanities_subjects + professional_subjects
    all_questions = []

    for subject in all_subjects:
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test")

            for item in dataset:
                question_text = item.get("question")
                if question_text:
                    if subject in stem_subjects:
                        domain = "STEM (Complex)"
                    elif subject in humanities_subjects:
                        domain = "Humanities/Social Sciences (Complex)"
                    else:
                        domain = "Professional/Other (Complex)"
                
                all_questions.append({
                    "domain": domain,
                    "subject_area": subject,
                    "question": question_text.strip()
                })
        except Exception as e:
            logging.error(f"Error loading dataset for subject {subject}: {e}")

    unique_questions_df = pd.DataFrame(all_questions).drop_duplicates(subset=["question"])

    if(len(unique_questions_df) > num_questions):
        final_df = unique_questions_df.sample(n=num_questions, random_state = 42)
    else:
        final_df = unique_questions_df
        logging.warning(f"Extracted {len(final_df)} unique questions, less than the target {num_questions}.")

    return final_df

def main():
    TARGET_QUESTIONS = 2500
    OUTPUT_DIR = "data/01_raw"
    OUTPUT_FILE = "master_topic_list_complex.csv"

    complex_topics_df = extract_mmlu_questions(num_questions=TARGET_QUESTIONS)

    if complex_topics_df.empty:
        logging.info("Extraction failed. Exiting.")
        return
    
    logging.info(f"\n--- Extraction Complete ---")
    logging.info(f"Successfully extracted {len(complex_topics_df)} complex topics.")
    logging.info("\nDistribution Summary:")
    print(complex_topics_df['subject_area'].value_counts())

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    complex_topics_df.to_csv(output_path, index=False)
    logging.info(f"Saved extracted topics to {output_path}")

if __name__ == "__main__":
    main()
