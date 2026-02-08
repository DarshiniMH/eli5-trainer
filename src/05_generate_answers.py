import openai
import pandas as pd
import time
import json
import os
import logging
from dotenv import load_dotenv
from tqdm.auto import tqdm

# Configuration for system logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialization of environment variables for API authentication
load_dotenv()

# Setup for the OpenAI client and validation of the API key
client = openai.OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    logging.error("Error: OPENAI_API_KEY is not found. Please check your .env file.")
    exit()

# Model and file path definitions
TEACHER_MODEL = "gpt-4o"
INPUT_CSV = "../data/01_raw/merged_master_topic_list.csv"
OUTPUT_JSONL = "../data/02_generated/eli5_dataset_raw.jsonl"

# System prompt defining pedagogical style, safety protocols, and JSON output format
SYSTEM_PROMPT = """You are an award-winning K-12 educator renowned for explaining complex concepts with exceptional clarity, enthusiasm, and factual accuracy. Your goal is to provide the simplest possible explanation tailored to a young audience (ages 5-12).

# 1. ADAPTIVE STYLE (ELI5/ELI12)
- Analyze the input concept/question.
- Determine the minimum complexity required to maintain factual accuracy.
- ELI5 (Ages 5-8): Use very simple vocabulary, short sentences.
- ELI12 (Ages 9-12): If the concept is inherently complex (e.g., inflation, genetics), use slightly more advanced vocabulary (avoiding jargon) and multi-step logic.
- The goal is maximum simplification without becoming incorrect.

# 2. EXPLANATION STRATEGIES (Clarity First)
- Choose the BEST strategy for clarity: Direct Logic OR Analogy.
- Strategy A (Direct Logic): Use clear definitions and step-by-step logic (e.g., 1, 2, 3...). This is preferred for mechanisms, processes, concrete facts, or simple definitions.
- Strategy B (Analogy): Use relatable analogies (toys, nature, everyday activities) ONLY if the concept is abstract or difficult to visualize AND the analogy is accurate and genuinely helpful.

# 3. TONE AND SAFETY
- General Tone: Be enthusiastic, patient, and encouraging. Never be patronizing.
- SAFETY PROTOCOL: If the input requests medical/financial advice, involves dangerous activities, or touches on highly sensitive topics, a direct refusal is required.
    - Refusal Tone: Serious, direct, and helpful. No enthusiastic phrasing.
    - Refusal Action: Deflect to a trusted adult.
    
# 4. OUTPUT FORMAT
Provide the output strictly as a JSON object with keys:
"internal_reflection": Analysis of input, complexity decision, and safety check.
"explanation": Final explanation text.
"""

def generate_explanation(input_text):
    """Executes API request to generate pedagogical explanations with retry logic."""
    retries = 3
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model = TEACHER_MODEL,
                response_format= {"type": "json_object"},
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": input_text}
                ],
                temperature=0.6,
                max_tokens = 1024
            )

            content = response.choices[0].message.content.strip()
            data = json.loads(content)

            # Validation of required JSON keys
            if 'internal_reflection' in data and "explanation" in data:
                return data['internal_reflection'], data['explanation']
            else:
                logging.warning(f"Invalid JSON structure received for input: {input_text[:50]}...")
                continue
        
        except openai.RateLimitError:
            # Exponential backoff for rate limiting
            wait_time = (2 ** attempt) * 10
            logging.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            logging.error(f"Error generating explanation (Attempt {attempt+1}/{retries}): {e}")
            time.sleep(20)
    return None, None

def main():
    """Main execution loop for dataset processing and session resumption."""
    logging.info(f"Starting answer generation using {TEACHER_MODEL}...")

    # Input file verification
    if not os.path.exists(INPUT_CSV):
        logging.error(f"Input CSV not found:{INPUT_CSV}")
        return
    
    # Dataset loading with error handling
    try:
        df = pd.read_csv(INPUT_CSV, engine = 'python', on_bad_lines = 'skip', keep_default_na=False)
    except Exception as e:
        logging.error(f"Error reading input CSV: {e}")
        return
    
    logging.info(f"Loaded {len(df)} inputs from {INPUT_CSV}.")

    # Tracking of processed indices for resumption
    processed_indices = set()

    if OUTPUT_JSONL:
        os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok = True)

    # Recovery of previously processed records
    if os.path.exists(OUTPUT_JSONL):
        logging.info("Output file exists. checking for previous progress..")
        try:
            with open(OUTPUT_JSONL, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if 'original_index' in data:
                            processed_indices.add(int(data['original_index']))
                    except json.JSONDecodeError:
                        logging.warning("Skipping corrupted line in existing JSONL file.")
                        continue
        except Exception as e:
            logging.error(f"Error processing existing JSONL file: {e}")
            return
        
        if processed_indices:
            logging.info(f"Resuming. {len(processed_indices)} entries already processed.")

    # Main processing and persistence loop
    with open(OUTPUT_JSONL, 'a') as f:
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating Answers"):
            # Skip indices present in the resume set
            if index in processed_indices:
                continue

            question = str(row["question"]).strip()
            if not question:
                logging.warning(f"Empty question at index {index}. Skipping.")
                continue

            reflection, explanation = generate_explanation(question)

            # Record persistence and disk synchronization
            if explanation:
                output_data = {
                    "original_index" : index,
                    "domain": row.get("domain"),
                    "subject_area": row.get("subject_area"),
                    "generating_model": row.get("generating_model"),
                    "complexity": row.get("complexity"),
                    "input": question,
                    "output": explanation,
                    "teacher_reflection": reflection
                }

                f.write(json.dumps(output_data) + '\n')
                f.flush()
            else:
                logging.error(f"Failed to generate explanation for index {index}. Skipping.")

            # Intentional pacing between API calls
            time.sleep(0.05)
    logging.info(f"\n--- Generation Complete ---")
    logging.info(f"Saved raw dataset to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()