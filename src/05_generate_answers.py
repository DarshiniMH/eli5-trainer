import openai
import pandas as pd
import time
import json
import os
from dotenv import load_dotenv
from tqdm.auto import tqdm

# Import centralized configurations and shared utilities
from src.config import TEACHER_MODEL, MERGED_TOPICS_CSV, ELI5_RAW_JSONL
from src.utils import setup_logging, ensure_dir, logging

# -----------------------------
# 1) API INITIALIZATION
# -----------------------------
setup_logging() # Standardized logging format
load_dotenv()

# Setup for the OpenAI client and validation of the API key
client = openai.OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    logging.error("Error: OPENAI_API_KEY is not found. Please check your .env file.")
    exit()

# -----------------------------
# 2) SYSTEM PROMPT (PEDAGOGICAL)
# -----------------------------
# Original educator persona and safety protocol strictly preserved
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
                model = TEACHER_MODEL, # Managed via config.py
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

            # Validation of required JSON keys preserved
            if 'internal_reflection' in data and "explanation" in data:
                return data['internal_reflection'], data['explanation']
            else:
                logging.warning(f"Invalid JSON structure received for input: {input_text[:50]}...")
                continue
        
        except openai.RateLimitError:
            # Exponential backoff for rate limiting preserved
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

    # Input file verification utilizing config path
    if not os.path.exists(MERGED_TOPICS_CSV):
        logging.error(f"Input CSV not found:{MERGED_TOPICS_CSV}")
        return
    
    # Dataset loading with error handling
    try:
        df = pd.read_csv(MERGED_TOPICS_CSV, engine = 'python', on_bad_lines = 'skip', keep_default_na=False)
    except Exception as e:
        logging.error(f"Error reading input CSV: {e}")
        return
    
    logging.info(f"Loaded {len(df)} inputs from {MERGED_TOPICS_CSV}.")

    # Tracking of processed indices for resumption
    processed_indices = set()
    ensure_dir(ELI5_RAW_JSONL) # Managed via utils.py

    # Recovery of previously processed records
    if os.path.exists(ELI5_RAW_JSONL):
        logging.info("Output file exists. checking for previous progress..")
        try:
            with open(ELI5_RAW_JSONL, 'r') as f:
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

    # Main processing and persistence loop with original metadata fields
    with open(ELI5_RAW_JSONL, 'a') as f:
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
    logging.info(f"Saved raw dataset to {ELI5_RAW_JSONL}")

if __name__ == "__main__":
    main()