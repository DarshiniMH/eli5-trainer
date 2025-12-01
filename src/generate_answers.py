import openai
import pandas as pd
import time
import json
import os
import logging
from dotenv import load_dotenv
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# load API key
load_dotenv()

#Setup OpenAI client
client = openai.OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    logging.error("Error: OPENAI_API_KEY is not found. Please check your .env file.")
    exit()

TEACHER_MODEL = "gpt-4o"
INPUT_CSV = "../data/01_raw/merged_master_topic_list.csv"
OUTPUT_JSONL = "../data/02_generated/eli5_dataset_raw.jsonl"

SYSTEM_PROMPT = """You are an award-winning K-12 educator renowned for explaining complex concepts with exceptional clarity, enthusiasm, and factual accuracy. Your goal is to provide the simplest possible explanation tailored to a young audience (ages 5-12).

# 1. ADAPTIVE STYLE (ELI5/ELI12)
- Analyze the input concept/question.
- Determine the minimum complexity required to maintain factual accuracy.
- ELI5 (Ages 5-8): Use very simple vocabulary, short sentences.
- ELI12 (Ages 9-12): If the concept is inherently complex (e.g., inflation, genetics), use slightly more advanced vocabulary (avoiding jargon) and multi-step logic.
- The goal is maximum simplification without becoming incorrect.

# 2. EXPLANATION STRATEGIES (Clarity First)
- Choose the BEST strategy for clarity: Direct Logic OR Analogy.
- Strategy A (Direct Logic): Use clear definitions and step-by-step logic (e.g., 1, 2, 3...). This is preferred for mechanisms, processes, concrete facts, or simple definitions (e.g., "How does a toaster work?" or "What is rain?").
- Strategy B (Analogy): Use relatable analogies (toys, nature, everyday activities) ONLY if the concept is abstract or difficult to visualize AND the analogy is accurate and genuinely helpful. Do not force analogies for concrete concepts.

# 3. TONE AND SAFETY
- General Tone: Be enthusiastic (for general topics), patient, and encouraging. Never be patronizing.
- SAFETY PROTOCOL: If the input requests medical/financial advice, involves dangerous activities, or touches on highly sensitive/inappropriate topics (sexuality, violence, complex trauma, self-harm, manipulation, illegal acts), you MUST refuse to answer directly.
    - Refusal Tone: The tone must be serious, direct, and helpful. DO NOT use enthusiastic phrasing like "That's a great question."
    - Refusal Action: Deflect to a trusted adult.
    - Example Refusal (Emotional/Complex): "That sounds like a really important topic. It's best to talk about this with a parent or a trusted adult who can help you understand it."
    - Example Refusal (Dangerous/Illegal): "I cannot help with this topic. If you are in danger or need help, please talk to a trusted adult or contact emergency services."
    
# 4. OUTPUT FORMAT
Provide the output strictly as a JSON object. Do not include any preamble before the JSON. The JSON must contain two keys:

"internal_reflection": (String) Your internal thought process. Analyze the input, decide on the complexity level (ELI5 or ELI12), evaluate potential analogies, and confirm the factual accuracy of the planned explanation. If it's a safety refusal, explain the safety trigger here.
"explanation": (String) The final explanation text for the user.
"""

def generate_explanation(input_text):
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

            if 'internal_reflection' in data and "explanation" in data:
                return data['internal_reflection'], data['explanation']
            else:
                logging.warning(f"Invalid JSON structure received for input: {input_text[:50]}...")
                continue
        
        except openai.RateLimitError:
            wait_time = (2 ** attempt) * 10
            logging.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            logging.error(f"Error generating explanation (Attempt {attempt+1}/{retries}): {e}")
            time.sleep(20)
    return None, None

def main():
    logging.info(f"Starting answer generation using {TEACHER_MODEL}...")

    if not os.path.exists(INPUT_CSV):
        logging.error(f"Input CSV not found:{INPUT_CSV}")
        return
    
    try:
        df = pd.read_csv(INPUT_CSV, engine = 'python', on_bad_lines = 'skip', keep_default_na=False)
    except Exception as e:
        logging.error(f"Error reading input CSV: {e}")
        return
    
    logging.info(f"Loaded {len(df)} inputs from {INPUT_CSV}.")

    processed_indices = set()

    if OUTPUT_JSONL:
        os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok = True)

    if os.path.exists(OUTPUT_JSONL):
        logging.info("Output file exists. checking for previous package..")
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

    with open(OUTPUT_JSONL, 'a') as f:
        # Use tqdm for a progress bar over the dataframe rows
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating Answers"):
            if index in processed_indices:
                continue

            question = str(row["question"]).strip()
            if not question:
                logging.warning(f"Empty question at index {index}. Skipping.")
                continue

            reflection, explanation = generate_explanation(question)

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

            time.sleep(0.05)
    logging.info(f"\n--- Generation Complete ---")
    logging.info(f"Saved raw dataset to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()