from ctransformers import AutoModelForCausalLM
import json
import os
import logging
from tqdm.auto import tqdm

# -----------------------------
# 1) LOGGING AND CONFIGURATION
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Specification of GGUF quantized model for resource-constrained environments
MODEL_REPO = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
MODEL_FILE = "mistral-7b-instruct-v0.2.Q4_K_M.gguf" 

# File path definitions for validation and result persistence
VALIDATION_FILE = "data/04_processed/prototype/validation.jsonl"
OUTPUT_FILE = "results_prototype/baseline_validation_results.jsonl"

# -----------------------------
# 2) DATA UTILITIES
# -----------------------------
def load_jsonl(filepath):
    """Loads records from a JSONL file into a list of dictionaries."""
    data = []
    if not os.path.exists(filepath):
        logging.error(f"Input file not found: {filepath}")
        return []
    try:
        with open(filepath, 'r', encoding="utf-8") as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logging.error(f"Error reading JSONL file: {e}")
        return []
    return data

# -----------------------------
# 3) MAIN INFERENCE ENGINE
# -----------------------------
def main():
    """Executes baseline generation using CPU-optimized quantized Mistral weights."""
    logging.info(f"Starting baseline generation using CPU-Optimized {MODEL_REPO}...")

    # Data ingestion phase
    validation_data = load_jsonl(VALIDATION_FILE)
    if not validation_data:
        logging.error("No validation data found. Execution terminated.")
        return
    
    # Initialization of quantized model for CPU execution
    # Parameter gpu_layers=0 ensures execution remains on host processor RAM
    try:
        logging.info("Loading model into RAM...")
        llm = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO,
            model_file=MODEL_FILE,
            model_type="mistral",
            gpu_layers=0, 
            context_length=2048
        )
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    logging.info("Model loaded. Commencing generation loop...")

    # Output directory creation
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Generation and persistence of baseline results
    with open(OUTPUT_FILE, 'w') as f:
        for item in tqdm(validation_data, desc="Generating answers"):
            input_text = item['input']

            # Formatting according to Mistral-Instruct prompt template
            formatted_prompt = f"<s>[INST] {input_text} [/INST]"

            try:
                # Execution of text generation with configured sampling parameters
                generated_text = llm(
                    formatted_prompt, 
                    max_new_tokens=512, 
                    temperature=0.7, 
                    top_p=0.95
                )
                
                # Compilation of input, target, and baseline output for evaluation
                result_data = {
                    "input": input_text,
                    "expected_output": item['output'],
                    "baseline_output": generated_text.strip()
                }
                f_out_line = json.dumps(result_data) + '\n'
                f.write(f_out_line)
                f.flush()
            except Exception as e:
                logging.error(f"Error during generation: {e}")
                continue

    logging.info(f"Baseline generation complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()