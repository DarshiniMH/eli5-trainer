from ctransformers import AutoModelForCausalLM
import json
import os
import logging
from tqdm.auto import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# We use a GGUF (Quantized) version of Mistral. This fits in your 16GB RAM.
MODEL_REPO = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
MODEL_FILE = "mistral-7b-instruct-v0.2.Q4_K_M.gguf" # 4-bit quantization (approx 4.5GB)

VALIDATION_FILE = "data/04_processed/prototype/validation.jsonl"
OUTPUT_FILE = "results_prototype/baseline_validation_results.jsonl"

def load_jsonl(filepath):
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

def main():
    logging.info(f"Starting baseline generation using CPU-Optimized {MODEL_REPO}...")

    validation_data = load_jsonl(VALIDATION_FILE)
    if not validation_data:
        logging.error("No validation data found.")
        return
    
    # Load the model explicitly for CPU using ctransformers
    # gpu_layers=0 means we run entirely on CPU (safest for your Intel Mac)
    try:
        logging.info("Loading model into RAM... (This might take a minute)")
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

    logging.info("Model loaded. Starting generation...")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        for item in tqdm(validation_data, desc="Generating answers"):
            input_text = item['input']

            # Mistral Instruct format
            formatted_prompt = f"<s>[INST] {input_text} [/INST]"

            try:
                # Generate text directly
                generated_text = llm(
                    formatted_prompt, 
                    max_new_tokens=512, 
                    temperature=0.7, 
                    top_p=0.95
                )
                
                result_data = {
                    "input": input_text,
                    "expected_output": item['output'],
                    "baseline_output": generated_text.strip()
                }
                f.write(json.dumps(result_data) + '\n')
                f.flush()
            except Exception as e:
                logging.error(f"Error during generation: {e}")
                continue

    logging.info(f"Baseline generation complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()