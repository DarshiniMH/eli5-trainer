from ctransformers import AutoModelForCausalLM
import json
import os
from tqdm.auto import tqdm

# Import centralized configurations and shared utilities
from src.config import PRO_DIR
from src.utils import setup_logging, load_jsonl, ensure_dir, logging

# -----------------------------
# 1) LOGGING AND CONFIGURATION
# -----------------------------
setup_logging() # Replaces standard basicConfig

# Specification of GGUF quantized model preserved
MODEL_REPO = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
MODEL_FILE = "mistral-7b-instruct-v0.2.Q4_K_M.gguf" 

# File path definitions utilizing centralized config paths
VALIDATION_FILE = os.path.join(PRO_DIR, "prototype/validation.jsonl")
OUTPUT_FILE = "results_prototype/baseline_validation_results.jsonl"

# -----------------------------
# 2) MAIN INFERENCE ENGINE
# -----------------------------
def main():
    """Executes baseline generation using CPU-optimized quantized Mistral weights."""
    logging.info(f"Starting baseline generation using CPU-Optimized {MODEL_REPO}...")

    # Data ingestion using centralized load utility
    # Note: load_jsonl in utils returns a DataFrame; we convert to list of dicts to match original loop
    df_val = load_jsonl(VALIDATION_FILE)
    if df_val.empty:
        logging.error("No validation data found. Execution terminated.")
        return
    validation_data = df_val.to_dict('records')
    
    # Initialization of quantized model for CPU execution preserved
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

    # Output directory creation using centralized utility
    ensure_dir(OUTPUT_FILE)

    # Generation and persistence of baseline results logic strictly preserved
    with open(OUTPUT_FILE, 'w') as f:
        for item in tqdm(validation_data, desc="Generating answers"):
            input_text = item['input']

            # Formatting according to Mistral-Instruct prompt template
            formatted_prompt = f"<s>[INST] {input_text} [/INST]"

            try:
                # Execution of text generation with original sampling parameters
                generated_text = llm(
                    formatted_prompt, 
                    max_new_tokens=512, 
                    temperature=0.7, 
                    top_p=0.95
                )
                
                # Compilation of input, target, and baseline output preserved
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