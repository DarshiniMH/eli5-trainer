import os
import json
import hydra
from omegaconf import DictConfig
from tqdm.auto import tqdm
from ctransformers import AutoModelForCausalLM

# Import shared utilities
from src.utils import setup_logging, load_jsonl, ensure_dir, logging

# Use Hydra to manage configurations
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Executes baseline generation using CPU-optimized quantized model weights."""
    
    # Initialize standard logging
    setup_logging()

    # Extract file paths and model parameters from the config
    model_repo = cfg.inference.model_repo
    model_file = cfg.inference.model_file
    validation_file = os.path.join(cfg.files.pro_dir, cfg.inference.validation_file)
    output_file = cfg.inference.output_file

    logging.info(f"Starting baseline generation using CPU-Optimized {model_repo}...")

    # Load the validation dataset and convert it to a list of dictionaries for iteration
    df_val = load_jsonl(validation_file)
    if df_val.empty:
        logging.error("No validation data found. Execution terminated.")
        return
    validation_data = df_val.to_dict('records')
    
    # Load the quantized model directly into host RAM
    try:
        logging.info("Loading model into RAM...")
        
        llm = AutoModelForCausalLM.from_pretrained(
            model_repo,
            model_file=model_file,
            model_type=cfg.inference.get("model_type", "mistral"),
            gpu_layers=cfg.inference.get("gpu_layers", 0), 
            context_length=cfg.inference.get("context_length", 2048)
        )
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    logging.info("Model loaded. Commencing generation loop...")

    # Ensure the output directory exists
    ensure_dir(output_file)

    # Generate answers and stream the results to the output file
    with open(output_file, 'w') as f:
        for item in tqdm(validation_data, desc="Generating answers"):
            input_text = item['input']

            # Format the prompt according to the required instruction template
            formatted_prompt = f"<s>[INST] {input_text} [/INST]"

            try:
                # Execute text generation using configured sampling parameters
                generated_text = llm(
                    formatted_prompt, 
                    max_new_tokens=cfg.inference.get("max_new_tokens", 512), 
                    temperature=cfg.inference.get("temperature", 0.7), 
                    top_p=cfg.inference.get("top_p", 0.95)
                )
                
                # Compile the input, target, and baseline output
                result_data = {
                    "input": input_text,
                    "expected_output": item['output'],
                    "baseline_output": generated_text.strip()
                }
                
                # Write the result to the JSONL file immediately
                f.write(json.dumps(result_data) + '\n')
                f.flush()
            except Exception as e:
                logging.error(f"Error during generation: {e}")
                continue

    logging.info(f"Baseline generation complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()