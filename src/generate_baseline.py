import torch
from transformers import AutoModelForCausalLLM, AutoTokenizer, pipeline
import json
import os
import logging
from tqdm.auto import tqdm

logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(levelname)s - %(message)s")
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
VALIDATION_FILE = "data/04_processed/prototype/validation.jsonl"
OUTPUT_FILE = "results_prototype/baseline_validation_results.jsonl"

def load_jsonl(filepath):
    data = []
    if not os.path.exists(filepath):
        logging.error(f"Input file not found: {filepath}")
        return []
    try:
        with open(filepath, 'r', encoding ="utf-8") as f:
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
    logging.info(f"Starting baseline generation using {BASE_MODEL}...")

    validation_data = load_jsonl(VALIDATION_FILE)
    if not validation_data:
        return
    
    # Load model and tokenizer
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map = "auto",
        torch_dtype = compute_dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    pipe = pipeline(
        "text-generation",
        model = model,
        tokenizer = tokenizer,
        max_new_tokens = 512,
        temperature = 0.7,
        top_p = 0.95,
        do_sample = True
    )

    logging.info("Starting generation... ")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok = True)

    with open(OUTPUT_FILE, 'w') as f:
        for item in tqdm(validation_data, desc = "Generating baseline answers"):
            input_text = item['input']

            messages = [{"role": "user", "content":input_text}]

            try:
                output = pipe(messages, return_full_text = False)
                generated_text = output[0]["generated_text"].strip()
                result_data = {
                    "input": input_text,
                    "expected_output": item['output'],
                    "baseline_output": generated_text
                }
                f.write(json.dumps(result_data) + '\n')
                f.flush()
            except Exception as e:
                logging.error(f"Error during generation for input: {input_text}. Error: {e}")
                continue
    logging.info(f"Baseline generation complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()