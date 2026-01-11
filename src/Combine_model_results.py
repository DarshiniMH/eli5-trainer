import torch
import json
import os
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- CONFIGURATION ---
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# UPDATE THESE PATHS TO MATCH YOUR FOLDERS
PATH_UNSTABLE_ADAPTER = "models/Mistral-7B-ELI5-Full_Dataset"
PATH_OPTIMIZED_ADAPTER = "models/Mistral-7B-ELI5_Optimized"

# INPUT: Your FULL validation file
INPUT_FILE = "data/04_processed/full/validation_complete_model.jsonl"
# OUTPUT: The file containing A/B/C answers for all 820 questions
OUTPUT_FILE = "data/05_tuned_results/tuned_model_answer_generation_combined.jsonl"

def main():
    # 1. SETUP MODEL
    print("Loading Base Mistral Model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"":0},
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 2. LOAD ADAPTERS (Hot-Swap)
    print("Loading Adapters...")
    # Load Unstable (Rank 64)
    model = PeftModel.from_pretrained(base_model, PATH_UNSTABLE_ADAPTER, adapter_name="unstable")
    # Load Optimized (Rank 32)
    model.load_adapter(PATH_OPTIMIZED_ADAPTER, adapter_name="optimized")

    # 3. LOAD DATA
    with open(INPUT_FILE, 'r') as f:
        questions = [json.loads(line) for line in f]

    print(f"Processing ALL {len(questions)} validation examples...")

    # 4. GENERATION LOOP (With Resume Logic)
    # Check if file exists to resume
    start_idx = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            start_idx = sum(1 for _ in f)
        print(f"Resuming from index {start_idx}...")

    # Open in 'a' (append) mode to save progress line-by-line
    with open(OUTPUT_FILE, 'a') as f_out:
        for i, item in enumerate(tqdm(questions)):
            if i < start_idx: continue

            question = item['input']
            # We grab the tag if it exists just for reference, but the judge won't rely on it
            subject = item.get('subject_area', 'Unknown')

            prompt = f"[INST] {question} [/INST]"
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

            # A. BASELINE
            with model.disable_adapter():
                outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id)
                base_ans = tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()

            # B. UNSTABLE (Rank 64)
            model.set_adapter("unstable")
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id)
            unstable_ans = tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()

            # C. OPTIMIZED (Rank 32)
            model.set_adapter("optimized")
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id)
            opt_ans = tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()

            result = {
                "question": question,
                "subject_area": subject,
                "base_model": base_ans,
                "unstable_model": unstable_ans,
                "optimized_model": opt_ans
            }

            f_out.write(json.dumps(result) + "\n")
            f_out.flush()

    print(f"Success! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()