import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

ADAPTER_PATH = "models/Mistral-7B-ELI5-Prototype_with_eval_loss"

# Path to the GROUND TRUTH file (Questions + GPT Answers)
VALIDATION_FILE = "data/04_processed/prototype/validation.jsonl"

# Path to the OLD BASELINE file (Mistral Base Answers)
BASELINE_FILE = "results_prototype/baseline_validation_results.jsonl"

# Output File (Where results will be saved)
OUTPUT_FILE = "data/05_tuned_results/final_comparison_results.jsonl"

BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

print("Loading Model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16
    )

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config = bnb_config,
    device_map = "auto"
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

print("\nStarting Comparision...\n")

with open(VALIDATION_FILE, "r") as f_val, open(BASELINE_FILE, 'r') as f_base,\
open(OUTPUT_FILE, 'w') as f_out:
    for i, (line_val, line_base) in enumerate(zip(f_val, f_base)):
      #if i >= 5: break

      data_val = json.loads(line_val)
      data_base = json.loads(line_base)

      question = data_val['input']
      expected_answer = data_val["output"]
      baseline_mistral_answer = data_base["baseline_output"]

      prompt = f"<s>[INST] {question} [/INST]"
      inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

      with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = 200,
            temperature = 0.7,
            do_sample = True,
            top_p = 0.9
        )

      tuned_ans = tokenizer.decode(outputs[0], skip_special_tokens= True)

      tuned_ans_clean = tuned_ans.replace(prompt, "").strip()

      result_entry = {
          "question": question,
          "baseline_mistral": baseline_mistral_answer,
          "expected_gpt": expected_answer,
          "tuned_model": tuned_ans_clean
        }
        
      # Write it as a single line of JSON
      f_out.write(json.dumps(result_entry) + "\n")

      # --- OPTIONAL: PRINT TO SCREEN (To verify it's working) ---
      print(f"Processed Question {i+1}...")