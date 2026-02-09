import os
import json
import time
import logging
import hashlib
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# -----------------------------
# 1) LOGGING AND PATH CONFIGURATION
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# Baseline file containing existing answers for comparison
BASELINE_FILE = "data/05_tuned_results/answer_gen_validation_base_ckpt425_final.jsonl"

# Master output file for multi-agent judge evaluation
OUTPUT_FILE = "data/06_LLM_as_judge_results/eval_runA_runB_runC_multicall.jsonl"

# Mapping of sweep identifiers to their respective adapter paths
RUN_ADAPTERS = {
    "runA": "models/adapters_sweeps/runA_lr1e4_const_r64/best",
    "runB": "models/adapters_sweeps/runB_lr1e4_cosine_r64/best",
    "runC": "models/adapters_sweeps/runC_lr1e4_cosine_r32/best",
}

# Parameters for generation consistency
MAX_NEW_TOKENS = 256
DO_SAMPLE = False
FSYNC_EVERY_N = 25
SLEEP_SEC = 0.01

# -----------------------------
# 2) SAFETY CLASSIFIER UTILITY
# -----------------------------
def is_unsafe_question(question: str, model, tokenizer) -> bool:
    """
    Performs rapid classification to identify questions requiring refusal.
    Utilizes vanilla base model weights to ensure original safety alignment.
    """
    # Classification prompt designed for binary output
    safety_prompt = f"""[INST] Classify this question as 'Safe' or 'Unsafe'. 
    Unsafe includes: medical advice, illegal acts, or dangerous activities.
    Question: {question}
    Category: [/INST]"""
    
    inputs = tokenizer(safety_prompt, return_tensors="pt").to("cuda")
    
    # Greedy decoding with minimal token limit for maximum speed
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        prediction = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
    
    return "unsafe" in prediction

# -----------------------------
# 3) DATA TRACKING HELPERS
# -----------------------------
def make_eval_id(question: str) -> str:
    """Generates stable MD5 hash for consistent tracking across runs."""
    return hashlib.md5(question.strip().encode("utf-8")).hexdigest()[:16]

def load_done_ids(path: str) -> set:
    """Retrieves processed IDs to support session resumption."""
    done = set()
    if not os.path.exists(path): return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                eid = json.loads(line).get("eval_id")
                if eid: done.add(str(eid))
            except Exception: continue
    return done

@torch.inference_mode()
def generate_answer(model, tokenizer, inputs) -> str:
    """Generates text from the active model adapter."""
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# -----------------------------
# 4) MAIN GENERATION PIPELINE
# -----------------------------
def main():
    """Manages adapter hot-swapping and safety-gated generation loop."""
    # Persistence of processed records
    done_ids = load_done_ids(OUTPUT_FILE)
    logging.info(f"Resume enabled. {len(done_ids)} rows already present.")

    # Model initialization with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    logging.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, 
        quantization_config=bnb_config, 
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    )
    base_model.config.use_cache = True  

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Integration of multiple adapters into one PEFT model
    adapter_names = list(RUN_ADAPTERS.keys())
    model = PeftModel.from_pretrained(base_model, RUN_ADAPTERS[adapter_names[0]], adapter_name=adapter_names[0])
    for name in adapter_names[1:]:
        logging.info(f"Adding adapter: {name}")
        model.load_adapter(RUN_ADAPTERS[name], adapter_name=name)
    model.eval()

    written = 0
    # Process baseline file and generate missing model outputs
    with open(BASELINE_FILE, "r") as f_in, open(OUTPUT_FILE, "a") as f_out:
        for line in tqdm(f_in, desc="Processing with Safety Gate"):
            row = json.loads(line)
            question = row.get("question") or row.get("input")
            eval_id = make_eval_id(question)
            if not question or eval_id in done_ids: continue

            # --- STEP 1: SAFETY CLASSIFICATION ---
            # Disabling adapters ensures vanilla safety guardrails are applied
            with model.disable_adapter():
                is_safe = not is_unsafe_question(question, model, tokenizer)

            outputs = {}
            if not is_safe:
                # Standardized refusal for unsafe queries
                refusal_msg = "I cannot answer this question as it involves safety-sensitive topics. Please consult a trusted adult."
                outputs = {name: refusal_msg for name in adapter_names}
            else:
                # --- STEP 2: MULTI-ADAPTER INFERENCE ---
                prompt = f"<s>[INST] {question} [/INST]"
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                for name in adapter_names:
                    model.set_adapter(name)
                    outputs[name] = generate_answer(model, tokenizer, inputs)

            # Record compilation for final dataset
            out_row = {
                "eval_id": eval_id,
                "question": question,
                "subject_area": row.get("subject_area", "Unknown"),
                "is_safety_refusal": not is_safe,
                "base_model": row.get("base_model", ""),
                "tuned_ckpt_425": row.get("tuned_ckpt_425", ""),
                **{f"tuned_{name}": outputs[name] for name in adapter_names}
            }

            f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            f_out.flush()
            written += 1
            # Periodic synchronization to disk to prevent data loss
            if written % FSYNC_EVERY_N == 0: os.fsync(f_out.fileno())

    logging.info(f"Generation complete. Wrote {written} new entries.")

if __name__ == "__main__":
    main()