import os
import json
import hashlib
import torch
import hydra
from omegaconf import DictConfig
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Import shared utilities
from src.utils import setup_logging, logging

# -----------------------------
# SAFETY CLASSIFIER UTILITY
# -----------------------------
def is_unsafe_question(question: str, model, tokenizer) -> bool:
    """Uses the un-adapted base model to quickly flag requests that require a refusal."""
    
    
    safety_prompt = f"""[INST] Classify this question as 'Safe' or 'Unsafe'. 
    Unsafe includes: medical advice, illegal acts, or dangerous activities.
    Question: {question}
    Category: [/INST]"""
    
    inputs = tokenizer(safety_prompt, return_tensors="pt").to("cuda")
    
    # Use greedy decoding and strictly limit tokens to maximize classification speed
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        prediction = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
    
    return "unsafe" in prediction

# -----------------------------
# DATA TRACKING HELPERS
# -----------------------------
def make_eval_id(question: str) -> str:
    """Creates a consistent MD5 hash to track rows across multiple script runs."""
    return hashlib.md5(question.strip().encode("utf-8")).hexdigest()[:16]

def load_done_ids(path: str) -> set:
    """Reads the output file to find which questions have already been processed."""
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
def generate_answer(model, tokenizer, inputs, max_new_tokens: int, do_sample: bool) -> str:
    """Generates the actual response using whichever adapter is currently active."""
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# -----------------------------
# MAIN GENERATION PIPELINE
# -----------------------------
# Use Hydra to manage configurations
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Manages adapter hot-swapping and the safety-gated generation loop."""
    
    # Initialize standard logging
    setup_logging()

    # Extract file paths from the centralized files config
    baseline_file = os.path.join(cfg.files.pro_dir, cfg.files.baseline_file)
    output_file = os.path.join(cfg.files.pro_dir, cfg.files.output_file)
    
    # Extract model and generation parameters from the generation config
    base_model_name = cfg.tuned_generation.base_model
    max_new_tokens = cfg.tuned_generation.get("max_new_tokens", 256)
    do_sample = cfg.tuned_generation.get("do_sample", False)
    fsync_every_n = cfg.tuned_generation.get("fsync_every_n", 25)

    # Load previously processed IDs to allow the script to resume safely
    done_ids = load_done_ids(output_file)
    logging.info(f"Resume enabled. {len(done_ids)} rows already present.")

    # Configure 4-bit quantization to reduce memory overhead during multi-adapter inference
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    logging.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        quantization_config=bnb_config, 
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    )
    base_model.config.use_cache = True  

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load all evaluation adapters into a single PEFT model to enable hot-swapping
    adapter_paths = dict(cfg.generation.run_adapters)
    adapter_names = list(adapter_paths.keys())
    
    model = PeftModel.from_pretrained(base_model, adapter_paths[adapter_names[0]], adapter_name=adapter_names[0])
    for name in adapter_names[1:]:
        logging.info(f"Adding adapter: {name}")
        model.load_adapter(adapter_paths[name], adapter_name=name)
    model.eval()

    written = 0
    
    # Iterate through the baseline file and generate responses for each missing record
    with open(baseline_file, "r") as f_in, open(output_file, "a") as f_out:
        for line in tqdm(f_in, desc="Processing with Safety Gate"):
            row = json.loads(line)
            question = row.get("question") or row.get("input")
            eval_id = make_eval_id(question)
            
            # Skip empty questions or those that have already been processed
            if not question or eval_id in done_ids: continue

            # --- STEP 1: SAFETY CLASSIFICATION ---
            # Temporarily disable all adapters so the classifier uses pure base model weights
            with model.disable_adapter():
                is_safe = not is_unsafe_question(question, model, tokenizer)

            outputs = {}
            if not is_safe:
                # Apply a standard refusal message across all adapters if the question is unsafe
                refusal_msg = "I cannot answer this question as it involves safety-sensitive topics. Please consult a trusted adult."
                outputs = {name: refusal_msg for name in adapter_names}
            else:
                # --- STEP 2: MULTI-ADAPTER INFERENCE ---
                # Generate a unique answer from each adapter for safe questions
                prompt = f"<s>[INST] {question} [/INST]"
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                for name in adapter_names:
                    model.set_adapter(name)
                    outputs[name] = generate_answer(model, tokenizer, inputs, max_new_tokens, do_sample)

            # Compile the generated answers with the original record data
            out_row = {
                "eval_id": eval_id,
                "question": question,
                "subject_area": row.get("subject_area", "Unknown"),
                "is_safety_refusal": not is_safe,
                "base_model": row.get("base_model", ""),
                "tuned_ckpt_425": row.get("tuned_ckpt_425", ""),
                **{f"tuned_{name}": outputs[name] for name in adapter_names}
            }

            # Write the result to the output file immediately
            f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            f_out.flush()
            written += 1
            
            # Force a disk sync periodically to prevent data corruption during crashes
            if written % fsync_every_n == 0: os.fsync(f_out.fileno())

    logging.info(f"Generation complete. Wrote {written} new entries.")

if __name__ == "__main__":
    main()