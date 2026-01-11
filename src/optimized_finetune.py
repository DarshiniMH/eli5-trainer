import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import os

# --- 1. CONFIGURATION ---
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
drive_save_path = "models/Mistral_7B_ELI5_Optimized"

# --- 2. LOAD DATA ---
dataset = load_dataset(
    "json",
    data_files={
        "train": "data/04_processed/full/train_complete_model.jsonl",
        "validation": "data/04_processed/full/validation_complete_model.jsonl",
    },
)

# --- 3. QUANTIZATION (L4 Optimized) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# --- 4. LOAD MODEL ---
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"":0},
    #attn_implementation="flash_attention_2" # L4 Speed Boost
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# --- 5. TOKENIZER ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.add_bos_token = True # Let tokenizer handle the start token

# --- 6. PREPARE FOR QLORA ---
model = prepare_model_for_kbit_training(model)

# STRATEGY CHANGE 1: Reduced Rank (r) to 32 to prevent overfitting
peft_config = LoraConfig(
    lora_alpha=16,      # Matched to r/2
    lora_dropout=0.1,
    r=32,               # Reduced from 64
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# --- 7. FORMATTING FUNCTION ---
def formatting_prompts_func(example):
    # Handle list (batch) or single string
    if isinstance(example['input'], list):
        output_texts = []
        for i in range(len(example['input'])):
            # STRATEGY CHANGE 2: Removed manual <s> to fix "Double BOS" bug
            text = f"[INST] {example['input'][i]} [/INST] {example['output'][i]} </s>"
            output_texts.append(text)
        return output_texts
    else:
        return f"[INST] {example['input']} [/INST] {example['output']} </s>"

# --- 8. TRAINING ARGUMENTS (Optimized Strategy) ---
training_args = SFTConfig(
    output_dir="./results_final",
    num_train_epochs=1,

    # Batch Size Strategy: Increased Gradient Accumulation
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,   # Effective Batch Size = 32 (Smoother updates)

    optim="paged_adamw_32bit",

    # STRATEGY CHANGE 3: Cosine Scheduler
    learning_rate=2e-4,
    lr_scheduler_type="cosine",      # Starts high, smooths out at the end
    warmup_ratio=0.03,

    # STRATEGY CHANGE 4: Load Best Model
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    load_best_model_at_end=True,     # Keeps the checkpoint with lowest Validation Loss
    save_total_limit=2,              # Save disk space
    metric_for_best_model="eval_loss",

    logging_steps=25,
    weight_decay=0.01,
    fp16=False,
    bf16=True,                       # Essential for L4
    max_grad_norm=0.3,
    group_by_length=True,
    report_to="none",

    max_length=1024,
    packing=False,
)

# --- 9. TRAINER ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_args,
    formatting_func=formatting_prompts_func,
)

# --- 10. RUN ---
print("Starting Optimized Training...")
trainer.train()

# --- 11. SAVE ---
print(f"Saving Best Model at {drive_save_path}...")
trainer.model.save_pretrained(drive_save_path)
tokenizer.save_pretrained(drive_save_path)
print(f"Done! Model saved to folder: {drive_save_path}")