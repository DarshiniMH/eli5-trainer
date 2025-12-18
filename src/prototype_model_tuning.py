import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# --- 1. PROJECT CONFIGURATION ---
# Define the core paths and model ID here for easy updates
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_ADAPTER_PATH = "models/Mistral-7B-ELI5-Prototype"
DATA_PATH_TRAIN = "data/04_processed/prototype/train.jsonl"
DATA_PATH_VAL = "data/04_processed/prototype/validation.jsonl"

# --- 2. LOAD DATASET ---
# Load our custom JSONL files into a Hugging Face Dataset object
print("Loading datasets...")
eli5_dataset = load_dataset(
    "json",
    data_files={
        "train": DATA_PATH_TRAIN,
        "validation": DATA_PATH_VAL,
    },
)

# --- 3. QUANTIZATION CONFIG (4-bit Loading) ---
# Configure bitsandbytes to load the massive model in 4-bit mode to fit on a T4/L4 GPU.
# We use 'nf4' (Normal Float 4) for better precision and 'bfloat16' for stability.
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 is crucial for L4 stability
    bnb_4bit_use_double_quant=False,        # False = slightly more memory, better accuracy
)

# --- 4. LOAD BASE MODEL ---
# Load the pre-trained Mistral model with the quantization settings
print(f"Loading base model: {BASE_MODEL_ID}...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=quant_config,
    device_map="auto",  # Automatically assigns layers to GPU
)

# Disable caching to save memory during training (re-enable for inference later)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# --- 5. TOKENIZER SETUP ---
# Load the tokenizer associated with Mistral
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Mistral doesn't have a pad token, so we use EOS
tokenizer.padding_side = "right"           # Right padding is standard for generation tasks

# --- 6. PREPARE FOR QLORA ---
# Prepares the model for k-bit training (stabilizes normalization layers)
base_model = prepare_model_for_kbit_training(base_model)

# Configure LoRA (Low-Rank Adaptation)
# This tells the trainer to only train a tiny fraction of the weights (adapters)
lora_config = LoraConfig(
    lora_alpha=16,      # Scaling factor for LoRA weights
    lora_dropout=0.1,   # Dropout probability to prevent overfitting
    r=64,               # Rank of the update matrices (higher = more parameters to train)
    bias="none",
    task_type="CAUSAL_LM",
    # Target all linear layers in the transformer block for best results
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# --- 7. FORMATTING FUNCTION ---
# Converts our JSON rows into the specific prompt format Mistral expects:
# <s>[INST] Question [/INST] Answer </s>
def format_instruction_prompt(example):
    return f"<s>[INST] {example['input']} [/INST] {example['output']} </s>"

# --- 8. TRAINING ARGUMENTS ---
# Modern configuration using SFTConfig (replaces TrainingArguments)
training_config = SFTConfig(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,   # Batch size per GPU
    gradient_accumulation_steps=4,   # Accumulate gradients to simulate larger batch size (4x4=16)
    optim="paged_adamw_32bit",       # Paged optimizer to handle memory spikes
    learning_rate=2e-4,              # Standard QLoRA learning rate
    weight_decay=0.001,
    fp16=False,
    bf16=True,                       # Use bfloat16 for training (L4 friendly)
    max_grad_norm=0.3,               # Gradient clipping for stability
    warmup_ratio=0.03,               # Warmup steps to stabilize early training
    group_by_length=True,            # Groups similar length sequences (faster training)
    lr_scheduler_type="constant",    # Keep learning rate constant
    report_to="none",                # Disable WandB/Tensorboard for this run
    
    # Logging and Evaluation Strategy
    logging_steps=10,                # Print stats every 10 steps
    save_steps=25,                   # Save a checkpoint every 25 steps
    eval_strategy="steps",           # Enable evaluation during training
    eval_steps=25,                   # Run validation every 25 steps
    
    # Sequence Length configuration
    max_length=1024,                 # Max tokens per sequence
    packing=False,                   # Don't pack multiple sequences into one
)

# --- 9. INITIALIZE TRAINER ---
# The SFTTrainer handles the training loop
trainer = SFTTrainer(
    model=base_model,
    args=training_config,
    train_dataset=eli5_dataset["train"],
    eval_dataset=eli5_dataset["validation"],
    peft_config=lora_config,
    processing_class=tokenizer,          # Pass tokenizer here (renamed in recent TRL versions)
    formatting_func=format_instruction_prompt,
)

# --- 10. START TRAINING ---
print("Starting QLoRA Fine-tuning...")
trainer.train()

# --- 11. SAVE ARTIFACTS ---
# Save the trained adapter and the tokenizer to Google Drive for persistence
print(f"Saving final model to: {OUTPUT_ADAPTER_PATH}...")
trainer.model.save_pretrained(OUTPUT_ADAPTER_PATH)
tokenizer.save_pretrained(OUTPUT_ADAPTER_PATH)

print(f"✅ Training Complete! Model saved successfully at: {OUTPUT_ADAPTER_PATH}")