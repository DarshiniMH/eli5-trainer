import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
drive_save_path = "models/Mistral_7B_ELI5_Baseline"

dataset = load_dataset(
    "json",
    data_files={
        "train": "data/04_processed/full/train_complete_model.jsonl",
        "validation": "data/04_processed/full/validation_complete_model.jsonl",
    },
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"":0},
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# important for QLoRA stability
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

def formatting_prompts_func(example):
    return f"<s>[INST] {example['input']} [/INST] {example['output']} </s>"

sft_args = SFTConfig(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="none",
    eval_strategy = "steps",
    eval_steps = 25,


    # moved from SFTTrainer(...) into SFTConfig in newer TRL
    max_length=1024,
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    processing_class=tokenizer,          # new name (tokenizer -> processing_class)
    formatting_func=formatting_prompts_func,
)

print("Starting Training...")
trainer.train()

print(f"Saving Model at {drive_save_path}...")
trainer.model.save_pretrained(drive_save_path)
tokenizer.save_pretrained(drive_save_path)
print(f"Done! Model saved to folder: {drive_save_path}")