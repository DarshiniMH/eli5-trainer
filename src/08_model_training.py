import os
import json
import math
import shutil
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import TrainerCallback
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# -----------------------------
# 0) HYPERPARAMETER CONFIGURATIONS
# -----------------------------
RUN_CONFIGS = {
    "runA_lr2e4_const_r64": dict(lr=2e-4, scheduler="constant", r=64, alpha=16),
    "runB_lr1e4_const_r64": dict(lr=1e-4, scheduler="constant", r=64, alpha=16),
    "runC_lr1e4_cosine_r64": dict(lr=1e-4, scheduler="cosine", r=64, alpha=16),
    "runD_lr1e4_cosine_r32": dict(lr=1e-4, scheduler="cosine", r=32, alpha=8),
}

RUN_NAME = "runA_lr2e4_const_r64"
assert RUN_NAME in RUN_CONFIGS, f"RUN_NAME must be one of: {list(RUN_CONFIGS.keys())}"
cfg = RUN_CONFIGS[RUN_NAME]

# -----------------------------
# 1) DATASET INITIALIZATION
# -----------------------------
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

TRAIN_FILE = "data/04_processed/full_revised/train_full_revised.jsonl"
VAL_FILE   = "data/04_processed/full_revised/validation_full_revised.jsonl"

# Subset size for ablation studies
TRAIN_SUBSET_SIZE = None

# Evaluation batch subset size
EVAL_SUBSET_SIZE = 256

dataset = load_dataset(
    "json",
    data_files={"train": TRAIN_FILE, "validation": VAL_FILE},
)

# Selection of training data subset
if TRAIN_SUBSET_SIZE is not None:
    dataset["train"] = dataset["train"].shuffle(seed=123).select(range(TRAIN_SUBSET_SIZE))

# Selection of validation data subset
eval_ds = dataset["validation"]
if EVAL_SUBSET_SIZE is not None:
    eval_ds = eval_ds.select(range(min(EVAL_SUBSET_SIZE, len(eval_ds))))

# -----------------------------
# 2) PERSISTENCE DIRECTORIES
# -----------------------------

# Directory for best adapter storage
DRIVE_BEST_ADAPTER_DIR = f"models/adapters_sweeps/{RUN_NAME}/best"
os.makedirs(DRIVE_BEST_ADAPTER_DIR, exist_ok=True)

# Directory for periodic checkpoint storage
ENABLE_RESUME_TO_DRIVE = True
DRIVE_RESUME_ROOT = f"models/resume_checkpoints/{RUN_NAME}"
os.makedirs(DRIVE_RESUME_ROOT, exist_ok=True)

# -----------------------------
# 3) TRAINING FLOW PARAMETERS
# -----------------------------
# Step alignment for local saves and evaluation
SAVE_STEPS_LOCAL = 25
EVAL_STEPS = 25
SAVE_TOTAL_LIMIT_LOCAL = 3

# Frequency and capacity for resume checkpoints
RESUME_TO_DRIVE_EVERY_STEPS = 250
KEEP_LAST_N_RESUME_CKPTS = 2

# Resumption status from external storage
RESUME_FROM_DRIVE = False

# -----------------------------
# 4) TRAINER CALLBACKS
# -----------------------------
class SaveBestAdapterToDriveCallback(TrainerCallback):
    """
    Persistence of LoRA adapter and tokenizer upon evaluation loss improvement.
    """
    def __init__(self, save_dir: str, tokenizer):
        self.save_dir = save_dir
        self.tokenizer = tokenizer
        self.best_loss = math.inf
        os.makedirs(save_dir, exist_ok=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return control
        loss = metrics.get("eval_loss", None)
        if loss is None:
            return control

        if loss < self.best_loss:
            self.best_loss = loss
            model = kwargs.get("model", None)
            if model is None:
                return control

            # Storage of optimal model and tokenizer
            model.save_pretrained(self.save_dir)
            self.tokenizer.save_pretrained(self.save_dir)

            # Metadata persistence for metrics
            with open(os.path.join(self.save_dir, "best_eval.json"), "w") as f:
                json.dump({"global_step": int(state.global_step), "eval_loss": float(loss)}, f)

            print(f"\nNew best eval_loss={loss:.4f} at step {state.global_step}. Saved best adapter -> {self.save_dir}")

        return control


class PeriodicResumeCheckpointToDriveCallback(TrainerCallback):
    """
    Replication of local checkpoint directories to external storage.
    """
    def __init__(self, local_root: str, drive_root: str, every_steps: int, keep_last: int = 2):
        self.local_root = local_root
        self.drive_root = drive_root
        self.every_steps = every_steps
        self.keep_last = keep_last
        os.makedirs(drive_root, exist_ok=True)

    def on_save(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step <= 0 or (step % self.every_steps) != 0:
            return control

        src = os.path.join(self.local_root, f"checkpoint-{step}")
        if not os.path.exists(src):
            return control

        dst = os.path.join(self.drive_root, f"checkpoint-{step}")

        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

        # Cleanup of outdated checkpoints
        ckpts = []
        for name in os.listdir(self.drive_root):
            if name.startswith("checkpoint-"):
                try:
                    ckpts.append((int(name.split("-")[1]), name))
                except Exception:
                    pass
        ckpts.sort()
        while len(ckpts) > self.keep_last:
            _, old = ckpts.pop(0)
            shutil.rmtree(os.path.join(self.drive_root, old), ignore_errors=True)

        print(f"\nSaved resume checkpoint: {dst}")
        return control


def find_latest_checkpoint(root: str) -> str | None:
    """Detection of most recent checkpoint directory within specified root path."""
    if not os.path.exists(root):
        return None
    best = None
    for name in os.listdir(root):
        if name.startswith("checkpoint-"):
            try:
                step = int(name.split("-")[1])
                if best is None or step > best[0]:
                    best = (step, os.path.join(root, name))
            except Exception:
                continue
    return best[1] if best else None

# -----------------------------
# 5) MODEL AND TOKENIZER SETUP
# -----------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Preparation of model for k-bit training architecture
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    lora_alpha=cfg["alpha"],
    lora_dropout=0.1,
    r=cfg["r"],
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

def formatting_prompts_func(example):
    """Conversion of input-output pairs into standardized instruction strings."""
    return f"<s>[INST] {example['input']} [/INST] {example['output']} </s>"

# -----------------------------
# 6) TRAINING SPECIFICATIONS
# -----------------------------
sft_args = SFTConfig(
    output_dir=LOCAL_CKPT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_strategy="steps",
    eval_strategy="steps",
    save_steps=SAVE_STEPS_LOCAL,
    eval_steps=EVAL_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT_LOCAL,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=10,
    learning_rate=cfg["lr"],
    lr_scheduler_type=cfg["scheduler"],
    warmup_ratio=0.03,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    group_by_length=True,
    report_to="none",
    max_length=1024,
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=dataset["train"],
    eval_dataset=eval_ds,
    peft_config=peft_config,
    processing_class=tokenizer,
    formatting_func=formatting_prompts_func,
)

# Implementation of persistence and resumption callbacks
trainer.add_callback(SaveBestAdapterToDriveCallback(DRIVE_BEST_ADAPTER_DIR, tokenizer))
if ENABLE_RESUME_TO_DRIVE:
    trainer.add_callback(
        PeriodicResumeCheckpointToDriveCallback(
            local_root=LOCAL_CKPT_DIR,
            drive_root=DRIVE_RESUME_ROOT,
            every_steps=RESUME_TO_DRIVE_EVERY_STEPS,
            keep_last=KEEP_LAST_N_RESUME_CKPTS,
        )
    )

print(f"\nSTART TRAINING: {RUN_NAME}")

# Determination of checkpoint resumption path
resume_path = None
if RESUME_FROM_DRIVE:
    resume_path = find_latest_checkpoint(DRIVE_RESUME_ROOT)
    print("Resuming from checkpoint:", resume_path)

trainer.train(resume_from_checkpoint=resume_path)

# Final storage of optimized model and tokenizer
trainer.model.save_pretrained(DRIVE_BEST_ADAPTER_DIR)
tokenizer.save_pretrained(DRIVE_BEST_ADAPTER_DIR)

print("\nDONE")