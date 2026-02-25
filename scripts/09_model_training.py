import os
import json
import math
import shutil
import torch
import hydra
from omegaconf import DictConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import TrainerCallback
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Import shared utilities
from src.utils import setup_logging, logging

# -----------------------------
# TRAINER CALLBACKS
# -----------------------------
class SaveBestAdapterToDriveCallback(TrainerCallback):
    """Saves the model and tokenizer whenever the evaluation loss reaches a new low."""
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

            # Overwrite the previous best model and tokenizer
            model.save_pretrained(self.save_dir)
            self.tokenizer.save_pretrained(self.save_dir)

            # Record the step and loss metrics
            with open(os.path.join(self.save_dir, "best_eval.json"), "w") as f:
                json.dump({"global_step": int(state.global_step), "eval_loss": float(loss)}, f)

            print(f"\nNew best eval_loss={loss:.4f} at step {state.global_step}. Saved best adapter -> {self.save_dir}")

        return control

class PeriodicResumeCheckpointToDriveCallback(TrainerCallback):
    """Copies local checkpoints to external storage at set intervals to prevent data loss."""
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

        # Copy the latest checkpoint folder
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

        # Delete older backups to save storage space
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
    """Scans a directory to find the checkpoint folder with the highest step number."""
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

def formatting_prompts_func(example):
    """Formats input-output pairs into standard instruction strings."""
    return f"<s>[INST] {example['input']} [/INST] {example['output']} </s>"

# -----------------------------
# CORE EXECUTION LOGIC
# -----------------------------
# Use Hydra to manage configurations
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Initialize standard logging
    setup_logging()

    # Extract training hyperparameters from the config
    run_name = cfg.train.run_name
    lr = cfg.train.lr
    scheduler = cfg.train.scheduler
    r = cfg.train.lora_r
    alpha = cfg.train.lora_alpha

    # Extract file paths and model names from the config
    model_name = cfg.model.base_model
    train_file = os.path.join(cfg.files.pro_dir, "full_revised/train.jsonl")
    val_file   = os.path.join(cfg.files.pro_dir, "full_revised/validation.jsonl")

    # Load the dataset
    dataset = load_dataset(
        "json",
        data_files={"train": train_file, "validation": val_file},
    )

    # Downsample the training data if a limit is defined in the config
    train_subset_size = cfg.train.get("train_subset_size", None)
    if train_subset_size is not None:
        dataset["train"] = dataset["train"].shuffle(seed=123).select(range(train_subset_size))

    # Downsample the evaluation data if a limit is defined in the config
    eval_subset_size = cfg.train.get("eval_subset_size", 256)
    eval_ds = dataset["validation"]
    if eval_subset_size is not None:
        eval_ds = eval_ds.select(range(min(eval_subset_size, len(eval_ds))))

    # Establish persistence directories
    drive_best_adapter_dir = f"models/adapters_sweeps/{run_name}/best"
    os.makedirs(drive_best_adapter_dir, exist_ok=True)

    drive_resume_root = f"models/resume_checkpoints/{run_name}"
    os.makedirs(drive_resume_root, exist_ok=True)

    local_ckpt_dir = f"./local_checkpoints/{run_name}" 

    # Extract checkpoint parameters from the config
    save_steps_local = cfg.train.get("save_steps", 25)
    eval_steps = cfg.train.get("eval_steps", 25)
    resume_from_drive = cfg.train.get("resume_from_drive", False)

    # Configure 4-bit quantization to reduce memory usage
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    # Load the base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare the model architecture for quantized fine-tuning
    model = prepare_model_for_kbit_training(model)

    # Apply the LoRA configuration
    
    peft_config = LoraConfig(
        lora_alpha=alpha,
        lora_dropout=0.1,
        r=r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Define the supervised fine-tuning parameters
    sft_args = SFTConfig(
        output_dir=local_ckpt_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=save_steps_local,
        eval_steps=eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        learning_rate=lr,
        lr_scheduler_type=scheduler,
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

    # Initialize the trainer with datasets, model, and callbacks
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset["train"],
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
        formatting_func=formatting_prompts_func,
    )

    # Attach custom callbacks for saving and resuming
    trainer.add_callback(SaveBestAdapterToDriveCallback(drive_best_adapter_dir, tokenizer))
    trainer.add_callback(
        PeriodicResumeCheckpointToDriveCallback(
            local_root=local_ckpt_dir,
            drive_root=drive_resume_root,
            every_steps=250,
            keep_last=2,
        )
    )

    print(f"\nSTART TRAINING: {run_name}")

    # Determine if a previous checkpoint exists to resume from
    resume_path = None
    if resume_from_drive:
        resume_path = find_latest_checkpoint(drive_resume_root)
        print("Resuming from checkpoint:", resume_path)

    # Execute the training loop
    trainer.train(resume_from_checkpoint=resume_path)

    # Save the final optimized model and tokenizer
    trainer.model.save_pretrained(drive_best_adapter_dir)
    tokenizer.save_pretrained(drive_best_adapter_dir)

    print("\nDONE")

if __name__ == "__main__":
    main()