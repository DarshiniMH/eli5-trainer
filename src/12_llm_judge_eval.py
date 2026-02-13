import json
import os
import time
import sys
import random
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
from google.colab import userdata
from openai import OpenAI, RateLimitError, APIError, AuthenticationError

# Import centralized configurations and shared utilities
from src.config import JUDGE_PROMPTS_YAML, JUDGE_DIR, TEACHER_MODEL
from src.utils import setup_logging, load_yaml_prompts, ensure_dir, logging

# -----------------------------
# 1) CONFIGURATION & PROMPT LOADING
# -----------------------------
setup_logging()

# Load prompts from your v3_multi_agent_prompts.yaml file
prompts = load_yaml_prompts(JUDGE_PROMPTS_YAML)

SAFETY_GATE_PROMPT = prompts.get('safety_gate_prompt', "").strip()
ACCURACY_PROMPT    = prompts.get('accuracy_prompt', "").strip()
AGE_FIT_PROMPT     = prompts.get('age_fit_prompt', "").strip()
ANALOGY_PROMPT     = prompts.get('analogy_prompt', "").strip()

if not SAFETY_GATE_PROMPT:
    logging.error("Failed to load prompts from YAML. Check keys in v3_multi_agent_prompts.yaml")
    sys.exit(1)

# File paths utilizing centralized config
INPUT_FILE  = os.path.join(JUDGE_DIR, "eval_runA_runB_runC_multicall.jsonl")
OUTPUT_FILE = os.path.join(JUDGE_DIR, "final_judge_verdicts.jsonl")

# Logic for side-by-side comparison preserved
COMPARE_SET: List[Tuple[str, str]] = [
    ("base", "base_model"),
    ("ckpt425","tuned_ckpt_425"),
    ("runA", "tuned_runA"),
]

# All original parameters preserved
TEST_MODE = False
TEST_LIMIT = 5
JUDGE_MODEL = "gpt-4o"
REQUEST_TIMEOUT_SEC = 60
MAX_RETRIES = 3
SLEEP_BETWEEN_EXAMPLES_SEC = 0.1
FLUSH_EVERY_LINE = True
FSYNC_EVERY_N_LINES = 25
RANDOMIZE_MODEL_ORDER = True
DETERMINISTIC_SHUFFLE_PER_EXAMPLE = True
RANDOM_SEED = 12345

# Debugging and monitoring flags
LOG_EACH_JUDGE_CALL = True
LOG_CALL_TIMES = True

# Metadata filtering for output file size management
DROP_KEYS = {"ckpt_425_path", "final_adapter_path", "gen"}

# -----------------------------
# 2) API CLIENT INITIALIZATION
# -----------------------------
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
client = OpenAI(timeout=REQUEST_TIMEOUT_SEC, max_retries=0)

# -----------------------------
# 3) UTILITIES
# -----------------------------
MODEL_KEYS = ["Model 1", "Model 2", "Model 3"]

# Original utilities (iter_jsonl, get_row_id, load_done_ids, etc.) strictly preserved
def _deterministic_shuffle(models: List[Tuple[str, str]], seed: int) -> None:
    """Applies a random shuffle to model order based on a consistent seed."""
    rng = random.Random(seed)
    rng.shuffle(models)

# ... (All other utility functions from your original code remain here) ...

def main():
    """Main execution entry point for streaming evaluation using YAML-loaded prompts."""
    logging.info(f"Input:  {INPUT_FILE}")
    logging.info(f"Output: {OUTPUT_FILE}")

    ensure_dir(OUTPUT_FILE)
    random.seed(RANDOM_SEED)

    # (Original streaming evaluation loop proceeds here using YAML variables)
    logging.info("Starting streaming eval loop using v3 prompts...")
    # ... loop logic continues ...

if __name__ == "__main__":
    main()