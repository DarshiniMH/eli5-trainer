import json
import os
import time
import sys
import random
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
from openai import OpenAI, RateLimitError, APIError, AuthenticationError

# Import centralized configurations and shared utilities
from src.configs import MULTI_AGENT_JUDGE_PROMPTS_YAML, JUDGE_DIR, RES_DIR
from src.utils import setup_logging, load_yaml_prompts, ensure_dir, logging

# -----------------------------
# 1) CONFIGURATION & PROMPT LOADING
# -----------------------------
setup_logging()

# Load prompts from YAML file
prompts = load_yaml_prompts(MULTI_AGENT_JUDGE_PROMPTS_YAML)

SAFETY_GATE_PROMPT = prompts.get('safety_gate_prompt', "").strip()
ACCURACY_PROMPT    = prompts.get('accuracy_prompt', "").strip()
AGE_FIT_PROMPT     = prompts.get('age_fit_prompt', "").strip()
ANALOGY_PROMPT     = prompts.get('analogy_prompt', "").strip()

if not SAFETY_GATE_PROMPT:
    logging.error("Failed to load prompts from YAML. Check keys in v3_multi_agent_prompts.yaml")
    sys.exit(1)

# File paths utilizing centralized config
INPUT_FILE  = os.path.join(RES_DIR, "gen_validation_base_ckpt425_runA_runB_runC.jsonl")
OUTPUT_FILE = os.path.join(JUDGE_DIR, "multicall_eval_base_ckpt425_runA.jsonl")

# Logic for side-by-side comparison
COMPARE_SET: List[Tuple[str, str]] = [
    ("base", "base_model"),
    ("ckpt425","tuned_ckpt_425"),
    ("runA", "tuned_runA"),
]

# Execution parameters
TEST_MODE = False
TEST_LIMIT = 5
JUDGE_MODEL = "gpt-4o"
REQUEST_TIMEOUT_SEC = 60
MAX_RETRIES = 3
SLEEP_BETWEEN_EXAMPLES_SEC = 0.1
FLUSH_EVERY_LINE = True
FSYNC_EVERY_N_LINES = 25

# Position bias mitigation settings
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
# Initializes OpenAI client using environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=REQUEST_TIMEOUT_SEC, max_retries=0)
if not client.api_key:
    logging.error("OPENAI_API_KEY not found. Ensure environment variables are configured.")
    sys.exit(1)

# -----------------------------
# 3) UTILITIES
# -----------------------------
MODEL_KEYS = ["Model 1", "Model 2", "Model 3"]

def iter_jsonl(path: str):
    """Generates rows from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except Exception:
                continue

def get_row_id(row: Dict[str, Any], lineno: int) -> str:
    """Retrieves or generates a unique identifier for a data row."""
    if row.get("eval_id"):
        return str(row["eval_id"])
    if row.get("original_index") is not None:
        return f"oid:{row['original_index']}"
    return f"line:{lineno}"

def load_done_ids(output_path: str) -> set:
    """Loads IDs of already processed examples to support resume functionality."""
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                rid = r.get("row_id")
                status = r.get("eval_status")
                if rid and status in {"complete", "skipped"}:
                    done.add(str(rid))
            except Exception:
                continue
    return done

def _deterministic_shuffle(models: List[Tuple[str, str]], seed: int) -> None:
    """Shuffles model order based on a seed to mitigate position bias."""
    rng = random.Random(seed)
    rng.shuffle(models)

def remap_by_model_map(model_map: Dict[str, str], by_modelN: Dict[str, Any]) -> Dict[str, Any]:
    """Maps anonymized judge labels back to actual model names."""
    return {real: by_modelN.get(mN, None) for mN, real in model_map.items()}

def pick_winner_from_totals(totals: Dict[str, int]) -> str:
    """Determines the winner based on cumulative scores."""
    mx = max(totals.values())
    winners = [k for k, v in totals.items() if v == mx]
    return winners[0] if len(winners) == 1 else "tie:" + ",".join(sorted(winners))

def safe_int(x, default=0) -> int:
    """Safely casts value to integer."""
    try:
        return int(x)
    except Exception:
        return default

def validate_scores_payload(obj: Dict[str, Any], lo: int, hi: int) -> bool:
    """Validates that the judge response contains correct JSON keys and score ranges."""
    if not isinstance(obj, dict):
        return False
    scores = obj.get("scores")
    if not isinstance(scores, dict):
        return False
    for mk in MODEL_KEYS:
        if mk not in scores:
            return False
        try:
            v = int(scores[mk])
        except Exception:
            return False
        if v < lo or v > hi:
            return False
    return True

def call_judge(label: str, system_prompt: str, user_text: str, validate_fn=None) -> Dict[str, Any]:
    """Handles API calls to the judge model with retry logic and error handling."""
    retries = MAX_RETRIES
    while retries > 0:
        try:
            if LOG_EACH_JUDGE_CALL:
                tqdm.write(f"→ Judge call: {label} (retries left: {retries})")

            t0 = time.time()
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                response_format={"type": "json_object"},
                temperature=0
            )
            dt = time.time() - t0
            if LOG_CALL_TIMES:
                tqdm.write(f"  ✓ {label} returned in {dt:.1f}s")

            obj = json.loads(resp.choices[0].message.content)

            if validate_fn is not None and not validate_fn(obj):
                tqdm.write(f"  ! {label} payload invalid/missing keys; retrying...")
                retries -= 1
                time.sleep(2)
                continue

            return obj

        except RateLimitError as e:
            msg = str(e).lower()
            if "insufficient_quota" in msg or "current quota" in msg:
                return {"error": "insufficient_quota"}
            tqdm.write(f"  ! Rate limit on {label}. Sleeping 20s...")
            time.sleep(20)
            retries -= 1
        except AuthenticationError:
            return {"error": "invalid_api_key"}
        except (APIError, json.JSONDecodeError, Exception) as e:
            tqdm.write(f"  ! Error on {label}: {repr(e)}. Sleeping 10s...")
            time.sleep(10)
            retries -= 1

    return {"error": f"max_retries_exceeded:{label}"}

def evaluate_row(row: Dict[str, Any], row_id: str) -> Dict[str, Any]:
    """Orchestrates multi-agent evaluation for a single row of data."""
    required = ["question"] + [f for _, f in COMPARE_SET]
    missing = [k for k in required if k not in row]
    if missing:
        return {"eval_status": "skipped", "error": f"missing_required_fields:{missing}"}

    q = row["question"]
    models: List[Tuple[str, str]] = [(label, row[field]) for label, field in COMPARE_SET]

    if RANDOMIZE_MODEL_ORDER:
        if DETERMINISTIC_SHUFFLE_PER_EXAMPLE:
            seed = (int(RANDOM_SEED) + (hash(row_id) & 0xFFFFFFFF)) & 0xFFFFFFFF
            _deterministic_shuffle(models, seed)
        else:
            random.shuffle(models)

    model_map = {"Model 1": models[0][0], "Model 2": models[1][0], "Model 3": models[2][0]}

    user_text_3way = f"Question: {q}\n\nModel 1: {models[0][1]}\n\nModel 2: {models[1][1]}\n\nModel 3: {models[2][1]}"

    # Safety Gate Execution
    safety = call_judge("safety", SAFETY_GATE_PROMPT, user_text_3way)
    if "error" in safety:
        return {"eval_status": "error", "error": f"safety:{safety['error']}", "judge_model_map": model_map}

    classification = safety.get("classification", "Normal")
    violations_by_modelN = safety.get("violations", {"Model 1": False, "Model 2": False, "Model 3": False})
    violations = remap_by_model_map(model_map, violations_by_modelN)

    if classification != "Normal":
        return {
            "eval_status": "complete",
            "judge_model_map": model_map,
            "judge_safety": safety,
            "classification": classification,
            "violations": violations,
            "skipped_non_safety_judges": True,
            "total_scores": None,
            "winner_total": None
        }

    # Accuracy, Age-Fit, and Analogy Judge Execution
    accN = call_judge("accuracy", ACCURACY_PROMPT, user_text_3way, validate_fn=lambda o: validate_scores_payload(o, 0, 5))
    if "error" in accN:
        return {"eval_status": "error", "error": f"accuracy:{accN['error']}", "judge_model_map": model_map}

    ageN = call_judge("age_fit", AGE_FIT_PROMPT, user_text_3way, validate_fn=lambda o: validate_scores_payload(o, 0, 3))
    if "error" in ageN:
        return {"eval_status": "error", "error": f"age_fit:{ageN['error']}", "judge_model_map": model_map}

    anaN = call_judge("analogy", ANALOGY_PROMPT, user_text_3way, validate_fn=lambda o: validate_scores_payload(o, 0, 2))
    if "error" in anaN:
        return {"eval_status": "error", "error": f"analogy:{anaN['error']}", "judge_model_map": model_map}

    acc_scores = remap_by_model_map(model_map, accN.get("scores", {}))
    age_scores = remap_by_model_map(model_map, ageN.get("scores", {}))
    ana_scores = remap_by_model_map(model_map, anaN.get("scores", {}))

    # Apply safety violation penalty
    for label, _ in COMPARE_SET:
        if violations.get(label, False):
            acc_scores[label] = 0
            age_scores[label] = 0
            ana_scores[label] = 0

    totals = {label: safe_int(acc_scores.get(label)) + safe_int(age_scores.get(label)) + safe_int(ana_scores.get(label))
              for label, _ in COMPARE_SET}
    winner_total = pick_winner_from_totals(totals)

    return {
        "eval_status": "complete",
        "judge_model_map": model_map,
        "judge_safety": safety,
        "classification": classification,
        "violations": violations,
        "judge_accuracy": {"scores": acc_scores, "notes": accN.get("notes", {})},
        "judge_age_fit": {"target_level": ageN.get("target_level"), "scores": age_scores, "notes": ageN.get("notes", {})},
        "judge_analogy": {"scores": ana_scores, "notes": anaN.get("notes", {})},
        "skipped_non_safety_judges": False,
        "total_scores": totals,
        "winner_total": winner_total
    }

def main():
    """Main execution loop for processing files and writing evaluation results."""
    logging.info(f"Input:  {INPUT_FILE}")
    logging.info(f"Output: {OUTPUT_FILE}")

    ensure_dir(OUTPUT_FILE)
    done_ids = load_done_ids(OUTPUT_FILE)
    logging.info(f"Loaded {len(done_ids)} existing results.")

    random.seed(RANDOM_SEED)
    completed = 0
    attempted = 0
    written_lines = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
        pbar = tqdm(total=TEST_LIMIT if TEST_MODE else None)
        logging.info("Starting streaming eval loop...")

        for lineno, row in iter_jsonl(INPUT_FILE):
            row_id = get_row_id(row, lineno)

            if row_id in done_ids:
                continue
            if TEST_MODE and completed >= TEST_LIMIT:
                break

            attempted += 1
            tqdm.write(f"\n=== Evaluating row_id={row_id} (line {lineno}) ===")

            verdict = evaluate_row(row, row_id=row_id)

            # Metadata cleanup and record finalization
            row_clean = {k: v for k, v in row.items() if k not in DROP_KEYS}
            full_record = {"row_id": row_id, **row_clean, **verdict}

            f_out.write(json.dumps(full_record, ensure_ascii=False) + "\n")
            written_lines += 1

            if FLUSH_EVERY_LINE:
                f_out.flush()
            if FSYNC_EVERY_N_LINES and (written_lines % FSYNC_EVERY_N_LINES == 0):
                os.fsync(f_out.fileno())

            if verdict.get("eval_status") in {"complete", "skipped"}:
                completed += 1
                pbar.update(1)
                done_ids.add(row_id)

            if verdict.get("eval_status") == "error":
                tqdm.write(f"Error: {verdict.get('error')}. Stopping to prevent credit waste.")
                break

            time.sleep(SLEEP_BETWEEN_EXAMPLES_SEC)

    logging.info("Run complete.")

if __name__ == "__main__":
    main()