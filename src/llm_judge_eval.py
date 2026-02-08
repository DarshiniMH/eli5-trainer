import json
import os
import time
import random
import logging
from dotenv import load_dotenv
import yaml
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
from openai import OpenAI, RateLimitError, APIError, AuthenticationError

# -----------------------------
# 1) CONFIGURATION
# -----------------------------
INPUT_FILE  = "data/05_tuned_results/gen_validation_base_ckpt425_runA_runB_runC.jsonl"
OUTPUT_FILE = "data/06_LLM_as_judge_results/eval_base_ckpt425_runA.jsonl"
CONFIG_PATH = "configs/multi_agent.yaml"

COMPARE_SET: List[Tuple[str, str]] = [
    ("base", "base_model"),
    ("ckpt425","tuned_ckpt_425"),
    ("runA", "tuned_runA"),
]

TEST_MODE  = False
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

LOG_EACH_JUDGE_CALL = True
LOG_CALL_TIMES = True

DROP_KEYS = {"ckpt_425_path", "final_adapter_path", "gen"}

# -----------------------------
# 2) ASSET LOADING & API SETUP
# -----------------------------
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    logging.error("Error: OPENAI_API_KEY is not found. Please check your .env file.")
    exit()

def load_config(path: str) -> Dict[str, str]:
    """Loads judge prompts from the specified YAML file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

PROMPTS = load_config(CONFIG_PATH)
SAFETY_GATE_PROMPT = PROMPTS['safety_gate']
ACCURACY_PROMPT    = PROMPTS['accuracy']
AGE_FIT_PROMPT     = PROMPTS['age_fit']
ANALOGY_PROMPT     = PROMPTS['analogy']

# -----------------------------
# 3) UTILITIES
# -----------------------------
MODEL_KEYS = ["Model 1", "Model 2", "Model 3"]

def iter_jsonl(path: str):
    """Yields rows from a jsonl file."""
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
    """Retrieves a unique identifier for the current row."""
    if row.get("eval_id"):
        return str(row["eval_id"])
    if row.get("original_index") is not None:
        return f"oid:{row['original_index']}"
    return f"line:{lineno}"

def load_done_ids(output_path: str) -> set:
    """Identifies rows already processed to allow for resuming."""
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

def remap_by_model_map(model_map: Dict[str, str], by_modelN: Dict[str, Any]) -> Dict[str, Any]:
    """Maps anonymous model identifiers back to their original labels."""
    return {real: by_modelN.get(mN, None) for mN, real in model_map.items()}

def pick_winner_from_totals(totals: Dict[str, int]) -> str:
    """Determines the model with the highest aggregate score."""
    mx = max(totals.values())
    winners = [k for k, v in totals.items() if v == mx]
    return winners[0] if len(winners) == 1 else "tie:" + ",".join(sorted(winners))

def validate_scores_payload(obj: Dict[str, Any], lo: int, hi: int) -> bool:
    """Validates that the judge response contains appropriate score values."""
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
            if v < lo or v > hi:
                return False
        except Exception:
            return False
    return True

def call_judge(label: str, system_prompt: str, user_text: str, validate_fn=None) -> Dict[str, Any]:
    """Executes a call to the LLM judge with retry logic."""
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
                tqdm.write(f"  ! {label} payload invalid; retrying...")
                retries -= 1
                time.sleep(2)
                continue

            return obj

        except RateLimitError as e:
            if "quota" in str(e).lower():
                return {"error": "insufficient_quota"}
            time.sleep(20)
            retries -= 1
        except Exception as e:
            tqdm.write(f"  ! Exception on {label}: {repr(e)}")
            retries -= 1
            time.sleep(5)

    return {"error": f"max_retries_exceeded:{label}"}

# -----------------------------
# 4) EXECUTION LOGIC
# -----------------------------
def evaluate_row(row: Dict[str, Any], row_id: str) -> Dict[str, Any]:
    """Processes a single row through the multi-agent judge panel."""
    required = ["question"] + [f for _, f in COMPARE_SET]
    if any(k not in row for k in required):
        return {"eval_status": "skipped", "error": "missing_fields"}

    q = row["question"]
    models = [(label, row[field]) for label, field in COMPARE_SET]

    if RANDOMIZE_MODEL_ORDER:
        seed = (int(RANDOM_SEED) + (hash(row_id) & 0xFFFFFFFF)) & 0xFFFFFFFF
        random.Random(seed).shuffle(models)

    model_map = {f"Model {i+1}": m[0] for i, m in enumerate(models)}
    user_text_3way = f"Question: {q}\n\n" + "\n\n".join([f"Model {i+1}: {m[1]}" for i, m in enumerate(models)])

    # Safety Evaluation
    safety = call_judge("safety", SAFETY_GATE_PROMPT, user_text_3way)
    if "error" in safety:
        return {"eval_status": "error", "error": f"safety:{safety['error']}", "judge_model_map": model_map}

    classification = safety.get("classification", "Normal")
    violations = remap_by_model_map(model_map, safety.get("violations", {}))

    if classification != "Normal":
        return {
            "eval_status": "complete",
            "judge_model_map": model_map,
            "judge_safety": safety,
            "classification": classification,
            "violations": violations,
            "skipped_non_safety_judges": True
        }

    # Parallel Judge Evaluations
    accN = call_judge("accuracy", ACCURACY_PROMPT, user_text_3way, lambda o: validate_scores_payload(o, 0, 5))
    ageN = call_judge("age_fit", AGE_FIT_PROMPT, user_text_3way, lambda o: validate_scores_payload(o, 0, 3))
    anaN = call_judge("analogy", ANALOGY_PROMPT, user_text_3way, lambda o: validate_scores_payload(o, 0, 2))

    if any("error" in res for res in [accN, ageN, anaN]):
        return {"eval_status": "error", "error": "judge_call_failed"}

    acc_scores = remap_by_model_map(model_map, accN.get("scores", {}))
    age_scores = remap_by_model_map(model_map, ageN.get("scores", {}))
    ana_scores = remap_by_model_map(model_map, anaN.get("scores", {}))

    # Calculate Totals
    totals = {}
    for label, _ in COMPARE_SET:
        if violations.get(label, False):
            totals[label] = 0
        else:
            totals[label] = int(acc_scores[label]) + int(age_scores[label]) + int(ana_scores[label])

    return {
        "eval_status": "complete",
        "judge_model_map": model_map,
        "judge_safety": safety,
        "classification": classification,
        "violations": violations,
        "judge_accuracy": {"scores": acc_scores, "notes": accN.get("notes", {})},
        "judge_age_fit": {"target_level": ageN.get("target_level"), "scores": age_scores, "notes": ageN.get("notes", {})},
        "judge_analogy": {"scores": ana_scores, "notes": anaN.get("notes", {})},
        "total_scores": totals,
        "winner_total": pick_winner_from_totals(totals)
    }

def main():
    """Main entry point for the evaluation process."""
    done_ids = load_done_ids(OUTPUT_FILE)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
        for lineno, row in iter_jsonl(INPUT_FILE):
            row_id = get_row_id(row, lineno)
            if row_id in done_ids:
                continue
            
            verdict = evaluate_row(row, row_id)
            full_record = {"row_id": row_id, **{k: v for k, v in row.items() if k not in DROP_KEYS}, **verdict}
            
            f_out.write(json.dumps(full_record, ensure_ascii=False) + "\n")
            if FLUSH_EVERY_LINE:
                f_out.flush()
            
            if verdict.get("eval_status") == "error":
                break
            time.sleep(SLEEP_BETWEEN_EXAMPLES_SEC)

if __name__ == "__main__":
    main()