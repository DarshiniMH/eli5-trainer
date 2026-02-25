import os
import json
import time
from typing import Dict, Any, Optional
from tqdm.auto import tqdm
from openai import OpenAI, RateLimitError, APIError
import hydra
from omegaconf import DictConfig

# Import centralized shared utilities only
from src.utils import setup_logging, ensure_dir, logging

# -----------------------------
# CONFIGURATION AND API SETUP
# -----------------------------
setup_logging()
client = OpenAI()

# -----------------------------
# REWRITING SYSTEM PROMPT
# -----------------------------
SYSTEM_PROMPT_COMPLEX_NATURAL = """
You are a K–12 teacher explaining for ages 9–12 (ELI12).

Write a natural explanation in a smooth flow. Do NOT use:
- numbered sections like "1) ..."
- headings like "Definition:", "Explanation:", "Example:"

Guidelines:
- Start with a clear definition, then explain the nuances involved in simple terms.
- If it’s a process/mechanism ("how/why/steps") that can be better explained with steps, you may use bullet points.
- Elaborate as needed to be accurate and complete.
- Include at least one simple example or an analogy near the end to make the topic concrete.
- If you use analogy, keep it brief and follow it with the real explanation.

Return JSON only:
{"explanation": "..."}
""".strip()

# -----------------------------
# CLASSIFICATION UTILITIES
# -----------------------------
def word_count(text: str) -> int:
    """Returns the total number of words in a string for length validation."""
    return len((text or "").split())

def looks_like_refusal(text: str) -> bool:
    """Identifies content containing refusal markers or redirection to adults."""
    t = (text or "").lower()
    markers = [
        "talk to a trusted adult", "parent or a trusted adult",
        "best to talk to", "i can't help", "i cannot help",
        "i can’t help", "contact emergency services",
        "i’m not able to help", "i am not able to help"
    ]
    return any(m in t for m in markers) and word_count(t) <= 140

def is_safety_refusal_row(row: Dict[str, Any]) -> bool:
    """Determines if a record belongs to safety or refusal categories for preservation."""
    subject = (row.get("subject_area") or "").lower()
    domain = (row.get("domain") or "").lower()

    if "safety" in subject or "refusal" in subject:
        return True
    if "safety" in domain or "refusal" in domain:
        return True
    if looks_like_refusal(row.get("output") or ""):
        return True
    return False

# -----------------------------
# PERSISTENCE AND RESUME LOGIC
# -----------------------------
def load_done_ids(output_path: str) -> set:
    """Retrieves processed indices from output file to support session resumption."""
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r") as f:
        for line in f:
            try:
                r = json.loads(line)
                oid = r.get("original_index", None)
                if oid is not None:
                    done.add(int(oid))
            except Exception:
                continue
    return done

# -----------------------------
# GENERATION INTERFACE
# -----------------------------
def _call_teacher(model_name: str, system_prompt: str, user_msg: str, temperature: float, max_tokens: int, max_retries: int) -> Optional[str]:
    """Executes API request with exponential backoff for rate limits."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            payload = json.loads(resp.choices[0].message.content.strip())
            explanation = (payload.get("explanation") or "").strip()
            return explanation if explanation else None

        except (RateLimitError, APIError) as e:
            wait = (2 ** attempt) * 5
            logging.warning(f"API error. Retrying in {wait}s...")
            time.sleep(wait)
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            time.sleep(3)
    return None

def rewrite_complex_natural(question: str, existing_answer: str, domain: str, subject: str, cfg: DictConfig) -> Optional[str]:
    """Orchestrates the rewriting and optional expansion of complex answers."""
    user_msg = f"QUESTION: {question}\nDOMAIN: {domain}\nSUBJECT: {subject}\n\nEXISTING_ANSWER:\n{existing_answer}"

    # Draft generation
    draft = _call_teacher(
        model_name=cfg.models.teacher,
        system_prompt=SYSTEM_PROMPT_COMPLEX_NATURAL,
        user_msg=user_msg,
        temperature=cfg.rewriting.draft_temp,
        max_tokens=cfg.rewriting.draft_max_tokens,
        max_retries=cfg.rewriting.max_retries
    )
    
    if not draft or looks_like_refusal(draft):
        return None

    # Expansion pass for ELI12 targets falling below word threshold
    if word_count(draft) < cfg.rewriting.min_complex_words:
        expand_msg = f"Expand this for ages 9–12. Keep it natural. QUESTION: {question}\n\nDRAFT:\n{draft}"
        expanded = _call_teacher(
            model_name=cfg.models.teacher,
            system_prompt=SYSTEM_PROMPT_COMPLEX_NATURAL,
            user_msg=expand_msg,
            temperature=cfg.rewriting.expand_temp,
            max_tokens=cfg.rewriting.expand_max_tokens,
            max_retries=cfg.rewriting.max_retries
        )
        if expanded and not looks_like_refusal(expanded):
            return expanded
            
    return draft

# -----------------------------
# MAIN EXECUTION LOOP
# -----------------------------
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Manages the processing pipeline including deduplication and file synchronization."""
    input_file = cfg.files.eli5_raw_jsonl
    output_file = cfg.files.eli5_rewritten_jsonl
    
    ensure_dir(output_file)
    done_ids = load_done_ids(output_file)
    logging.info(f"Resume enabled. Found {len(done_ids)} rows in output.")

    metrics = {"total": 0, "written": 0, "rewritten": 0, "skipped_safety": 0, "skipped_done": 0, "skipped_noncomplex": 0}

    def write_record(f_out, record: Dict[str, Any]):
        """Writes record to disk and ensures data persistence."""
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
        f_out.flush()
        if cfg.rewriting.force_drive_sync:
            os.fsync(f_out.fileno())
        metrics["written"] += 1
        oid_local = record.get("original_index", None)
        if oid_local is not None:
            done_ids.add(int(oid_local))

    with open(input_file, "r") as fin, open(output_file, "a") as fout:
        for line in tqdm(fin, desc="Rewriting complex questions"):
            metrics["total"] += 1
            if cfg.rewriting.limit_rows and metrics["total"] > cfg.rewriting.limit_rows:
                break

            row = json.loads(line)
            oid = row.get("original_index", None)

            if oid is not None and int(oid) in done_ids:
                metrics["skipped_done"] += 1
                continue

            # Check for safety/refusal before attempting rewrites
            if is_safety_refusal_row(row):
                metrics["skipped_safety"] += 1
                write_record(fout, row)
                continue

            complexity = (row.get("complexity") or "").lower().strip()
            if complexity != "complex":
                metrics["skipped_noncomplex"] += 1
                write_record(fout, row)
                continue

            # Core rewriting logic for complex queries
            new_answer = rewrite_complex_natural(
                row.get("input", ""), 
                row.get("output", ""), 
                row.get("domain", ""), 
                row.get("subject_area", ""),
                cfg
            )

            if new_answer:
                metrics["rewritten"] += 1
                out_row = dict(row)
                out_row.update({"output": new_answer, "rewritten": True, "rewriter_model": cfg.models.teacher})
                write_record(fout, out_row)
            else:
                write_record(fout, row)

            time.sleep(cfg.rewriting.sleep_sec)

    logging.info(f"Process complete. Rewritten: {metrics['rewritten']}")

if __name__ == "__main__":
    main()