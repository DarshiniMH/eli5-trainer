import os

# Base Directories
RAW_DIR = "data/01_raw"
GEN_DIR = "data/02_generated"
CUR_DIR = "data/03_curated"
PRO_DIR = "data/04_processed"
RES_DIR = "data/05_tuned_results"
JUDGE_DIR = "data/06_LLM_as_judge_results"

# File Names
TOPIC_COMPLEX_CSV = os.path.join(RAW_DIR, "master_topic_list_complex_mini.csv")
TOPIC_SIMPLE_MINI_CSV = os.path.join(RAW_DIR, "master_topic_list_tagged_4o_mini.csv")
TOPIC_SIMPLE_4O_CSV = os.path.join(RAW_DIR, "master_topic_list_tagged_4o.csv")
MERGED_TOPICS_CSV = os.path.join(RAW_DIR, "merged_master_topic_list.csv")

ELI5_RAW_JSONL = os.path.join(GEN_DIR, "eli5_dataset_raw.jsonl")
ELI5_REWRITTEN_JSONL = os.path.join(GEN_DIR, "eli5_dataset_rewritten_complex.jsonl")

CURATION_SAMPLE_CSV = os.path.join(CUR_DIR, "review_sample.csv")

# Model Names
TEACHER_MODEL = "gpt-4o"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# ... existing paths ...
MULTI_AGENT_JUDGE_PROMPTS_YAML = "src/v3_multi_agent_prompts.yaml"