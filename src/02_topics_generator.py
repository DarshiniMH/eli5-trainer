import argparse
import json
import os
import time
import pandas as pd
import openai
from dotenv import load_dotenv

# Import centralized configurations and shared utilities
from src.config import (
    TOPIC_COMPLEX_CSV, 
    TOPIC_SIMPLE_MINI_CSV, 
    TOPIC_SIMPLE_4O_CSV
)
from src.utils import setup_logging, ensure_dir, logging

# -----------------------------
# 1) CONFIGURATION AND API SETUP
# -----------------------------
setup_logging() # Utilizes centralized logging format
load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    logging.error("OPENAI_API_KEY not found. Ensure .env file is configured.")
    exit()

# -----------------------------
# 2) PROMPTS
# -----------------------------
COMPLEX_PROMPT = """ You are an expert academic researcher and lexicographer.
Your task is to generate a diverse list of complex concepts, specific jargon, academic terminology, significant historical events, or technical mechanisms related to the specified subject area.
These terms should be sophisticated and require detailed explanation to a layperson.
Provide the output strictly as a JSON object with a single key "concepts" containing a list of strings.
Example: {"concepts": ["Cognitive Dissonance", "Keynesian Economics", "The Krebs Cycle", "Aerodynamic Lift"]}
"""

SIMPLE_GENERAL_PROMPT = """You are an expert curriculum designer and educator specializing in K-12 education.
Your task is to generate diverse, simple, and realistic questions that a child (ages 5-12)might ask about a given topic or subject area.
the questions should be clear and cover a wide range of subtopics within the subject.
The questions should reflect genuine curiosity (e.g., "Why...?", "How does...?"),
Provide the output strictly as a JSON object with a single key "questions" containing a list of strings.
Example:{"questions": ["Why is the sky blue?", "How do plants grow?", "Where do mountains come from?"]}
"""

SIMPLE_SAFETY_PROMPT = """You are an AI safety expert. 
Your task is to generate examples of questions that a user might ask, but which are inappropriate, unsafe, or too sensitive for an AI assistant to answer directly. 
These require deflection to a trusted adult or a refusal to answer.
Include topics related to medical advice, sensitive relationships, dangerous activities, and complex emotional situations.
Provide the output strictly as a JSON object with a single key "questions" containing a list of strings.
Example: {"questions": ["How are babies made?", "How do I treat this burn?", "Why do my parents fight?"]}
"""

# -----------------------------
# 3) TAXONOMY MANAGEMENT
# -----------------------------
TAXONOMY_COMPLEX = [
    ("Hard Sciences", "Physics", 100), ("Hard Sciences", "Chemistry", 100),
    ("Hard Sciences", "Math/Logic", 100), ("Hard Sciences", "Astronomy", 100),
    ("Hard Sciences", "Theoretical Physics", 100), ("Life Sciences", "Biology", 100),
    ("Life Sciences", "Medicine", 100), ("Life Sciences", "Neuroscience", 75),
    ("Life Sciences", "Genetics", 75), ("Life Sciences", "Ecology", 50),
    ("Life Sciences", "Health", 50), ("Life Sciences", "Psychology", 100),
    ("Technology & Engineering", "Computing/AI", 200), ("Technology & Engineering", "Engineering & Mechanics", 200),
    ("Technology & Engineering", "Technology Applications", 200), ("Humanities & Social Systems", "History & Events", 125),
    ("Humanities & Social Systems", "Economics & Finance", 125), ("Humanities & Social Systems", "Culture & Society", 125),
    ("Humanities & Social Systems", "Philosophy & Ethics", 125), ("Arts & Literature", "Literature & Writing", 125),
    ("Arts & Literature", "Visual Arts", 125), ("Arts & Literature", "Performing Arts", 100),
    ("Specialized & Meta", "Abstract Concepts", 200), ("Specialized & Meta", "Everyday Life", 100)
]

# Simple taxonomy extends complex with safety categories
TAXONOMY_SIMPLE = TAXONOMY_COMPLEX + [("Specialized & Meta", "Safety/Refusal", 150)]

# -----------------------------
# 4) CORE EXECUTION LOGIC
# -----------------------------
def call_llm(model, system_prompt, user_prompt, key_name):
    """Executes API call with retry logic and JSON validation."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            data = json.loads(response.choices[0].message.content.strip())
            if key_name in data and isinstance(data[key_name], list):
                return data[key_name]
        except Exception as e:
            logging.error(f"Attempt {attempt+1} failed: {e}")
            time.sleep(10)
    return []

def main():
    parser = argparse.ArgumentParser(description="Consolidated Topic Generator")
    parser.add_argument("--mode", choices=["complex", "simple"], required=True)
    parser.add_argument("--model", choices=["gpt-4o", "gpt-4o-mini"], required=True)
    args = parser.parse_args()

    # Determine output path and configuration based on mode and model using config.py
    if args.mode == "complex":
        taxonomy = TAXONOMY_COMPLEX
        key_name = "concepts"
        output_path = TOPIC_COMPLEX_CSV
        sys_prompt_base = COMPLEX_PROMPT
    else:
        taxonomy = TAXONOMY_SIMPLE
        key_name = "questions"
        sys_prompt_base = SIMPLE_GENERAL_PROMPT
        # Naming convention follows model choice for simple questions
        output_path = TOPIC_SIMPLE_4O_CSV if args.model == "gpt-4o" else TOPIC_SIMPLE_MINI_CSV

    all_topics = []
    logging.info(f"Starting {args.mode} generation using {args.model}...")

    for domain, subject, target in taxonomy:
        count = 0
        logging.info(f"Processing {subject} (Target: {target})")

        # Select prompt based on subject classification
        if args.mode == "simple" and subject == "Safety/Refusal":
            current_sys_prompt = SIMPLE_SAFETY_PROMPT
        else:
            current_sys_prompt = sys_prompt_base

        while count < target:
            needed = min(target - count, 100)
            user_msg = f"Generate {needed} distinct {key_name} for the subject area: {subject}."
            
            results = call_llm(args.model, current_sys_prompt, user_msg, key_name)
            if not results:
                logging.warning(f"Failed batch for {subject}. Advancing to next subject.")
                break

            for item in results:
                if count < target and item.strip():
                    all_topics.append({
                        "domain": domain, 
                        "subject_area": subject, 
                        "question": item.strip()
                    })
                    count += 1
            time.sleep(1)

    # Persistence of unique topics to CSV using utility directory management
    df = pd.DataFrame(all_topics).drop_duplicates(subset=["question"])
    ensure_dir(output_path)
    df.to_csv(output_path, index=False)
    logging.info(f"Saved {len(df)} entries to {output_path}")

if __name__ == "__main__":
    main()