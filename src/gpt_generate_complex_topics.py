import openai
import pandas as pd
import time
import json
import os
import logging
from dotenv import load_dotenv

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

client = openai.OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    logging.error("Error: OPENAI_API_KEY is not found. Please check your .env file.")
    exit()

TEACHER_MODEL = "gpt-4o-mini"

TAXONOMY = [
    # Domain I: Hard Sciences (500)
    ("Hard Sciences", "Physics", 100),
    ("Hard Sciences", "Chemistry", 100),
    ("Hard Sciences", "Math/Logic", 100),
    ("Hard Sciences", "Astronomy", 100),
    ("Hard Sciences", "Theoretical Physics", 100),
    
    # Domain II: Life Sciences (550)
    ("Life Sciences", "Biology", 100),
    ("Life Sciences", "Medicine", 100),
    ("Life Sciences", "Neuroscience", 75),
    ("Life Sciences", "Genetics", 75),
    ("Life Sciences", "Ecology", 50),
    ("Life Sciences", "Health", 50), 
    # Note: Ensure the spelling matches your simple generation script if merging taxonomies later
    ("Life Sciences", "Psychology", 100), 

    # Domain III: Technology & Engineering (600)
    ("Technology & Engineering", "Computing/AI", 200),
    ("Technology & Engineering", "Engineering & Mechanics", 200),
    ("Technology & Engineering", "Technology Applications", 200),

    # Domain IV: Humanities & Social Systems (500)
    ("Humanities & Social Systems", "History & Events", 125),
    ("Humanities & Social Systems", "Economics & Finance", 125),
    ("Humanities & Social Systems", "Culture & Society", 125),
    ("Humanities & Social Systems", "Philosophy & Ethics", 125),

    # Domain V: Arts & Literature (350)
    ("Arts & Literature", "Literature & Writing", 125),
    ("Arts & Literature", "Visual Arts", 125),
    ("Arts & Literature", "Performing Arts", 100),  

    # Domain VI: Specialized & Meta (300)
    ("Specialized & Meta", "Abstract Concepts", 200),
    ("Specialized & Meta", "Everyday Life", 100),
]

SYSTEM_PROMPT = """ You are an expert academic researcher and lexicographer.
Your task is to generate a diverse list of complex concepts, specific jargon, academic terminology, significant historical events, or technical mechanisms related to the specified subject area.
These terms should be sophisticated and require detailed explanation to a layperson.
Provide the output strictly as a JSON object with a single key "concepts" containing a list of strings.
Example: {"concepts": ["Cognitive Dissonance", "Keynesian Economics", "The Krebs Cycle", "Aerodynamic Lift"]}
"""

def generate_concepts(subject_area, count):
    """Calls the Teacher Model to generate a batch of concepts."""

    batch_size = min(count, 100)
    user_prompt = f"Generate {batch_size} distinct complex concepts or terms related to the subject area: {subject_area}."

    n_retry = 3
    for attempt in range(n_retry):
        try:
            response = client.chat.completions.create(
                model = TEACHER_MODEL,
                response_format = {"type": "json_object"},
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )

            content = response.choices[0].message.content.strip()

            data = json.loads(content)

            if "concepts" in data and isinstance (data["concepts"],list):
                return data["concepts"]
            else:
                logging.warning(f"Invalid JSON structure received for {subject_area}.")
                return []
        except Exception as e:
            logging.error(f"Error generating concepts for {subject_area} (Attempt {attempt+1}/{retries}): {e}")
            time.sleep(20)
    return []

def main():
    all_topics = []
    logging.info("Starting controlled concept generation...")

    for domain, subject, target_count in TAXONOMY:
        generated_count = 0
        logging.info("--- Processing {subject} (Target: {target_count}) ---")

        while generated_count < target_count:
            needed = target_count - generated_count
            concepts = generate_concepts(subject, needed)

            if not concepts:
                logging.warning(f"Failed to generate batch for {subject}. Moving to next subject.")
                break

            for concept in concepts:
                cleaned_concept = concept.strip()
                if generated_count < target_count:
                    if cleaned_concept:
                        all_topics. append({
                            "domain": domain,
                            "subject_area": subject,
                            "question": cleaned_concept
                        })
                        generated_count += 1
                else:
                    break
            
            time.sleep(1)
    
    if not all_topics:
        logging.error("No concepts were generated. Exiting without saving.")
        return
    
    output_df = pd.DataFrame(all_topics)

    initial_count = len(output_df)
    df = output_df.drop_duplicates(subset = ["question"])
    duplicates_removed = initial_count - len(df)

    logging.info(f"\n--- Generation Complete ---")
    logging.info(f"Total unique concepts generated: {len(df)} (Removed {duplicates_removed} duplicates)")
    logging.info("\nDistribution Summary:")
    print(df['subject_area'].value_counts())

    output_dir = "../data/01_raw"
    os.makedirs(output_dir, exist_ok = True)
    output_file = "master_topic_list_complex_mini.csv"
    output_path = os.path.join(output_dir, output_file)
    df.to_csv(output_path, index = False)
    logging.info(f"Saved generated concepts to {output_path}")

if __name__ == "__main__":
    main()

            
