import random
import pandas as pd
import openai 
import time
import json
import os
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    logging.error("Error: OPENAI_API_KEY is not found. Please check your .env file.")
    exit()

TEACHER_MODEL = "gpt-4o"

TAXONOMY = [
    # Domain I: Hard Sciences (750)
    ("Hard Sciences", "Physics", 150),
    ("Hard Sciences", "Chemistry", 150),
    ("Hard Sciences", "Math/Logic", 150),
    ("Hard Sciences", "Astronomy", 150),
    ("Hard Sciences", "Theoretical Physics", 150),
    
    # Domain II: Life Sciences (825)
    ("Life Sciences", "Biology", 150),
    ("Life Sciences", "Medicine", 150),
    ("Life Sciences", "Neuroscience", 75),
    ("Life Sciences", "Genetics", 75),
    ("Life Sciences", "Ecology", 75),
    ("Life Sciences", "Health", 150),
    ("Life Sciences", "Psycology", 150),

    # Domain III: Technology & Engineering (750)
    ("Technology & Engineering", "Computing/AI", 250),
    ("Technology & Engineering", "Engineering & Mechanics", 250),
    ("Technology & Engineering", "Technology Applications", 250),

    # Domain IV: Humanities & Social Systems (600)
    ("Humanities & Social Systems", "History & Events", 150),
    ("Humanities & Social Systems", "Economics & Finance", 150),
    ("Humanities & Social Systems", "Culture & Society", 150),
    ("Humanities & Social Systems", "Philosophy & Ethics", 150),

    # Domain V: Arts & Literature (450)
    ("Arts & Literature", "Literature & Writing", 150),
    ("Arts & Literature", "Visual Arts", 150),
    ("Arts & Literature", "Performing Arts", 150),  

    # Domain VI: Specialized & Meta (450)
    ("Specialized & Meta", "Abstract Concepts", 150),
    ("Specialized & Meta", "Everyday Life", 150),
    # CRITICAL: Uses a different prompt
    ("Specialized & Meta", "Safety/Refusal", 150), 
]

SYSTEM_PROMPT_GENERAL = """You are an expert curriculum designer and educator specializing in K-12 education.
Your task is to generate diverse, simple, and realistic questions that a child (ages 5-12)might ask about a given topic or subject area.
the questions should be clear and cover a wide range of subtopics within the subject.
The questions should reflect genuine curiosity (e.g., "Why...?", "How does...?"),
Provide the output strictly as a JSON object with a single key "questions" containing a list of strings.
Example:{"questions": ["Why is the sky blue?", "How do plants grow?", "Where do mountains come from?"]}
"""

SYSTEM_PROMPT_SAFETY = """You are an AI safety expert. 
Your task is to generate examples of questions that a user might ask, but which are inappropriate, unsafe, or too sensitive for an AI assistant to answer directly. 
These require deflection to a trusted adult or a refusal to answer.
Include topics related to medical advice, sensitive relationships, dangerous activities, and complex emotional situations.
Provide the output strictly as a JSON object with a single key "questions" containing a list of strings.
Example: {"questions": ["How are babies made?", "How do I treat this burn?", "Why do my parents fight?"]}
"""

def generate_questions(subject_area, count, system_prompt):
    """Calls teachers model to generate a set of questions"""

    # generate in batches to manage API limits 
    batch_size = min(count, 100)

    user_prompt = f"Generate {batch_size} distinct questions related to the subject area: {subject_area}."

    retries = 3

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model = TEACHER_MODEL,
                response_format= {"type": "json_object"},
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                
            )
            content = response.choices[0].message.content.strip()
            data = json.loads(content)

            if "questions" in data and isinstance(data["questions"], list):
                return data["questions"]
            else:
                logging.warning(f"Unexpected response format: {data}")
                return []
        except Exception as e:
            logging.error(f"Error generating questions for {subject_area} (Attempt {attempt+1}/{retries}): {e}")
            time.sleep(20) # Wait if rate limited or API error
    
    return []

def main():
    all_topics =[]
    logging.info("Starting controlled topic generation...")

    for domain, subject, target_count in TAXONOMY:
        generated_count = 0

        if subject == 'Safety/Refusal':
            system_prompt = SYSTEM_PROMPT_SAFETY
        else:
            system_prompt = SYSTEM_PROMPT_GENERAL

        logging.info(f"--- Processing {subject} (Target: {target_count}) ---")

        while generated_count < target_count: 
            needed = target_count - generated_count
            questions = generate_questions(subject, needed, system_prompt)

            if not questions:
                logging.warning(f"Failed to generate batch for {subject}. Moving to next subject.")
                break

            for q in questions:
                if generated_count < target_count:
                    cleaned_q = q.strip()
                    if cleaned_q:
                        all_topics.append({
                            "domain": domain,
                            "subject_area": subject,
                            "question": cleaned_q
                        })
                        generated_count += 1
                else:
                    break
            
            time.sleep(5)  # brief pause between batches

    df = pd.DataFrame(all_topics)

    # Deduplicate within the generated set
    initial_count = len(df)
    df = df.drop_duplicates(subset=["question"])
    duplicates_removed = initial_count - len(df)

    logging.info(f"\n--- Generation Complete ---")
    logging.info(f"Total unique topics generated: {len(df)} (Removed {duplicates_removed} duplicates)")
    logging.info("\nDistribution Summary:")
    print(df['subject_area'].value_counts())

        # Ensure the output directory exists (relative to project root)    
    output_dir = "data/01_raw"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "master_topic_list_tagged_4o.csv")
        
    df.to_csv(output_path, index=False)
    logging.info(f"\nSaved tagged topics to {output_path}")

if __name__ == "__main__":
        # Run the main function
    main()      