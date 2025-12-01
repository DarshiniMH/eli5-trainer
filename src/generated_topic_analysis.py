import pandas as pd

file_mini = "data/01_raw/master_topic_list_tagged_4o_mini.csv"
file_4o = "data/01_raw/master_topic_list_tagged_4o.csv"

try:
    df_mini = pd.read_csv(file_mini)
    df_4o = pd.read_csv(file_4o)

except FileNotFoundError as e:
    print(f"Error: CSV file not found in the current directory.")
    exit()

print(f"Mini shape: {df_mini.shape}")
print(f"4o shape:{df_4o.shape}\n")

set_mini = set(df_mini['question'])
set_4o = set(df_4o['question'])

overlap = set_mini.intersection(set_4o)

overlap_count = len(overlap)
union_count = len(set_mini.union(set_4o))

print(f"Total Unique Questions (if combined): {union_count}")
print(f"Number of Identical Questions (Overlap): {overlap_count}")
print(f"Overlap Percentage (vs Mini): {overlap_count / len(set_mini) * 100:.2f}%")

def compare_samples(subject, num_samples = 10):
    print(f"\n---- comparing samples: {subject}-----\n")

    try:
        samples_4o = df_4o[df_4o["subject_area"]==subject].sample(n = num_samples, random_state=42)
        samples_mini = df_mini[df_mini["subject_area"]==subject].sample(n = num_samples, random_state = 42)
    except ValueError as e:
        print(f"Not enough samples for subject {subject}. Skipping sample comparison.")
        samples_4o = df_4o[df_4o["subject_area"]==subject]
        samples_mini = df_mini[df_mini["subject_area"]==subject]
    
    comparision_df = pd.DataFrame({
        "GPT-4o-mini": samples_mini["question"].reset_index(drop = True),
        "GPT-4o": samples_4o["question"].reset_index(drop = True)
    })

    pd.set_option('display.max_colwidth', None)
    print(comparision_df)
    #display(comparision_df)

compare_samples("Math/Logic")
compare_samples("Physics")
compare_samples("Genetics")
compare_samples("Safety/Refusal")


