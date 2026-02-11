# ELI5 Teacher-Bot: Data-Centric Fine-Tuning of Mistral-7B

## Introduction
This project presents an end-to-end pipeline for fine-tuning a Large Language Model (**Mistral-7B-Instruct-v0.2**) to adopt a specific persona: an "Award-Winning K-12 Educator."

While generic LLMs are capable of simplification, they typically suffer from two extremes: they either hallucinate facts to sound simple (often inventing fake titles or events) or remain too technical for a young audience, providing dry dictionary definitions. Furthermore, standard benchmarks like MMLU are designed for exam performance, not pedagogical clarity.

My system tackles these fundamental challenges by implementing a **Data-Centric AI** approach. Instead of relying solely on hyperparameter tuning, I engineered a custom, iterative dataset generation process. I demonstrated that improving data quality—specifically through an **agentic rewriting pipeline** that enforces analogies—yields significantly higher performance gains than model architecture tweaks. The final model is optimized using **QLoRA** (4-bit quantization + LoRA adapters) and rigorously validated using a novel **Multicall LLM-as-a-Judge** framework.

## Key Features
My system ensures high-quality pedagogical interactions through several advanced features:

*   **Custom "Curious Child" Taxonomy:** A synthetic dataset of 8,000+ examples generated via a custom taxonomy, covering STEM, Humanities, and "Sensitive" topics.
*   **Agentic Data Refinement:** Implements a secondary "Teacher Agent" pipeline that reviews and rewrites complex training examples to enforce the use of vivid analogies and natural flow.
*   **Safety-First Architecture:** Explicitly trained on "refusal" triggers to handle unsafe queries (e.g., medical advice, dangerous acts) by validating inputs before answering.
*   **Efficient QLoRA Training:** Fine-tunes a 7B parameter model on consumer hardware using 4-bit quantization and Low-Rank Adapters (LoRA).
*   **Multicall Evaluation Engine:** A robust evaluation harness that mitigates single-judge bias by performing four distinct assessment calls (Safety, Accuracy, Age-Fit, Analogy) per generated answer.

## Technical Architecture & Implementation
This project was an exercise in rigorous experimental design, moving from a failing prototype to a robust final model through data engineering.

### 1. Data Acquisition: The Pivot from MMLU
My project began with the goal of using the **cais/mmlu** benchmark for training data. However, extensive analysis revealed that MMLU questions were too specific or multiple-choice focused to represent the natural, open-ended curiosity of a child.

**The Solution:** I pivoted to a generative approach. I designed a custom taxonomy of **2,500+ topics** ranging from simple curiosity (*"Why is the sky blue?"*) to complex abstractions (*"Quantum Entanglement"*). I built a script (`02_topic_generation.py`) utilizing GPT-4o to generate unique questions across three complexity tiers: Simple (ELI5), Complex (ELI12), and Safety/Refusal.

### 2. The "Quality Bottleneck" & Agentic Regeneration
After training an initial prototype (Dataset V1), evaluations showed the model was "safe but shallow"—it provided correct dictionary definitions but lacked teaching instinct. It failed to use examples or analogies effectively.

**The Engineering Fix:** I identified that hyperparameter tuning would not solve this semantic gap. Instead, I built a **Regeneration Pipeline** (`06_regenerate_answer...`). This script acted as a "Senior Editor," taking the initial synthetic answers and rewriting them with strict constraints:
*   *Do not use rigid headers.*
*   *Use a natural conversational flow.*
*   *You MUST include a concrete analogy (e.g., 'like a pirate swaying').*

**Result:** This data intervention proved to be the single biggest driver of performance.

**Experiment A: The Data Pivot (Dataset V1 vs V2)**
*Comparison of model behavior before and after regenerating the "Complex" training examples.*

| Feature | Dataset V1 (Old) | Dataset V2 (New/Regenerated) |
| :--- | :--- | :--- |
| **Model** | `Optimized Model` | `RunA / ckpt425` |
| **Analogy Quality** | **Generic.** *"It's like acting out a play."* | **Specific & Vivid.** *"Like a pirate swaying or a princess twirling."* |
| **Avg Judge Score** | ~7.7 - 8.3 | **9.2 - 9.3** |
| **Conclusion** | Safe but shallow. Dictionary definitions. | **True teaching.** Deconstructs concepts with imagery. |

### 3. Model Training (QLoRA & Hyperparameter Sweeps)
I fine-tuned **Mistral-7B-Instruct-v0.2** using the HuggingFace `trl` library with 4-bit NF4 quantization. I ran multiple full-training runs to determine optimal settings.

**Experiment B: Hyperparameter Sweep (RunA vs RunB vs RunC)**
*Testing robustness on the new dataset.*

| Metric | RunA (Constant, r64) | RunB (Cosine, r64) | RunC (Cosine, r32) |
| :--- | :--- | :--- | :--- |
| **Min Val Loss** | 0.821 | 0.803 | 0.813 |
| **Judge Accuracy** | 4.57 / 5.0 | 4.49 / 5.0 | 4.50 / 5.0 |
| **Analogy Score** | 1.81 / 2.0 | 1.73 / 2.0 | 1.72 / 2.0 |
| **Total Score** | **9.30** | 9.20 | 9.20 |

*   **Finding:** While Cosine (RunB) achieved lower *training loss*, Constant (RunA) achieved slightly higher *judge scores*. However, the difference is negligible (<1%).
*   **Verdict:** The model architecture is robust; performance is saturated by the data quality.

### 4. Ablation Study: Volume vs. Depth
To determine if the full dataset was necessary, I performed an ablation study by training a model on a stratified subset of **5,000 examples** (`Run5k`) versus the full **8,000+** dataset.

**Experiment C: Ablation Study (5k vs 8k)**
*Is 8,000 samples necessary?*

| Metric | Run5k (Ablation) | ckpt425 (Full 8k) | Delta |
| :--- | :--- | :--- | :--- |
| **Total Score** | 8.19 | **9.24** | +1.05 |
| **Accuracy** | 3.64 | 4.42 | +0.78 |
| **Avg Word Count** | ~47 words | ~103 words | +119% |

*   **Verdict:** 5k samples are sufficient to learn **Tone** (Age-Fit scores were identical), but insufficient for **Depth** (Accuracy scores dropped and answers became ~50% shorter).

### 5. Evaluation Strategy: Multicall LLM-as-a-Judge
Evaluating generative text is notoriously difficult. My initial `v1` judge (a single 1-10 score) proved too noisy and subjective. I iterated to a `v3` **Multicall System**:
*   **Methodology:** For every answer, the evaluation script makes **4 separate API calls**:
    1.  **Safety Gate:** Classifies the question and checks if the model failed to refuse a harmful prompt.
    2.  **Accuracy Judge (0-5):** Checks factual correctness.
    3.  **Age-Fit Judge (0-3):** Checks if vocabulary matches the ELI5/ELI12 target.
    4.  **Analogy Judge (0-2):** Specifically checks for the presence of helpful metaphors.
*   **Bias Mitigation:** The order of models (Base vs. Tuned A vs. Tuned B) is randomized and seeded for every row to prevent positional bias.

### Qualitative Comparison: "Choreographic Narrative"
*How different models explain the same abstract concept.*

| Model | Output Summary | Score | Verdict |
| :--- | :--- | :--- | :--- |
| **Base Mistral** | *"Title: 'The Seasons' Dance' is a choreographic narrative..."* | **Failure** | Hallucinates a fake play. |
| **Old Data (V1)** | *"It's like when you watch a movie... dancers show the story."* | **8/10** | Safe but generic. Dictionary definition. |
| **New Data (RunA)** | *"Imagine you have a favorite book... like a pirate swaying or a princess twirling... usage of body language..."* | **10/10** | **Winner.** Specific, vivid imagery. Teaches the concept. |
| **5k Ablation** | *"It's like when you make up a dance to your favorite song!"* | **9/10** | Correct tone, but **too short**. Lacks depth. |

## Getting Started
### Prerequisites
*   Python 3.10+
*   GPU with at least 24GB VRAM (recommended for training) or CPU (for GGUF inference)
*   OpenAI API Key (for Data Generation & Evaluation)

### Installation
1.  Clone the repository:
    ```bash
    git clone [REPO_URL]
    cd ELI5-Teacher-Bot
    pip install -r requirements.txt
    ```

### Pipeline Execution
1.  **Data Generation:** 
    ```bash
    python 05_generate_answers.py
    python 06_regenerate_answer_for_complex_questions.py  # Crucial step for quality
    ```
2.  **Training:** 
    ```bash
    python 09_model_training.py
    ```
3.  **Evaluation:** 
    ```bash
    python 12_llm_judge_eval.py
    ```