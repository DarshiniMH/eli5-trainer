# ELI5: Fine‑Tuning Mistral‑7B for Simple Explanations

An end‑to‑end, **data‑centric** pipeline to fine‑tune **Mistral‑7B‑Instruct v0.2** to explain any question in simple terms, effectively acting as an "Explain Like I'm 5" (ELI5):

- **Age‑appropriate**: ELI5 / ELI12 style (simple language without becoming wrong)
- **Accurate**: avoids over‑simplifying into misinformation
- **Concrete**: uses examples/analogies when helpful
- **Safety‑aware**: learns refusal/redirection for sensitive or professional‑advice requests

> Core thesis: **dataset quality (prompting + targets) mattered more than small hyperparameter tweaks.**

---

## Key features

- **Custom taxonomy + synthetic dataset (8k+)** spanning STEM, humanities, arts/everyday life, and safety/refusal prompts.
- **Data‑centric iteration**: regenerated and rewrote **complex (ELI12)** targets to improve natural flow + examples (Dataset v1 → Dataset v2).
- **Efficient QLoRA training**: 4‑bit NF4 quantization + LoRA adapters to fine‑tune a 7B model efficiently.
- **Evaluation evolution (v1 → v2 → v3)**:
  - **v1**: coarse single‑call scoring
  - **v2**: structured single‑call scoring with target level/strategy
  - **v3 (final)**: **multicall** judge (Safety / Accuracy / Age‑fit / Analogy) + deterministic shuffling to reduce order bias

---

## Project evolution (what changed and why)

### 0) Pivot: MMLU → child‑curiosity data
I started by extracting topics from **cais/mmlu**, but those questions were **exam‑style** and too specific for “what kids actually ask.”  
So I pivoted to a custom taxonomy and generated my own dataset.

Script: `01_topic_extraction.py`

### 1) Build dataset from taxonomy
Generated:
- Simple curiosity questions (“Why…?”, “How…?”)
- Complex concepts requiring ELI12 explanations
- Safety/refusal prompts

Merged + deduped topics (generated via GPT‑4o and GPT‑4o‑mini).

Scripts:
- `02_topic_generation.py`
- `03_generated_topic_analysis.py`
- `04_merge_topics.py`

### 2) Generate teacher‑style answers (Dataset v1)
Generated targets with a teacher persona prompt.

Script: `05_generate_answers.py`

### 3) Prototype training (sanity check, ~1000 samples)
Before committing to full training time, I ran a prototype (~900 train / 100 val). Loss dropped and outputs visibly changed → proceed to full training.

Prototype metrics:

| Step | Train Loss | Val Loss | Entropy | Num Tokens | Mean Token Acc |
|---:|---:|---:|---:|---:|---:|
| 25 | 1.336939 | 1.111256 | 1.176616 | 47055 | 0.690173 |
| 50 | 0.932878 | 0.933495 | 1.027300 | 93176 | 0.711735 |

### 4) Full training on Dataset v1 + early evaluation (Judge v1 & v2)
Trained on full dataset (~7379 train / 820 val) and compared two configs.  
Even with improved loss, **complex (ELI12) answers were not strong** → sign that **targets/data needed improvement**.

Dataset v1 training summary:

| Run | Scheduler | LoRA r | Batch size | Min val step | Min val loss | Last step | Last val loss |
|---|---|---:|---:|---:|---:|---:|---:|
| unstable_full_const_r64_bs16 | constant | 64 | 16 | 425 | 0.819623 | 450 | 0.863269 |
| optimized_full_cosine_r32_bs32 | cosine | 32 | 32 | 200 | 0.815179 | 200 | 0.815179 |

### 5) Dataset iteration: rewrite complex answers (Dataset v2)
To fix weak complex examples, I regenerated/re‑wrote the **complex subset** with constraints:
- natural flow (no rigid headings)
- ELI12 depth
- at least one example/analogy
- safety/refusal rows preserved

Script: `06_regenerate_answer_for_complex_questions.py`

### 6) Dataset v2 sweeps + final evaluation (Judge v3 multicall)
Final evaluation uses **4 separate judge calls** per example:
- Safety gate (classification + violations)
- Accuracy (0–5)
- Age‑fit (0–3)
- Analogy/example quality (0–2)

Total score for **Normal** questions = Accuracy + Age‑fit + Analogy (0–10)

---

## Training setup (QLoRA)

- Base model: `mistralai/Mistral-7B-Instruct-v0.2`
- Quantization: 4‑bit NF4 (bitsandbytes)
- LoRA target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Trainer: `trl.SFTTrainer` (`SFTConfig`)
- Sequence length: 1024
- BF16 compute (when supported)

---

## Results

### A) Quantitative (by judge numbers — v3 multicall, Normal questions only)

**v3 Multicall summary**

| Comparison set | N (Normal) | Mean Total (0–10) | Winner by mean | Tie rate |
|---|---:|---:|---|---:|
| base vs ckpt425 vs final | 805 | base **6.72**, ckpt425 **9.23**, final **8.71** | **ckpt425** | 45.0% |
| runA vs runB vs runC | 804 | runA **9.30**, runB **9.20**, runC **9.20** | **runA (tiny edge)** | **65.4%** |
| base vs ckpt425 vs runA | 802 | base **6.69**, ckpt425 **9.17**, runA **9.15** | **ckpt425 (tiny edge)** | **61.1%** |
| base vs ckpt425 vs run5k | 804 | base **6.78**, ckpt425 **9.24**, run5k **8.19** | **ckpt425** | 26.7% |

**What this means (numbers-only):**
- **Big gaps**: ckpt425 ≫ base, and full‑data (8k) ≫ run5k.
- **Small gaps**: runA/runB/runC/ckpt425 are all **within ~0.1** mean score and have **many ties** → basically the same tier.

### B) Qualitative (by reading answers)
Across **Dataset v2**, the “best” configs (**ckpt425, runA, runB, runC**) produce **very similar quality**. Differences are usually minor (wording/formatting/detail level).

The **largest visible improvement** came from:
1) **Dataset v1 → Dataset v2** (rewriting complex targets)  
2) **8k+ data → 5k** (5k answers were often shorter and less complete)

#### Example: same config, different datasets (clear data effect)
**Q: “what is choreographic narrative”**

<details>
<summary><strong>Base Mistral (untuned)</strong></summary>

> A choreographic narrative is a story told through dance. It is a sequence of movements and actions that convey a specific theme, idea, or emotion. The narrative can be abstract or representational, and it can be told through various dance genres, such as ballet, contemporary, or hip hop.
> The choreographer creates the narrative by designing the movements, music, costumes, and set design to convey the story. Dancers then perform these elements in a coordinated and expressive way to bring the narrative to life.
> Choreographic narratives can be based on a wide range of sources, including literature, mythology, history, or personal experiences. They can be used to explore complex themes, such as love, loss, or identity, or to entertain and delight audiences.
> Effective choreographic narratives require a strong sense of storytelling and a deep understanding of the dance medium. The choreographer must be able to convey the narrative through movement and expression, rather than relying on dialogue or spoken words. The dancers must also be able to embody the characters and emotions of the narrative, bringing it to life in a compelling and engaging way.  
> Overall, a choreographic narrative is.
> (Formal + comprehensive, but loses the simple tone)
</details>

<details>
<summary><strong>Dataset v1 tuned (LR=2e‑4, constant, r=64, α=16)</strong></summary>

> Imagine you're watching a dance performance. The dancers are moving around the stage to tell a story. This is called a 'choreographic narrative.' It's like when you watch a movie, and you see the characters acting out a story. In dance, the dancers use their movements to show the story or feelings, just like actors do in a movie.
> (Simplistic + relatable, but lacks depth)
</details>

<details>
<summary><strong>Dataset v2 tuned (same config)</strong></summary>

> Choreographic narrative is a way of telling a story through dance. Imagine dance as a special language that uses movement to express feelings and ideas. Just like a storybook has words and pictures, a dance can have steps and music. When dancers perform, they use their bodies to show what is happening in the story. For example, if the story is about a journey, the dancers might move in ways that show traveling, like running or skipping. If the story is about love, the dancers might move close together or twirl around each other. The music helps tell the story too, just like a soundtrack in a movie. So, choreographic narrative is all about using dance to make the story come alive and show what is happening in a fun and creative way.
> (more complete + clearer + grounded)

</details>

<details>
<summary><strong>Dataset v2 — 5k ablation (same config)</strong></summary>

> Choreographic narrative is like telling a story through dance. Imagine when you act out a scene from a book or a movie. You might use your body to show how a character feels or what they are doing. In dance, choreographers use movements to tell a story or show an idea. It's like when you make up a dance to your favorite song! 
> (teacher tone preserved, but noticeably shorter)

</details>

---

## Key takeaways (final)

1) **Dataset quality drove the biggest gains.**  
   Rewriting complex targets (Dataset v2) produced the largest “step change” in answer quality.

2) **Hyperparameter sweeps were mostly ties in practice.**  
   **runA/runB/runC/ckpt425** are extremely close. If choosing strictly by numbers: **ckpt425 or runA**.

3) **Best checkpoint can beat final weights.**  
   In the constant‑LR run, **checkpoint ~425** aligned best with judged quality.

4) **5k vs 8k matters.**  
   5k kept the teacher tone but produced **shorter / less complete** answers and lower judged scores.

---

## How to run the pipeline

### Prerequisites
- Python 3.10+
- GPU recommended for QLoRA training
- `OPENAI_API_KEY` (dataset generation + evaluation)

### Install
```bash
pip install -r requirements.txt

### Pipeline Execution
'''bash
# Data generation
python 05_generate_answers.py
python 06_regenerate_answer_for_complex_questions.py   # crucial for quality

# Training
python 09_model_training.py

# Generate answers (for evaluation)
python 10_generate_tuned_model_outputs.py

# Evaluation (LLM-as-Judge)
python 12_llm_judge_eval.py