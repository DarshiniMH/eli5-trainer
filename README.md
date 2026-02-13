# ELI5 Teacher Bot: Fine‑Tuning Mistral‑7B for K–12 Explanations

This project builds an end‑to‑end pipeline to fine‑tune **Mistral‑7B‑Instruct v0.2** into an “Explain Like I’m 5” assistant that explains questions the way an award‑winning K–12 teacher would:

- **Age‑appropriate**: ELI5 (5–8) vs ELI12 (9–12)  
- **Factual + simple** (without turning into a dry dictionary definition)  
- **Uses examples/analogies when helpful**  
- **Safety aware**: refuses / redirects on unsafe or professional‑advice requests  

Training uses **QLoRA** (4‑bit quantization + LoRA adapters) so the pipeline is practical on consumer GPUs / Colab.

---

## Why this project exists

Generic LLMs can “simplify,” but they often fail in two ways:

1) **Too technical**: correct but not accessible for kids  
2) **Over‑simplified**: short and friendly, but missing key nuance  
3) **Safety edge cases**: sensitive questions (medical/safety) require refusal or redirection  

Key learning from this project: **data quality mattered more than hyperparameter tweaking** for teacher‑style behavior. When the tuned model underperformed on complex questions, the biggest improvement came from regenerating and rewriting training data—*not* from repeatedly tuning training knobs.

---

## Key features

- **Custom “Curious Child” taxonomy** to generate a balanced dataset across STEM, humanities, arts/everyday life, and safety/refusal prompts.
- **Prompt-driven data refinement** (data-centric iteration): rewrote the teacher prompt to enforce clearer ELI12 explanations with natural flow and explicit examples.
- **QLoRA fine‑tuning** on Mistral‑7B (4‑bit NF4 + LoRA adapters) to keep training feasible.
- **LLM-as-Judge evaluation evolution (v1 → v2 → v3)**:
  - v1: coarse single-call score
  - v2: structured scoring with targets
  - v3: **multicall** (Safety / Accuracy / Age-fit / Analogy), plus deterministic shuffling to reduce order bias

---

## Technical Architecture and Implementation

### 0) Starting point: MMLU topic extraction (and pivot)
I initially extracted questions from **cais/mmlu** (STEM/humanities/professional subjects).  
Problem: questions were exam‑like and too specific for “what kids actually ask,” so I pivoted to generating my own dataset.

Script: `01_topic_extraction.py`

---

### 1) Building a custom dataset (taxonomy → topics → merged inputs)
I generated:
- **Simple curiosity questions** (“Why…?”, “How…?”)
- **Complex concepts** requiring ELI12 explanations
- **Safety/refusal prompts**

I generated topics using both **GPT‑4o** and **GPT‑4o‑mini** (to compare diversity), then merged + deduped.

Scripts:
- `02_topic_generation.py`
- `03_generated_topic_analysis.py`
- `04_merge_topics.py`

---

### 2) Teacher-style answer generation (Dataset v1)
I used a teacher persona prompt to generate training targets in JSON:
- `internal_reflection`
- `explanation`

Script: `05_generate_answers.py`

---

### 3) Prototype training (~1000 samples) to validate the loop
Before spending hours on full training, I ran a prototype (~900 train / 100 val). Loss decreased and outputs qualitatively changed → green light for full training.

Prototype metrics:

| Step | Train Loss | Val Loss | Entropy | Num Tokens | Mean Token Acc |
|---:|---:|---:|---:|---:|---:|
| 25 | 1.336939 | 1.111256 | 1.176616 | 47055 | 0.690173 |
| 50 | 0.932878 | 0.933495 | 1.027300 | 93176 | 0.711735 |

---

### 4) Full training on Dataset v1 + early evaluation (Judge v1 & v2)
I trained on the full dataset (~7379 train / 820 val) and compared two configurations:

**Dataset v1 — full-run configs**

| Run Name | Scheduler | LoRA Rank | Batch Size | Min Val Step | Min Val Loss | Last Step | Last Val Loss |
|---|---|---:|---:|---:|---:|---:|---:|
| `unstable_full_const_r64_bs16` | constant | 64 | 16 | 425 | 0.819623 | 450 | 0.863269 |
| `optimized_full_cosine_r32_bs32` | cosine | 32 | 32 | 200 | 0.815179 | 200 | 0.815179 |

I evaluated these tuned models using two judge versions:

- **LLM-as-Judge v1** (coarse single call, 1–10 score + rationale)
- **LLM-as-Judge v2** (structured single call: classification + targets + subscores)

Key learning: even when loss improved, the tuned model still struggled on **complex (ELI12) questions** → this looked like a **data quality** problem, not a pure hyperparameter problem.

---

### 5) Data iteration: rewrite complex answers (Dataset v2)
Because complex questions were weak, I regenerated/re‑wrote the complex subset with a new prompt enforcing:

- **Natural flow** (no rigid headings)
- **ELI12 depth**
- **At least one example/analogy**
- Safety/refusal rows were preserved (not rewritten)

Script: `06_regenerate_answers_for_complex_questions.py`

---

### 6) Dataset v2 training sweeps + final evaluation framework (Judge v3 multicall)
For Dataset v2, I switched to **LLM-as-Judge v3 multicall**:

- Safety gate: classification + per-answer violations  
- Accuracy (0–5)  
- Age-fit (0–3)  
- Analogy quality (0–2)  
- **Total = Accuracy + Age-fit + Analogy (0–10)**  
- Deterministic shuffling to reduce position bias

---

### 7) 5k ablation (Dataset v2): do we need ~8k?
I trained with **5k stratified samples** to test if dataset size mattered.

Result: **5k produced shorter, simpler answers but lower overall accuracy**, while the full dataset (~8k) produced better overall results.

---

## Training setup (QLoRA)

- Base model: `mistralai/Mistral-7B-Instruct-v0.2`
- Quantization: 4‑bit NF4 (`bitsandbytes`)
- LoRA adapters: target modules  
  `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Trainer: `trl.SFTTrainer` + `SFTConfig`
- Max sequence length: 1024
- BF16 compute (when supported)

---

## Experiments & results (most important section)

### A) Dataset v2 — Multicall evaluation (v3) summaries

#### 1) Best checkpoint vs final weights (same training run)
Evaluation set: **base vs ckpt425 vs final** (Dataset v2, `case1_lr2e4_const_r64_a16`)

| Model | Mean Accuracy (0–5) | Mean Age-fit (0–3) | Mean Analogy (0–2) | **Mean Total (0–10)** |
|---|---:|---:|---:|---:|
| Base | 4.544 | 1.002 | 1.170 | **6.716** |
| Final adapter (last weights) | 3.930 | 2.963 | 1.813 | **8.706** |
| **Checkpoint‑425** | 4.419 | 2.990 | 1.817 | **9.226** |

**Takeaway:** best checkpoint beat final weights → supports using best-checkpoint saving (`load_best_model_at_end=True`) and/or selecting the best checkpoint for inference.

---

#### 2) Best sweep vs best checkpoint (head-to-head)
Evaluation set: **base vs ckpt425 vs runA**

| Model | Mean Total (0–10) |
|---|---:|
| Base | 6.693 |
| **ckpt425** | **9.167** |
| **runA** | **9.153** |

**Takeaway:** runA and ckpt425 are effectively tied (differences are tiny).

---

#### 3) Scheduler & rank sweep comparison
Evaluation set: **runA vs runB vs runC** (all Dataset v2)

| Run | LR | Scheduler | r | α | Min Val Loss (step) | Mean Total (0–10) |
|---|---:|---|---:|---:|---:|---:|
| **runA** | 1e‑4 | constant | 64 | 16 | 0.8214 (425) | **9.305** |
| runB | 1e‑4 | cosine | 64 | 16 | **0.8032 (400)** | 9.195 |
| runC | 1e‑4 | cosine | 32 | 8 | 0.8137 (400) | 9.203 |

**Takeaway:** cosine achieved the lowest eval loss, but **did not translate into better teacher-style judge scores**.

---

#### 4) 5k ablation vs full (~8k)
Evaluation set: **base vs ckpt425 vs run5k**

| Model | Mean Total (0–10) |
|---|---:|
| Base | 6.779 |
| **ckpt425 (full ~8k)** | **9.244** |
| run5k | 8.189 |

**Takeaway:** 5k keeps teacher tone but **full data improves accuracy and completeness**.

---

## Qualitative examples (from evaluation logs)

<details>
<summary><strong>Example 1 — “Chronic disease management”</strong></summary>

| Model | v3 Total | Output (unedited) |
|---|---:|---|
| Base | 7/10 | (Long, technical explanation) |
| Run5k | 9/10 | “Chronic disease management is like taking care of a garden…” (simple but shorter) |
| **ckpt425** | **10/10** | Clear definition + daily habits + analogy + accurate framing |

</details>

<details>
<summary><strong>Example 2 — “Crowdfunding”</strong></summary>

| Model | v3 Total | Output (unedited) |
|---|---:|---|
| Base | 6/10 | Correct but verbose/technical |
| ckpt425 | 9/10 | Clear analogy + simple explanation |
| **runA** | **10/10** | Clear definition + simple steps + strong structure |

</details>

---

## How to run the pipeline

### Prerequisites
- Python 3.10+
- GPU recommended for QLoRA training
- OpenAI API key (dataset generation + judge evaluation)

### Install
```bash
git clone https://github.com/DarshiniMH/ELI5-Teacher-Bot.git
cd ELI5-Teacher-Bot
pip install -r requirements.txt



## Pipeline Execution

Follow these steps to reproduce the dataset generation, training, and evaluation process.

### 1. Data Generation
Generate the initial synthetic dataset and refine complex examples for higher quality.

```bash
# Generate initial dataset from source topics
python 05_generate_answers.py

# Refine complex questions (Crucial step for quality)
python 06_regenerate_answer_for_complex_questions.py