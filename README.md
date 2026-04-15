# Detecting Hallucinations via Open-Weight Proxy Analyzer Activations

**Akshita Singh · Prabesh Paudel · Siddhartha Roy**  
Khoury College of Computer Sciences, Northeastern University  
`{singh.akshita, paudel.pr, roy.sidd}@northeastern.edu`

---

## What this project does

We built a hallucination detection system that works without any access to the model
that generated the text. Given a source document, a question, and a candidate answer,
our proxy-analyzer reads the text through a small locally hosted open-weight model,
intercepts its internal activations, and decides whether the answer is likely
hallucinated.

The key idea is simple: hallucination is a property of the relationship between source
and answer, not a quirk of the generating model. Any transformer reading a contradictory
sentence against its source will show altered attention patterns and stronger memory
activations, regardless of which model originally produced that sentence. We exploit
exactly those signals.

We extract eighteen features from a single forward pass, train a stacking ensemble
classifier, and evaluate across seven analyzer architectures from 0.5 billion to 9
billion parameters. On the RAGTruth benchmark, every one of our configurations beats
ReDeEP's token-level AUC of 0.73 by between 7.4 and 10.3 percentage points. Our
Qwen2.5-7B setup reaches an F1 of 0.717, the first proxy-analyzer to exceed ReDeEP's
token-level F1 of 0.713. One of our most striking findings is that our 3B LLaMA
outperforms the 8B LLaMA on RAGTruth, showing that bigger is not always better even
within the same model family.

---

## Repository structure

```
.
├── notebooks/
│   ├── qwen25_05b/
│   │   ├── hallucination_detection_qwen25_05b.ipynb
│   │   └── features_qwen25_05b/
│   ├── pythia1b/
│   │   ├── hallucination_detection_pythia1b.ipynb
│   │   └── features_pythia1b/
│   ├── gemma2_2b/
│   │   ├── hallucination_detection_gemma2_2b.ipynb
│   │   └── features_gemma2_2b/
│   ├── llama3_3b/
│   │   ├── hallucination_detection_llama3_3b.ipynb
│   │   └── features_llama3_3b/
│   ├── qwen25_7b/
│   │   ├── hallucination_detection_qwen25_7b.ipynb
│   │   └── features_qwen25_7b/
│   ├── llama3_8b/
│   │   ├── hallucination_detection_llama3_8b.ipynb
│   │   └── features_llama3_8b/
│   └── gemma2_9b/
│       ├── hallucination_detection_gemma2_9b.ipynb
│       └── features_gemma2_9b/
├── requirements.txt
└── README.md
```

Each notebook follows the same eight-section pipeline. **Sections 1 to 4 differ per
model** because they handle environment setup, model loading, and early feature
prototyping. **Sections 5 to 8 are identical across all notebooks** and cover the full
production extraction, classifier training, calibration, and evaluation pipeline.

---

## The eight-section pipeline

### Sections 1 to 4 — model-specific setup (different per notebook)

These four sections are the only parts that change between notebooks. Each model has
its own version of these cells. If you are adapting this pipeline for a new model,
these are the only sections you need to edit.

---

#### Section 1: Environment setup

Section 1 imports all required libraries and sets the two variables that control where
everything gets saved. This is the only place you need to change the model name.

```python
import os, json, warnings
import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer
from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import login

# Some models are gated — paste your HuggingFace token here
login(token="your_hf_token_here")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")  # must say 'cuda' before continuing

# Change these two lines per model
OUTPUT_DIR   = "features_llama3_8b"
MODEL_PREFIX = "llama3_8b"

os.makedirs(OUTPUT_DIR, exist_ok=True)
```

The `MODEL_PREFIX` variable controls all downstream file names. Every artifact saved
by Sections 5 through 8 uses this prefix, so each model's outputs go into its own
folder and never overwrite each other. Always confirm the device is `cuda` before
continuing. CPU-only runs will work but extraction will be extremely slow.

---

#### Section 2: Model loading

This section loads the actual model using TransformerLens. The HuggingFace model ID,
dtype, and expected configuration differ per notebook. Below is what each model looks
like after loading:

| Model | HuggingFace ID | Layers | Heads | d\_model | Approx VRAM |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | `Qwen/Qwen2.5-0.5B` | 24 | 14 | 896 | ~1.2 GB |
| Pythia-1.4B | `EleutherAI/pythia-1.4b` | 24 | 16 | 2048 | ~3.2 GB |
| Gemma-2-2B | `google/gemma-2-2b` | 26 | 8 | 2304 | ~5.0 GB |
| LLaMA-3-3B | `meta-llama/Llama-3.2-3B` | 28 | 24 | 3072 | ~7.5 GB |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B` | 28 | 28 | 3584 | ~15 GB |
| LLaMA-3-8B | `meta-llama/Meta-Llama-3-8B` | 32 | 32 | 4096 | ~18 GB |
| Gemma-2-9B | `google/gemma-2-9b` | 42 | 16 | 3584 | ~20 GB |

All models are loaded in float16 via `HookedTransformer.from_pretrained`. After
loading, each notebook runs a quick cache key check to confirm that `resid_post`,
`pattern`, and `mlp_out` hooks are accessible before any extraction begins.

```python
# Example for LLaMA-3-8B — change the model ID per notebook
model = HookedTransformer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    dtype=torch.float16
)
model = model.to(device)
model.eval()

# Always verify hooks before starting extraction
dummy = model.to_tokens("The quick brown fox")
with torch.no_grad():
    _, cache = model.run_with_cache(
        dummy,
        names_filter=lambda n: any(
            k in n for k in ["resid_post", "pattern", "mlp_out"]
        )
    )

has_resid   = any("resid_post" in k for k in cache.keys())
has_pattern = any("pattern"    in k for k in cache.keys())
has_mlp     = any("mlp_out"    in k for k in cache.keys())

print(f"resid_post : {has_resid}")    # must be True
print(f"pattern    : {has_pattern}")  # must be True
print(f"mlp_out    : {has_mlp}")      # must be True
```

This verification step matters. If any cache key comes back False, the extraction will
silently produce wrong feature vectors. Always confirm all three are True before
running on the full dataset. Also verify the feature vector size matches the formula
`NL + 2 * NL * NH + NL + 4` for the early five-signal version in Section 4, and
`2 * NL * (1 + NH) + 19` for the full eighteen-signal version in Section 5.

---

#### Section 3: Dataset loading

This section loads the training datasets and combines them into one unified dataframe.
The datasets and their sources are the same across all notebooks. Every item goes
through its own parsing logic to produce a shared schema of
`(dataset, source_doc, question, answer, label)` where label 0 is faithful and label
1 is hallucinated.

```python
# HaluEval QA — entity substitution hallucination, balanced 50/50
raw_halueval = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

# RAGTruth — faithfulness failure across 6 LLMs, QA + summary + data2txt
raw_ragtruth = load_dataset("wandb/RAGTruth-processed", split="train")

# MedHallu — medical confabulation, all difficulty tiers
raw_medhallu = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_labeled",  split="train")
raw_medhallu_art = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_artificial", split="train")

# MiniCheck-Synthetic — claim verification
raw_minicheck = load_dataset(...)

# ANLI — human-adversarial NLI, three rounds combined
# Note: ANLI is human-annotated, NOT multi-LLM generated
raw_anli_r1 = load_dataset("facebook/anli", split="train_r1")
raw_anli_r2 = load_dataset("facebook/anli", split="train_r2")
raw_anli_r3 = load_dataset("facebook/anli", split="train_r3")
```

A few things worth knowing about how we parse each dataset:

**HaluEval:** Each item produces two rows. The `answer` field becomes label 0
(faithful) and the `hallucination` field becomes label 1. The source and question are
shared. This gives a perfectly balanced 10,000 faithful and 10,000 hallucinated rows.

**RAGTruth:** The label comes from the `hallucination_labels_processed` dictionary. A
response is marked hallucinated if `evident_conflict == 1` or `baseless_info == 1`.
RAGTruth is the only dataset generated by multiple LLMs (GPT-4, GPT-3.5, Mistral-7B,
Llama-2 in three sizes), which is why we train a separate specialist classifier on it.

**MedHallu:** We include all difficulty tiers (easy, medium, hard) from both the
pqa_labeled and pqa_artificial configurations. A deployed guard must handle all
difficulty levels, so filtering to easy examples only would give an overly optimistic
picture.

**ANLI:** This dataset is human-annotated adversarial NLI, not generated by any LLM.
We treat entailment as faithful (label 0) and contradiction as hallucinated (label 1).
We include it because it provides hard claim-verification examples that complement the
softer overlap between MiniCheck and RAGTruth.

All combined rows are shuffled with `random_state=42` before saving. This same seed is
reused in every train/val/test split throughout the pipeline to ensure full
reproducibility across runs.

---

#### Section 4: Prompt construction and early feature extraction

This section defines the core functions that the rest of the pipeline builds on. It
also contains an early five-signal prototype that we used to validate the approach
before building the full eighteen-signal pipeline in Section 5.

**Prompt builder.** Every (source, question, answer) triple gets formatted into a
structured text prompt before the model sees it:

```python
def build_prompt(source_doc, question, answer):
    src = source_doc.strip()[:400]
    ans = answer.strip()[:200]
    q   = question.strip()
    if src and q:
        return f"Document: {src}\n\nQuestion: {q}\n\nAnswer: {ans}"
    if src:
        return f"Document: {src}\n\nAnswer: {ans}"
    return f"Question: {q}\n\nAnswer: {ans}"
```

The explicit `Document:` / `Question:` / `Answer:` labels are intentional. They
create a clear structural boundary between source tokens and answer tokens in the
attention matrix, which makes Signal 2 (source-document attention) interpretable as a
genuine grounding measure rather than a positional artifact.

**Source token range finder.** We need to know where the source document tokens sit in
the tokenized prompt so we can measure how much the answer attends to them:

```python
def get_source_token_range(source_doc, question):
    src = source_doc.strip()[:400]
    if not src:
        return (0, 0)
    prefix_tokens = model.to_tokens("Document: ", prepend_bos=True)
    src_start = prefix_tokens.shape[1]
    src_tokens = model.to_tokens(src, prepend_bos=False)
    src_end = min(src_start + src_tokens.shape[1], src_start + 80)
    return (src_start, src_end)
```

**Early feature extractor (five signals).** This runs one forward pass and pulls out
the five core activation signals. The `names_filter` argument is critical for large
models: without it, caching all activations for a 32-layer 8B model uses around 60
percent more VRAM and causes out-of-memory errors on a standard A100.

```python
def extract_features(source_doc, question, answer):
    NL = model.cfg.n_layers
    NH = model.cfg.n_heads
    prompt   = build_prompt(source_doc, question, answer)
    src_s, src_e = get_source_token_range(source_doc, question)
    tokens   = model.to_tokens(prompt)
    if tokens.shape[1] > 512:
        tokens = tokens[:, :512]
        src_e  = min(src_e, 512)

    with torch.no_grad():
        logits, cache = model.run_with_cache(
            tokens,
            names_filter=lambda n: any(
                k in n for k in ["resid_post", "pattern", "mlp_out"]
            )
        )

    features = []
    # Signal 1: residual stream norms           (NL values)
    # Signal 2: source-doc attention scores     (NL x NH values)
    # Signal 3: attention head entropy          (NL x NH values)
    # Signal 4: MLP output norms                (NL values)
    # Signal 5: logit-lens commitment summary   (4 values: early, mid, late, var)
    ...
    return np.array(features, dtype=np.float32)
    # total size: NL + 2*NL*NH + NL + 4
```

For LLaMA-3-8B with 32 layers and 32 heads, this gives a vector of size
`32 + 2*32*32 + 32 + 4 = 2116`.

At the end of Section 4 the notebook produces a side-by-side visualization of all
five signals for a faithful versus hallucinated pair. This is saved as
`{MODEL_PREFIX}_feature_preview_all5.png`. It is a useful sanity check: you should
see faithful examples building steadily increasing residual norms and higher source
attention, while hallucinated examples plateau early and collapse source attention.

**Note:** The five signals in Section 4 are a prototype. The production extraction in
Section 5 expands to eighteen signals and adds AHI weighting, orthogonalization,
regime detection, and an external faithfulness score. Section 4 is kept in the
notebooks as a readable introduction to the approach, but the actual paper numbers
all come from Section 5 onward.

---

### Section 5: Feature extraction (identical across all notebooks)

This is the most compute-intensive step and the one that actually produces the feature
vectors used for training. We run one forward pass per sample through the analyzer
using TransformerLens hooks that cache `resid_post`, `pattern`, and `mlp_out`.

From these we compute eighteen signals. The most important ones are:

- **S2 (source-document attention):** mean attention from all answer token positions
  to source token positions, resolved per head and per layer. This alone accounts for
  about 50 percent of Random Forest feature importance.
- **S3 (attention entropy):** how focused or diffuse each head's attention is.
- **S4 (MLP norms):** how strongly parametric memory fires, similar to ReDeEP's PKS.
- **S8 (external faithfulness):** scored by Vectara HHEM-2.1-Open in a separate
  batched pass, verified for correct direction on every model before use.
- **S16 to S18 (token-level grounding):** minimum, variance, and slope of per-token
  source attention. Computed for free inside the S2 loop.

We also extract features for LLM-AggreFact at `src_max=3000` characters, compared to
the `src_max=1200` used for training. This difference is the main cause of the
out-of-distribution performance gap and is tracked as a known limitation.

**Runtime per model on a single A100:** roughly 4 to 6 GPU-hours for the full 72,135
training rows.

---

### Section 6: Augmentation, scaling, and classifier training (identical across all notebooks)

This section builds the full feature matrix and trains the classifiers. It:

1. Reconstructs the 70/15/15 stratified split using `random_state=42`
2. Selects the FIXED\_WINDOW by maximizing per-layer S2 AUC on RAGTruth training rows
   using 3-fold cross-validation
3. Computes Attention Head Importance weights from RAGTruth training labels only
4. Orthogonalizes S13 against S6, and S14/S17/S18 against the S2 mean
5. Trains five classifiers: Logistic Regression, Random Forest, HistGradientBoosting,
   XGBoost, and a Stacking meta-learner
6. Trains a specialist, RagtStacking, on RAGTruth training rows only
7. Runs the AggreFact OOD diagnostic as an informational check only

All scaler fitting and orthogonalization coefficients use training rows only. No
validation or test data leaks into this section.

---

### Section 7: Calibration (identical across all notebooks)

This section fits a three-tier isotonic calibration using validation data only:

- **QA regime:** fitted on HaluEval validation rows. RAGTruth is excluded because the
  Kolmogorov-Smirnov distance between the two distributions exceeds 0.45 across all
  models.
- **Claim regime:** fitted on MiniCheck-Synthetic and ANLI validation rows.
- **Global:** applied to everything else including RAGTruth.

Temperature is set to `T=2.0` across all models, chosen during the Gemma-2-9B run and
kept fixed for consistency. All calibration parameters are fitted on validation data
and applied to the test set without re-fitting.

---

### Section 8: Final evaluation (identical across all notebooks)

This section produces all results and saves seven plots (8A through 8G). It also
prints ten textual analyses (T1 through T10) to the console, covering AUC tables, F1
tables, balanced accuracy tables, per-dataset winner decisions, calibration gain,
per-generator RAGTruth breakdown, AggreFact per-subdataset, signal stability, layer
analysis, and a final clean summary of all paper numbers.

The plots saved are:

- **8A:** ROC and PR curves for all four systems on all datasets
- **8B:** Per-generator RAGTruth AUC and AggreFact OOD breakdown
- **8C:** Per-layer signal discriminability heatmap by task regime
- **8D:** Signal stability scatter showing which signals hold up out-of-distribution
- **8E:** AggreFact detailed per-subdataset results
- **8F:** Error analysis with confusion matrix breakdowns
- **8G:** Length diagnostics and complete summary table

---

## Datasets

We train on five publicly available datasets totalling 72,135 rows.

| Dataset | Rows | Hall% | Task type | Generator |
|---|---|---|---|---|
| HaluEval QA | 19,971 | 50% | Entity substitution | GPT-3.5 |
| RAGTruth | 15,090 | 44.5% | Faithfulness (RAG) | GPT-4, GPT-3.5, Mistral-7B, Llama-2 |
| MedHallu | 10,000 | 50% | Medical confabulation | GPT-4 |
| MiniCheck-Synthetic | 7,076 | 42% | Claim verification | Synthetic |
| ANLI | 19,998 | 50% | Adversarial NLI | Human annotated |

For out-of-distribution evaluation we use **LLM-AggreFact** (`lytang/LLM-AggreFact`).
We remove 16,371 RAGTruth-overlapping rows and one contaminated row, and invert labels
to match our faithful=0 convention. The resulting 12,948 rows span ten sub-tasks. This
dataset is never used in training or calibration.

---

## Analyzer models

| Model | Parameters | Layers | Heads | Feature dim | FIXED\_WINDOW |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.5B | 24 | 14 | 743 | 6--12 |
| Pythia-1.4B | 1.4B | 24 | 16 | 839 | 8--14 |
| Gemma-2-2B | 2B | 26 | 8 | 491 | 5--11 |
| LLaMA-3-3B | 3B | 28 | 24 | 1,419 | 7--13 |
| Qwen2.5-7B | 7B | 28 | 28 | 1,647 | 2--8 |
| LLaMA-3-8B | 8B | 32 | 32 | 2,135 | 7--13 |
| Gemma-2-9B | 9B | 42 | 16 | 1,451 | 9--15 |

Feature dimension follows `2 * N_layers * (1 + N_heads) + 19`.

---

## Setup

### 1. Install PyTorch with CUDA 12.8 first

```bash
pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu128
```

Run this before everything else. The CUDA index URL cannot go in requirements.txt.

### 2. Install all other dependencies

```bash
pip install -r requirements.txt
```

### 3. Optional: LaTeX tools for notebook PDF rendering

```bash
apt-get install -y texlive-xetex texlive-fonts-recommended texlive-plain-generic pandoc
```

### 4. Set your HuggingFace token

```bash
export HF_TOKEN=your_token_here
```

Or paste it directly into Section 1 of each notebook. You need HuggingFace approval
for the LLaMA and Gemma models. Qwen, Pythia, RAGTruth, and Vectara HHEM are all
open access.

---

## Running the notebooks

Set `MODEL_PREFIX` and `OUTPUT_DIR` in Section 1, then run all eight sections in
order. Do not skip sections. Each section loads artifacts saved by the previous one.

| Model | MODEL\_PREFIX | OUTPUT\_DIR |
|---|---|---|
| Qwen2.5-0.5B | `qwen25_05b` | `features_qwen25_05b` |
| Pythia-1.4B | `pythia1b` | `features_pythia1b` |
| Gemma-2-2B | `gemma2_2b` | `features_gemma2_2b` |
| LLaMA-3-3B | `llama3_3b` | `features_llama3_3b` |
| Qwen2.5-7B | `qwen25_7b` | `features_qwen25_7b` |
| LLaMA-3-8B | `llama3_8b` | `features_llama3_8b` |
| Gemma-2-9B | `gemma2_9b` | `features_gemma2_9b` |

---

## Key results

| Model | Best RT AUC | Best RT F1 | AggreFact BalAcc |
|---|---|---|---|
| Qwen2.5-0.5B | 0.825 | 0.706 | 0.678 |
| Pythia-1.4B | 0.819 | 0.692 | 0.687 |
| Gemma-2-2B | 0.814 | 0.679 | 0.678 |
| LLaMA-3-3B | 0.824 | 0.701 | 0.671 |
| Qwen2.5-7B | **0.835** | **0.717** | 0.690 |
| LLaMA-3-8B | 0.819 | 0.687 | 0.686 |
| Gemma-2-9B | **0.837** | 0.713 | **0.699** |
| ReDeEP (token) | 0.733 | 0.713 | — |
| ReDeEP (chunk) | 0.746 | 0.695 | — |

All seven models beat ReDeEP on AUC. AUC spans only 2.3 points across an eighteen-fold
parameter range. LLaMA-3-3B outperforms LLaMA-3-8B on both AUC and F1.

---

## Known limitations and planned fixes

**Source length mismatch.** Training uses `src_max=1200` while AggreFact averages
~3,000 characters. This causes a 5 to 8 percentage point OOD gap. Retraining at
`src_max=2000` with FEVER and VitaminC should close most of it.

**Synthetic per-generator splits.** The processed RAGTruth dataset does not store
generator labels, so per-generator analysis uses synthetic equal-sized chunks. The
directional finding is robust. Exact numbers carry this caveat.

**Incomplete model coverage.** Qwen2.5-1.5B, GPT-2 Small, GPT-2 XL, and LLaMA-1B
evaluations are still in progress.

**Unexpected LLaMA intra-family result.** LLaMA-3-3B outperforms LLaMA-3-8B on
RAGTruth despite fewer parameters. We are still investigating whether this is related
to attention head count, training data mixture, or differences in how the two models
organize source-grounding circuits.

---

## Citation

```bibtex
@article{singh2025proxyanalyzer,
  title     = {Detecting Hallucinations via Open-Weight Proxy Analyzer Activations},
  author    = {Singh, Akshita and Paudel, Prabesh and Roy, Siddhartha},
  year      = {2025},
  institution = {Khoury College of Computer Sciences, Northeastern University}
}
```

---
