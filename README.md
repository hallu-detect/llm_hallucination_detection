# Detecting Hallucinations via Open-Weight Proxy Analyzer Activations

**Akshita Singh · Prabesh Paudel · Siddhartha Roy**  
Khoury College of Computer Sciences, Northeastern University  
`{singh.akshita, paudel.pr, roy.sidd}@northeastern.edu`

---

## What this project does

We built a hallucination detection system that works without any access to the model that generated the text. Given a source document, a question, and a candidate answer, our proxy-analyzer reads the text through a small locally hosted open-weight model, intercepts its internal activations, and decides whether the answer is likely hallucinated.

The key insight is that hallucination is a property of the relationship between source and answer, not just a quirk of the generating model. Any transformer reading a contradictory sentence against its source will show altered attention patterns and stronger memory activations, regardless of which model originally produced that sentence. We exploit exactly those signals.

We extract eighteen mechanistically grounded features from a single forward pass, train a stacking ensemble classifier, and evaluate across six analyzer architectures ranging from 0.5 billion to 9 billion parameters. On the RAGTruth benchmark, every one of our configurations beats ReDeEP's token-level AUC of 0.73 by between 7.4 and 10.3 percentage points. Our Qwen2.5-7B setup reaches an F1 of 0.717, the first proxy-analyzer to exceed ReDeEP's token-level F1 of 0.713, without ever touching the generator's weights.

---

## Repository structure

```
.
├── notebooks/
│   ├── gemma2_9b/
│   │   ├── section5_feature_extraction.ipynb
│   │   ├── section6_classifier_training.ipynb
│   │   ├── section7_calibration.ipynb
│   │   └── section8_final_evaluation.ipynb
│   ├── qwen25_7b/
│   │   ├── section5_feature_extraction.ipynb
│   │   ├── section6_classifier_training.ipynb
│   │   ├── section7_calibration.ipynb
│   │   └── section8_final_evaluation.ipynb
│   ├── qwen25_05b/
│   │   └── ... (same structure)
│   ├── pythia1b/
│   │   └── ...
│   ├── gemma2_2b/
│   │   └── ...
│   └── llama3_8b/
│       └── ...
├── requirements.txt
└── README.md
```

Each model folder follows the same four-section pipeline. You can run them independently or sequentially. Sections 6, 7, and 8 load artifacts saved by the previous section, so they need to be run in order within a given model folder.

---

## The four-section pipeline in all the LLM notebooks

### Section 5: Feature Extraction

This is the most compute-intensive step. We run one forward pass per sample through the analyzer model using TransformerLens hooks to capture:

- `resid_post`: residual stream states after each layer
- `pattern`: full attention weight matrices per head per layer
- `mlp_out`: MLP output activations per layer

From these we compute eighteen signals. The most important ones are:

- **S2 (source-document attention):** mean attention from answer token positions to source token positions, resolved per head and per layer. This alone accounts for roughly 50% of Random Forest feature importance.
- **S3 (attention entropy):** how focused or diffuse each head's attention is.
- **S4 (MLP norms):** how strongly parametric memory is firing, analogous to the PKS score in ReDeEP.
- **S8 (external faithfulness):** scored by Vectara HHEM-2.1-Open in a separate batched pass.
- **S16–S18 (token-level grounding):** minimum, variance, and slope of per-token source attention across the answer. These are computed for free inside the S2 loop and catch structure that mean-pooled signals miss.

We also extract the full feature set for LLM-AggreFact (our OOD evaluation set) at `src_max=3000` characters rather than the training value of 1200. AggreFact is never used in any training or calibration step.

**Runtime per model on a single A100:** roughly 4–6 GPU-hours for the full 72,135-row training set.

### Section 6: Augmentation, Scaling, and Classifier Training

This section:

1. Reconstructs the 70/15/15 stratified split (always with `random_state=42`)
2. Selects the FIXED\_WINDOW by maximising per-layer S2 AUC on the RAGTruth training split using 3-fold cross-validation
3. Computes Attention Head Importance (AHI) weights from RAGTruth training labels
4. Orthogonalizes S13 against S6 and S14, S17, S18 against the S2 mean, using training rows only
5. Trains five classifiers: Logistic Regression, Random Forest, HistGradientBoosting, XGBoost, and a Stacking meta-learner
6. Trains a second specialist, RagtStacking, on RAGTruth training rows only
7. Runs the AggreFact OOD diagnostic (informational only, no training use)

All scaler fitting, orthogonalization coefficients, and signal power profiling are computed on training rows only. Nothing from validation or test leaks into this section.

### Section 7: Calibration

This section fits a three-tier isotonic calibration:

- **QA regime:** fitted on HaluEval validation rows only. We exclude RAGTruth here because the Kolmogorov-Smirnov distance between the two distributions exceeds 0.45 across all models.
- **Claim regime:** fitted on MiniCheck-Synthetic and ANLI validation rows.
- **Global:** applied to everything else, including RAGTruth.

Temperature is set to `T=2.0` across all models, chosen during the Gemma-2-9B calibration phase and kept fixed thereafter. All calibration parameters are estimated on validation data and applied to the test set without re-fitting.

### Section 8: Final Evaluation

This section reports all results and generates seven plots (8A through 8G):

- **8A:** Per-dataset results for all four systems (Stacking raw, Stacking+cal, RagtStacking raw, RagtStacking+cal) including ROC curves, PR curves, confusion matrices, Brier scores, and ECE.
- **8B:** Per-generator breakdown on RAGTruth and AggreFact OOD.
- **8C:** Per-layer signal analysis including the regime×layer discriminability heatmap.
- **8D:** All signal AUCs on both test and AggreFact, with the stability scatter and OOD gap ranking.
- **8E:** AggreFact OOD detailed per-subdataset breakdown.
- **8F:** Error analysis with TP/TN/FP/FN distributions for both classifiers.
- **8G:** Length diagnostics and complete paper numbers summary.

We also print ten textual/tabular analyses (T1 through T10) covering AUC tables, F1 tables, BalAcc tables, per-dataset winner decisions, calibration gain, per-generator RAGTruth, AggreFact per-subdataset, signal stability, layer analysis, and clean final numbers.

---

## Datasets

We train on five publicly available datasets totalling 72,135 rows.

| Dataset | Rows | Hall% | Task type | Generator |
|---|---|---|---|---|
| HaluEval QA | 19,971 | 50% | Entity substitution | GPT-3.5 |
| RAGTruth | 15,090 | 44.5% | Faithfulness (RAG) | GPT-4, GPT-3.5, Mistral-7B, Llama-2 |
| MedHallu | 10,000 | 50% | Medical confabulation | GPT-4 |
| MiniCheck-Synthetic | 7,076 | 42% | Claim verification | Synthetic |
| ANLI | 19,998 | 50% | NLI / adversarial | Human annotated |

For out-of-distribution evaluation we use **LLM-AggreFact** (`lytang/LLM-AggreFact`). We remove 16,371 RAGTruth-overlapping rows, one contaminated row, and invert the labels to match our faithful=0 convention. The resulting 12,948 rows span ten sub-tasks and outputs from multiple LLM families.

---

## Analyzer models

We tested six proxy-analyzer architectures across two model families. All models are loaded in float16 via TransformerLens.

| Model | Parameters | Layers | Heads | Feature dim |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.5B | 24 | 14 | 743 |
| Pythia-1.4B | 1.4B | 24 | 16 | 839 |
| Gemma-2-2B | 2B | 26 | 8 | 491 |
| Qwen2.5-7B | 7B | 28 | 28 | 1,647 |
| LLaMA-3-8B | 8B | 32 | 32 | 2,135 |
| Gemma-2-9B | 9B | 42 | 16 | 1,451 |

---

## Setup

### 1. Install PyTorch with CUDA 12.8

```bash
pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu128
```

### 2. Install all other dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Install LaTeX tools for notebook rendering

```bash
apt-get install -y texlive-xetex texlive-fonts-recommended texlive-plain-generic pandoc
```

### 4. Set your HuggingFace token

Several datasets and models require a HuggingFace account with access granted. Set your token before running notebooks:

```bash
export HF_TOKEN=your_token_here
```

Or inside a notebook cell:

```python
from huggingface_hub import login
login(token="your_token_here")
```

You will need access to:
- `meta-llama/Meta-Llama-3-8B` (requires HuggingFace approval)
- `google/gemma-2-9b` and `google/gemma-2-2b` (requires HuggingFace approval)
- `Qwen/Qwen2.5-0.5B` and `Qwen/Qwen2.5-7B` (open access)
- `EleutherAI/pythia-1.4b` (open access)
- `wandb/RAGTruth-processed` (open access)
- `vectara/hallucination_evaluation_model` (open access)

---

## Running the notebooks

Set the `MODEL_PREFIX` variable at the top of each notebook. This controls where features and artifacts are saved.

| Model | MODEL_PREFIX |
|---|---|
| Qwen2.5-0.5B | `qwen25_05b` |
| Pythia-1.4B | `pythia1b` |
| Gemma-2-2B | `gemma2_2b` |
| Qwen2.5-7B | `qwen25_7b` |
| LLaMA-3-8B | `llama3_8b` |
| Gemma-2-9B | `gemma2_9b` |

Each section saves its outputs to `features_{MODEL_PREFIX}/`. Section 6 reads from Section 5, Section 7 from Section 6, and Section 8 from all three. Do not skip sections.

---

## Key results

All six models beat ReDeEP's token-level AUC on RAGTruth, with gains of 7.4 to 10.3 percentage points. The most striking finding is how tightly the results cluster: despite an eighteen-fold difference in parameter count, the best AUC across all six models spans only 2.3 percentage points (0.814 to 0.837). A 0.5 billion parameter model is remarkably close to a 9 billion one.

| Model | Best RAGTruth AUC | Best RAGTruth F1 | AggreFact BalAcc |
|---|---|---|---|
| Qwen2.5-0.5B | 0.825 | 0.706 | 0.678 |
| Pythia-1.4B | 0.819 | 0.692 | 0.687 |
| Gemma-2-2B | 0.814 | 0.679 | 0.678 |
| Qwen2.5-7B | **0.835** | **0.717** | 0.690 |
| LLaMA-3-8B | 0.819 | 0.687 | 0.686 |
| Gemma-2-9B | **0.837** | 0.713 | **0.699** |
| ReDeEP (token) | 0.733 | 0.713 | — |
| ReDeEP (chunk) | 0.746 | 0.695 | — |

Qwen2.5-7B is the first proxy-analyzer to exceed ReDeEP's token-level F1 of 0.713. Architecture family turns out to be a better predictor of detection quality than raw parameter count. A 0.5B Qwen beats an 8B LLaMA on RAGTruth AUC.

---

## Known limitations and planned fixes

**Source context length mismatch.** We truncate source documents to 1,200 characters during training, while LLM-AggreFact sources average around 3,000. This causes a 5 to 7 percentage point gap in out-of-distribution balanced accuracy. We plan to retrain with 2,000-character sources and add FEVER and VitaminC claim data, which we expect will close the gap to roughly 1 to 3 percentage points.

**Synthetic per-generator splits.** RAGTruth's processed version does not retain generator labels, so our per-generator analysis uses equal-sized synthetic chunks. The directional finding (RagtStacking wins across all generators) is robust, but absolute per-generator numbers carry this caveat.

**Incomplete model coverage.** Qwen2.5-1.5B, GPT-2 Small, GPT-2 XL, and LLaMA-1B evaluations are in progress.

---

## Citation

If you use this code or build on our findings, please cite:

```bibtex
@article{singh2025proxyanalyzer,
  title     = {Detecting Hallucinations via Open-Weight Proxy Analyzer Activations},
  author    = {Singh, Akshita and Paudel, Prabesh and Roy, Siddhartha},
  year      = {2025},
  institution = {Khoury College of Computer Sciences, Northeastern University}
}
```

---

## Acknowledgments

We thank the TransformerLens team for making mechanistic
interpretability tooling accessible to the research community.
