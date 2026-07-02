# Experimental Results

This file tracks all quantitative results across experiments. Update after every training run.

---

## Baseline Metrics

> Run evaluation on test split before any modifications. Record these numbers first.

| Metric | Value | Date | Notes |
|---|---|---|---|
| BLEU-1 | TBD | — | Run `src/evaluation/text_metrics.py` on test split |
| BLEU-4 | TBD | — | Primary report quality metric |
| ROUGE-L | TBD | — | Longest common subsequence overlap |
| CIDEr | TBD | — | Consensus-based image description metric |
| Inference latency (CPU) | TBD ms | — | Average over 100 test images |
| Inference latency (GPU) | TBD ms | — | Average over 100 test images |

---

## Phase 2 — Multi-Label Classification (CheXpert/NIH)

### Training Configuration
| Parameter | Value |
|---|---|
| Dataset | TBD (CheXpert / NIH ChestX-ray14) |
| Label policy | TBD (U-zeros / U-ones) |
| λ1 (report weight) | 1.0 |
| λ2 (classification weight) | 0.5 |
| Epochs | TBD |

### Per-Class AUC-ROC
| Pathology | AUC-ROC | Threshold F1 |
|---|---|---|
| Atelectasis | TBD | TBD |
| Cardiomegaly | TBD | TBD |
| Consolidation | TBD | TBD |
| Edema | TBD | TBD |
| Enlarged Cardiomediastinum | TBD | TBD |
| Fracture | TBD | TBD |
| Lung Lesion | TBD | TBD |
| Lung Opacity | TBD | TBD |
| No Finding | TBD | TBD |
| Pleural Effusion | TBD | TBD |
| Pleural Other | TBD | TBD |
| Pneumonia | TBD | TBD |
| Pneumothorax | TBD | TBD |
| Support Devices | TBD | TBD |
| **Macro Average** | **TBD** | **TBD** |

### Impact on Report Generation
| Metric | Single-Task | Multi-Task | Delta |
|---|---|---|---|
| BLEU-4 | TBD | TBD | TBD |
| ROUGE-L | TBD | TBD | TBD |

---

## Phase 4 — Architecture Ablation Study

### Configurations
| Config | Encoder | Decoder | Decoding | BLEU-4 | ROUGE-L | CIDEr | Params (M) | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| Baseline | DenseNet-121 | Custom Transformer | Greedy | TBD | TBD | TBD | TBD | TBD |
| Enc-Upgrade | ViT-base | Custom Transformer | Greedy | TBD | TBD | TBD | TBD | TBD |
| Dec-Upgrade | DenseNet-121 | BioGPT-LoRA | Greedy | TBD | TBD | TBD | TBD | TBD |
| Full-Upgrade | ViT-base | BioGPT-LoRA | Greedy | TBD | TBD | TBD | TBD | TBD |
| Full+Beam | ViT-base | BioGPT-LoRA | Beam-4 | TBD | TBD | TBD | TBD | TBD |

### Decoding Strategy Comparison (Full-Upgrade model)
| Strategy | BLEU-4 | ROUGE-L | Avg. Report Length | Notes |
|---|---|---|---|---|
| Greedy | TBD | TBD | TBD | |
| Beam-4 | TBD | TBD | TBD | |
| Top-p (p=0.9) | TBD | TBD | TBD | |
| Temperature (T=0.7) | TBD | TBD | TBD | |

---

## Phase 6 — Uncertainty Quantification

### Calibration Results
| Model | ECE (before scaling) | ECE (after temp scaling) | Temperature T |
|---|---|---|---|
| Baseline | TBD | TBD | TBD |
| ViT+BioGPT | TBD | TBD | TBD |

### MC Dropout Performance (n_samples=20)
| Metric | Value | Notes |
|---|---|---|
| Single-pass latency (GPU) | TBD ms | |
| MC inference latency (GPU, n=20) | TBD ms | |
| Speedup from batched MC | TBD x | vs. sequential N passes |
| Mean report confidence (correct reports) | TBD | Expected: high |
| Mean report confidence (incorrect reports) | TBD | Expected: lower |
| AUC of confidence vs. correctness | TBD | How well confidence predicts quality |

---

## Phase 7 — RAG Ablation Study

### Retrieval Configurations
| Config | K | Retrieval | BLEU-4 | ROUGE-L | CIDEr | Hallucination Rate |
|---|---|---|---|---|---|---|
| No-RAG | 0 | None | TBD | TBD | TBD | TBD |
| RAG-K1 | 1 | FAISS exact | TBD | TBD | TBD | TBD |
| RAG-K3 | 3 | FAISS exact | TBD | TBD | TBD | TBD |
| RAG-K5 | 5 | FAISS exact | TBD | TBD | TBD | TBD |
| RAG-MMR | 3 | FAISS + MMR | TBD | TBD | TBD | TBD |
| RAG-Oracle | 3 | Ground truth | TBD | TBD | TBD | TBD |

### Retrieval Quality
| Metric | Value | Notes |
|---|---|---|
| Retrieval latency (avg) | TBD ms | FAISS search only |
| Pathology precision@1 | TBD | Do top-1 retrievals share the same pathology? |
| Pathology precision@3 | TBD | Do top-3 share the same pathology? |
| Index size | TBD MB | FAISS index file size |
| Index build time | TBD min | Full training set indexing time |

---

## Final System Summary

To be filled after all phases complete.

| Capability | Status | Key Metric |
|---|---|---|
| Report generation | TBD | BLEU-4: TBD |
| Pathology classification | TBD | AUC-ROC: TBD |
| Explainability (Grad-CAM) | TBD | Qualitative |
| DICOM support | TBD | — |
| Uncertainty quantification | TBD | ECE: TBD |
| RAG augmentation | TBD | BLEU-4 delta: TBD |
| Deployment (FastAPI + HF Spaces) | TBD | Latency: TBD ms |
