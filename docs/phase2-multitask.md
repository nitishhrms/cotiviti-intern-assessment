# Phase 2 — Multi-Label Pathology Classification

## Goal

Add a parallel classification head to the existing DenseNet-121 encoder so the model simultaneously generates a radiology report *and* predicts structured pathology labels (e.g., Atelectasis, Cardiomegaly, Pleural Effusion). This demonstrates multi-task learning on shared visual representations — a key technique in medical vision AI.

## Why This Matters for Your Resume

- Multi-task learning is a core technique in production medical AI (used at Viz.ai, Aidoc, Rad AI)
- Shows you can design loss functions for joint optimization
- Produces two clinically useful outputs from one inference call
- CheXpert AUC-ROC scores are a well-known benchmark — citing real numbers impresses interviewers

---

## Prerequisites

- [ ] Phase 1 complete (Grad-CAM integrated)
- [ ] Download NIH ChestX-ray14 dataset OR CheXpert dataset (both free with registration)
  - NIH: https://nihcc.app.box.com/v/ChestXray-NIHCC (112,000 images, 14 labels)
  - CheXpert: https://stanfordmlgroup.github.io/competitions/chexpert/ (224,000 images, 14 labels)
- [ ] Note: Indiana University dataset (current) does NOT have structured labels — need CheXpert or NIH for this phase
- [ ] Install: `scikit-learn` (for AUC-ROC), `albumentations` (for augmentation)

---

## Task 1 — Data Pipeline

### 1.1 — Dataset Loader
- [ ] Create `src/data/chexpert_dataset.py` (or `nih_dataset.py`)
- [ ] Implement `ChestXrayDataset(Dataset)`:
  - [ ] Accept CSV path, image root directory, label columns, and transform
  - [ ] Load 14 label columns as float tensors (handle NaN/uncertain labels: map -1 → 0 or use U-zero/U-ones policy from CheXpert paper)
  - [ ] Load report text if available (for joint training)
  - [ ] Return dict: `{'image': tensor, 'labels': tensor, 'report': str}`
- [ ] Implement train/val/test split logic (use official CheXpert splits or 80/10/10 random split)

### 1.2 — Label Handling Strategy
- [ ] Implement U-zeros policy: treat uncertain labels (-1) as negative (0)
- [ ] Implement U-ones policy: treat uncertain labels (-1) as positive (1)
- [ ] Add a config flag to switch between policies
- [ ] Document trade-offs in code comments

### 1.3 — Data Augmentation
- [ ] Create `src/data/transforms.py`
- [ ] Define training augmentation pipeline using `albumentations`:
  - [ ] `RandomHorizontalFlip(p=0.5)` — X-rays can be flipped left-right
  - [ ] `RandomRotate90(p=0.1)` — minor rotation variance
  - [ ] `RandomBrightnessContrast(p=0.3)` — simulate exposure variation
  - [ ] `GaussianBlur(p=0.1)` — simulate motion artifact
  - [ ] `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` — ImageNet stats
- [ ] Define val/test pipeline (normalize only, no augmentation)

### 1.4 — DataLoader Setup
- [ ] Create `src/data/datamodule.py`
- [ ] Use `torch.utils.data.DataLoader` with `num_workers=4`, `pin_memory=True`
- [ ] Handle class imbalance: compute per-label positive weight for `BCEWithLogitsLoss`
  - [ ] `pos_weight = (num_negative / num_positive)` per class

---

## Task 2 — Multi-Task Model Architecture

### 2.1 — Classification Head
- [ ] Create `src/models/classification_head.py`
- [ ] Implement `PathologyClassifier(nn.Module)`:
  - [ ] Input: 1024-dim DenseNet feature vector (before the existing 1024→256 projection)
  - [ ] Architecture: `Linear(1024, 512) → BatchNorm → ReLU → Dropout(0.3) → Linear(512, 14)`
  - [ ] Output: logits for 14 pathology classes (no sigmoid — use `BCEWithLogitsLoss`)
  - [ ] Add `predict_proba()` method that applies sigmoid for inference

### 2.2 — Unified Multi-Task Model
- [ ] Create `src/models/multitask_model.py`
- [ ] Implement `MultiTaskRadiologyModel(nn.Module)`:
  - [ ] Shared backbone: DenseNet-121 feature extractor (reuse `ImageEncoder` from `app.py`)
  - [ ] Branch 1: existing 1024→256 projection + Transformer decoder (report generation)
  - [ ] Branch 2: `PathologyClassifier` (pathology prediction)
  - [ ] `forward(images, captions=None)` returns `(report_logits, pathology_logits)`
  - [ ] Add `generate_report(image)` method for inference-only report generation
  - [ ] Add `classify(image)` method for inference-only classification

### 2.3 — Loss Function
- [ ] Create `src/training/multitask_loss.py`
- [ ] Implement `MultiTaskLoss`:
  - [ ] `report_loss`: `CrossEntropyLoss` on generated token logits (existing)
  - [ ] `classification_loss`: `BCEWithLogitsLoss` with `pos_weight` for class imbalance
  - [ ] Combined: `total_loss = λ1 * report_loss + λ2 * classification_loss`
  - [ ] Default: `λ1 = 1.0, λ2 = 0.5` (tune via ablation)
  - [ ] Log individual loss components separately for monitoring

---

## Task 3 — Training Loop

### 3.1 — Training Script
- [ ] Create `src/training/train_multitask.py`
- [ ] Support command-line args via `argparse`: `--epochs`, `--lr`, `--lambda1`, `--lambda2`, `--data_dir`
- [ ] Implement training loop:
  - [ ] Load `MultiTaskRadiologyModel` (optionally load pretrained encoder weights from `best_model.pth`)
  - [ ] Freeze encoder for first 5 epochs (train only new heads) → then unfreeze for fine-tuning
  - [ ] Use `AdamW` optimizer with `weight_decay=1e-4`
  - [ ] Use cosine annealing LR scheduler
  - [ ] Gradient clipping: `max_norm=1.0`
- [ ] Implement validation loop every epoch
- [ ] Save best model checkpoint based on validation AUC-ROC

### 3.2 — Logging
- [ ] Log to MLflow (Phase 3 prerequisite — use print statements for now, refactor in Phase 3):
  - [ ] Per-epoch: `train_loss`, `val_loss`, `report_loss`, `classification_loss`
  - [ ] Per-epoch: macro-averaged `val_auc_roc`, per-class AUC-ROC
  - [ ] Per-epoch: `val_bleu4`, `val_rouge_l`

---

## Task 4 — Evaluation

### 4.1 — Report Generation Metrics
- [ ] Install `nltk`, `rouge-score`
- [ ] Create `src/evaluation/text_metrics.py`
- [ ] Implement `compute_bleu4(hypotheses, references)` using `nltk.translate.bleu_score`
- [ ] Implement `compute_rouge(hypotheses, references)` using `rouge_score` library
- [ ] Implement `compute_cider(hypotheses, references)` (optional, use `pycocoevalcap` if available)
- [ ] Run evaluation on test split, record numbers in `docs/results.md`

### 4.2 — Classification Metrics
- [ ] Create `src/evaluation/classification_metrics.py`
- [ ] Implement `compute_auc_roc(y_true, y_pred_proba)` per class and macro-averaged
- [ ] Implement `compute_f1(y_true, y_pred, threshold=0.5)` per class and macro-averaged
- [ ] Generate per-class AUC table and save to `docs/results.md`
- [ ] Plot ROC curves for all 14 classes and save as `docs/figures/roc_curves.png`

### 4.3 — Ablation Study
- [ ] Train model with classification head only (no report generation) — record AUC
- [ ] Train model with report generation only (no classification) — record BLEU
- [ ] Train full multi-task model — record both
- [ ] Document in `docs/results.md`: does joint training help either task?

---

## Task 5 — UI Integration

### 5.1 — Update Inference in `app.py`
- [ ] Load `MultiTaskRadiologyModel` instead of single-task model
- [ ] Return both `(report_text, pathology_predictions, gradcam_overlay)` from inference
- [ ] Format pathology predictions as a structured list: `"Atelectasis: 87% | Cardiomegaly: 12% | ..."`

### 5.2 — Gradio UI Update
- [ ] Add a `gr.Label` component for pathology probabilities (shows bar chart automatically)
- [ ] Add confidence threshold slider (default 0.5) to control what counts as "positive"
- [ ] Highlight positive predictions in red, negative in green

---

## Task 6 — Testing

- [ ] Create `tests/test_multitask_model.py`
- [ ] Test that `MultiTaskRadiologyModel.forward()` returns correct output shapes
- [ ] Test `MultiTaskLoss` computes correct combined loss
- [ ] Test `compute_auc_roc` on synthetic data
- [ ] Test `ChestXrayDataset.__getitem__` returns correct label tensor shape

---

## File Structure After This Phase

```
src/
  data/
    __init__.py
    chexpert_dataset.py
    transforms.py
    datamodule.py
  models/
    __init__.py
    classification_head.py
    multitask_model.py
  training/
    __init__.py
    train_multitask.py
    multitask_loss.py
  evaluation/
    __init__.py
    text_metrics.py
    classification_metrics.py
tests/
  test_multitask_model.py
docs/
  results.md
  figures/
    roc_curves.png
```

---

## Definition of Done

- [ ] Multi-task model trains to completion without NaN loss
- [ ] Per-class AUC-ROC > 0.75 for at least 10 of 14 pathologies
- [ ] BLEU-4 not degraded vs. single-task baseline (within 2%)
- [ ] Gradio UI shows both report and pathology confidence bars
- [ ] Results documented in `docs/results.md` with real numbers

---

## References

- Irvin et al., "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels" (AAAI 2019)
- Wang et al., "ChestX-Ray8: Hospital-scale Chest X-ray Database and Benchmarks" (CVPR 2017)
- Ruder, "An Overview of Multi-Task Learning in Deep Neural Networks" (2017)
