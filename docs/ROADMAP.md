# X-Ray Report Generation — Extension Roadmap

## Project Overview

**Current State:** DenseNet-121 image encoder + custom Transformer decoder trained on the Indiana University Chest X-ray Collection. Generates radiology report impressions from chest X-ray images. Deployed via Gradio + Docker.

**Goal:** Transform this into a production-grade, explainable, multi-capability medical vision-language system that demonstrates both research depth and engineering maturity — suitable as a resume centerpiece for healthcare AI/ML roles.

---

## Strategic Narrative

> "End-to-end medical vision-language system for automated radiology report generation, extended with multi-task pathology classification, Grad-CAM explainability for clinical trust, and deployed as a production FastAPI service with CI/CD, model versioning on HuggingFace Hub, and real-time uncertainty quantification."

---

## Phases at a Glance

| Phase | Extension | Effort | Resume Signal |
|---|---|---|---|
| 1 | Grad-CAM Visual Explainability | Low–Medium | Explainability, clinical trust |
| 2 | Multi-Label Pathology Classification | Medium | Multi-task learning, evaluation rigor |
| 3 | Production MLOps Stack | Medium | Engineering, deployment, CI/CD |
| 4 | ViT + BioGPT Architecture Upgrade | High | SOTA architecture, fine-tuning |
| 5 | DICOM Support + Patient Conditioning | Medium | Clinical realism |
| 6 | Uncertainty Quantification (MC Dropout) | Medium | Safety, robustness |
| 7 | RAG-Augmented Report Generation | High | Research novelty, LLM grounding |

---

## Phase 1 — Visual Explainability with Grad-CAM
**Target:** 1–2 weeks | **Files:** [phase1-gradcam.md](phase1-gradcam.md)

Add heatmap overlays showing which X-ray regions drove the generated report. The #1 requirement for clinically trustworthy AI.

**Deliverables:**
- Grad-CAM heatmaps on DenseNet-121 feature maps
- Token-to-region cross-attention visualization
- Updated Gradio UI with overlay panel

---

## Phase 2 — Multi-Label Pathology Classification
**Target:** 2–3 weeks | **Files:** [phase2-multitask.md](phase2-multitask.md)

Add a parallel classification branch for 14 CheXpert pathologies sharing the DenseNet encoder with the report generator.

**Deliverables:**
- Multi-task model (shared encoder, two heads)
- CheXpert/NIH ChestX-ray14 data pipeline
- AUC-ROC per pathology + BLEU/ROUGE for reports
- Combined prediction display in UI

---

## Phase 3 — Production MLOps Stack
**Target:** 2–3 weeks | **Files:** [phase3-mlops.md](phase3-mlops.md)

Replace Gradio with a production FastAPI backend, add experiment tracking, model registry, and full CI/CD pipeline.

**Deliverables:**
- FastAPI async inference endpoint
- MLflow experiment tracking
- HuggingFace Hub model card + push script
- GitHub Actions CI/CD → Docker → HF Spaces / Render deploy

---

## Phase 4 — ViT + BioGPT Architecture Upgrade
**Target:** 3–4 weeks | **Files:** [phase4-architecture.md](phase4-architecture.md)

Replace DenseNet-121 + custom decoder with ViT encoder + BioGPT decoder. Fine-tune with LoRA.

**Deliverables:**
- ViT-base encoder (google/vit-base-patch16-224)
- BioGPT-Large decoder with LoRA adapters
- Ablation study: baseline vs ViT+BioGPT
- BLEU-4, ROUGE-L, CIDEr, METEOR metrics

---

## Phase 5 — DICOM Support + Patient Conditioning
**Target:** 1–2 weeks | **Files:** [phase5-dicom.md](phase5-dicom.md)

Accept real DICOM files and incorporate patient metadata (age, sex) as decoder conditioning signals.

**Deliverables:**
- pydicom loader with windowing presets
- Metadata embedding module
- Updated inference pipeline
- Example DICOM test cases

---

## Phase 6 — Uncertainty Quantification
**Target:** 2 weeks | **Files:** [phase6-uncertainty.md](phase6-uncertainty.md)

Enable MC Dropout at inference to produce confidence scores and uncertainty heatmaps per token and per image region.

**Deliverables:**
- MC Dropout inference mode
- Token-level confidence scores in report output
- Epistemic uncertainty heatmap on X-ray
- "Low confidence" warning flag in UI

---

## Phase 7 — RAG-Augmented Report Generation
**Target:** 3–4 weeks | **Files:** [phase7-rag.md](phase7-rag.md)

Build a FAISS vector store of training image embeddings → gold reports. At inference, retrieve similar cases and condition the decoder on them.

**Deliverables:**
- FAISS index of DenseNet image embeddings
- Top-K retrieval pipeline
- Decoder conditioned on retrieved report context
- Retrieved case display in UI

---

## Cross-Cutting Concerns

- All new code lives in `src/` with unit tests in `tests/`
- Each phase ships with an updated `README.md` section
- Evaluation metrics tracked in MLflow from Phase 3 onward
- Docker image updated at the end of each phase
- Model checkpoints pushed to HF Hub after Phase 3

---

## Success Metrics

| Metric | Baseline | Target |
|---|---|---|
| BLEU-4 | TBD (measure first) | +15% over baseline |
| ROUGE-L | TBD | +10% over baseline |
| Pathology AUC-ROC | N/A | >0.80 average across 14 classes |
| Inference latency | TBD | <2s per image on CPU |
| UI explainability | None | Grad-CAM overlay + confidence score |
| Deployment | Gradio local | FastAPI + Docker + CI/CD on HF Spaces |
