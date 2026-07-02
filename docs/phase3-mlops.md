# Phase 3 — Production MLOps Stack

## Goal

Replace the Gradio-only prototype with a production-grade system: FastAPI backend, MLflow experiment tracking, HuggingFace Hub model registry, and a GitHub Actions CI/CD pipeline that automatically builds, tests, and deploys. This phase is what separates "built a model in a notebook" from "shipped a production ML product."

## Why This Matters for Your Resume

- MLOps skills are in short supply — most ML candidates can't deploy
- FastAPI + Docker + CI/CD is the standard production stack at startups and big tech
- A public HuggingFace model card with real metrics is immediately visible to recruiters
- Shows you understand reproducibility, versioning, and reliability in production systems

---

## Prerequisites

- [ ] Phase 1 and 2 complete
- [ ] GitHub account with a public repo for this project
- [ ] HuggingFace account (free) at huggingface.co
- [ ] Docker installed locally
- [ ] Install: `fastapi`, `uvicorn`, `mlflow`, `huggingface_hub`, `pytest`, `httpx` (for API testing)

---

## Task 1 — FastAPI Backend

### 1.1 — Project Restructure
- [ ] Create `backend/` directory
- [ ] Move inference logic from `app.py` into `backend/inference.py`
  - [ ] Create `InferenceEngine` class that loads model once at startup
  - [ ] Expose `predict(image_bytes) -> dict` method
  - [ ] Handle model loading errors gracefully with informative messages
- [ ] Create `backend/main.py` as the FastAPI entrypoint

### 1.2 — API Endpoints
- [ ] Implement `POST /predict` endpoint in `backend/main.py`:
  - [ ] Accept `multipart/form-data` with image file upload
  - [ ] Validate file type (accept JPEG, PNG, DICOM)
  - [ ] Validate file size (reject > 20MB)
  - [ ] Return JSON: `{"report": str, "pathologies": dict[str, float], "confidence": float, "processing_time_ms": int}`
  - [ ] Return HTTP 422 for invalid input with clear error message
- [ ] Implement `GET /health` endpoint:
  - [ ] Return `{"status": "healthy", "model_loaded": bool, "version": str}`
  - [ ] Used by Docker health checks and load balancers
- [ ] Implement `GET /metadata` endpoint:
  - [ ] Return model version, vocabulary size, training dataset, supported pathologies
- [ ] Implement `GET /metrics` endpoint (Prometheus-compatible):
  - [ ] `inference_count_total`, `inference_latency_seconds`, `error_count_total`

### 1.3 — Request/Response Models
- [ ] Create `backend/schemas.py` using Pydantic:
  - [ ] `PredictionResponse(BaseModel)`: report, pathologies, confidence, processing_time_ms, model_version
  - [ ] `HealthResponse(BaseModel)`: status, model_loaded, version, uptime_seconds
  - [ ] `ErrorResponse(BaseModel)`: error, detail, request_id

### 1.4 — Middleware
- [ ] Add `RequestIdMiddleware`: attach UUID to every request for tracing
- [ ] Add `TimingMiddleware`: log inference latency per request
- [ ] Add CORS middleware (allow all origins for demo; restrict in production)
- [ ] Add request size limit middleware (20MB max)

### 1.5 — Async Inference
- [ ] Wrap model inference in `asyncio.get_event_loop().run_in_executor()` to avoid blocking the event loop
- [ ] Set up a thread pool with `max_workers=2` (one per GPU/CPU core allocated)

---

## Task 2 — MLflow Experiment Tracking

### 2.1 — Setup
- [ ] Install `mlflow` and add to `requirements.txt`
- [ ] Create `mlflow_tracking/` directory
- [ ] Set up MLflow tracking server (local SQLite backend for now):
  - [ ] `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns`
  - [ ] Add `mlflow.db` and `mlruns/` to `.gitignore`

### 2.2 — Instrument Training Scripts
- [ ] Wrap training in `with mlflow.start_run():` context
- [ ] Log hyperparameters with `mlflow.log_params()`:
  - [ ] `embed_dim`, `num_heads`, `num_decoder_layers`, `dropout`, `lr`, `batch_size`, `epochs`
  - [ ] `lambda1`, `lambda2` (multi-task weights)
  - [ ] `encoder_architecture`, `decoder_architecture`
  - [ ] `dataset_name`, `vocab_size`, `num_train_samples`
- [ ] Log metrics with `mlflow.log_metric(step=epoch)`:
  - [ ] `train_loss`, `val_loss`, `report_loss`, `classification_loss`
  - [ ] `val_bleu4`, `val_rouge_l`, `val_cider`
  - [ ] `val_auc_roc_macro`, per-class AUC-ROC values
- [ ] Log artifacts with `mlflow.log_artifact()`:
  - [ ] Best model checkpoint `best_model.pth`
  - [ ] `vocab.json`
  - [ ] `config.json`
  - [ ] ROC curve plots

### 2.3 — Model Registry
- [ ] Register best model with `mlflow.register_model()`
- [ ] Tag model with `stage=Staging` after initial training
- [ ] Transition to `stage=Production` only when AUC-ROC > 0.80 and BLEU-4 improves
- [ ] Write `scripts/promote_model.py` that checks metrics and transitions stage automatically

---

## Task 3 — HuggingFace Hub Integration

### 3.1 — Repository Setup
- [ ] Create a HuggingFace repository: `{your-username}/xray-report-generator`
- [ ] Add `README.md` as the model card (HF uses this as the card)
- [ ] Create `hf_model_card.md` template with:
  - [ ] Model description, intended use, limitations
  - [ ] Training data (Indiana University Chest X-ray Collection)
  - [ ] Evaluation metrics (BLEU-4, ROUGE-L, AUC-ROC table)
  - [ ] Example usage code snippet
  - [ ] Bias and ethical considerations section (required for medical AI)
  - [ ] `pipeline_tag: image-to-text`
  - [ ] `tags: medical, radiology, chest-xray, report-generation`

### 3.2 — Push Script
- [ ] Create `scripts/push_to_hub.py`:
  - [ ] Authenticate with `huggingface_hub.login()`
  - [ ] Push `best_model.pth` using `huggingface_hub.upload_file()`
  - [ ] Push `vocab.json`, `config.json`
  - [ ] Push ONNX model files
  - [ ] Update model card with latest metrics from MLflow
  - [ ] Tag commit with version number

### 3.3 — HF Spaces Deployment
- [ ] Create a separate HF Space repository: `{your-username}/xray-report-demo`
- [ ] Configure Space to use `gradio` SDK
- [ ] Update `app.py` for HF Spaces:
  - [ ] Load model from HF Hub using `huggingface_hub.hf_hub_download()`
  - [ ] Add `requirements.txt` in Space repo root
  - [ ] Add `README.md` with Space metadata (title, emoji, color theme)

---

## Task 4 — Docker Improvements

### 4.1 — Multi-Stage Dockerfile
- [ ] Rewrite `Dockerfile` as multi-stage build:
  - [ ] Stage 1 (`builder`): install Python deps into a virtualenv
  - [ ] Stage 2 (`runtime`): copy only the virtualenv, not build tools
  - [ ] Result: final image ~500MB instead of ~1.2GB
- [ ] Add `.dockerignore` to exclude: `*.pth` (download at runtime), `mlruns/`, `__pycache__/`, `.git/`

### 4.2 — Docker Compose
- [ ] Create `docker-compose.yml`:
  - [ ] `api` service: FastAPI backend on port 8000
  - [ ] `mlflow` service: MLflow tracking server on port 5000
  - [ ] `frontend` service: Gradio or static React frontend on port 3000
  - [ ] Shared volume for model artifacts
- [ ] Create `docker-compose.dev.yml` override for local development with hot reload

### 4.3 — Health Checks
- [ ] Add `HEALTHCHECK` instruction to Dockerfile: `curl -f http://localhost:8000/health`
- [ ] Set `--interval=30s --timeout=10s --retries=3`

---

## Task 5 — GitHub Actions CI/CD Pipeline

### 5.1 — Test Pipeline (on every PR)
- [ ] Create `.github/workflows/test.yml`
- [ ] Steps:
  - [ ] Checkout code
  - [ ] Set up Python 3.11
  - [ ] Cache pip dependencies
  - [ ] Install test dependencies
  - [ ] Run `pytest tests/ -v --tb=short`
  - [ ] Upload test results as artifact
  - [ ] Fail PR if tests fail

### 5.2 — Build Pipeline (on merge to main)
- [ ] Create `.github/workflows/build.yml`
- [ ] Steps:
  - [ ] Build Docker image: `docker build -t xray-report:${{ github.sha }}`
  - [ ] Run smoke test against built container: `curl http://localhost:8000/health`
  - [ ] Tag image as `latest` on success
  - [ ] Push to GitHub Container Registry (GHCR): `ghcr.io/{username}/xray-report`

### 5.3 — Deploy Pipeline (on tag release)
- [ ] Create `.github/workflows/deploy.yml`
- [ ] Trigger on: `push to tags v*`
- [ ] Steps:
  - [ ] Pull image from GHCR
  - [ ] Push model artifacts to HF Hub using `scripts/push_to_hub.py`
  - [ ] Deploy Gradio app to HF Spaces (sync via `huggingface_hub.Repository.push_to_hub()`)
  - [ ] (Optional) Deploy FastAPI to Render.com using Render Deploy Hook secret

### 5.4 — Secrets Management
- [ ] Add GitHub repository secrets:
  - [ ] `HF_TOKEN`: HuggingFace API token
  - [ ] `RENDER_DEPLOY_HOOK`: Render deployment webhook URL
- [ ] Document required secrets in `docs/deployment.md`

---

## Task 6 — Monitoring (Basic)

### 6.1 — Structured Logging
- [ ] Add `structlog` or `loguru` to `backend/`
- [ ] Log every inference: `request_id`, `image_size`, `processing_time_ms`, `report_length`, `top_pathology`
- [ ] Log errors with full stack trace and request context

### 6.2 — Simple Dashboard
- [ ] Create `scripts/log_analysis.py` that reads logs and prints:
  - [ ] Total requests, average latency, error rate
  - [ ] Top 3 most predicted pathologies
  - [ ] Requests per hour histogram

---

## Task 7 — Testing

- [ ] Create `tests/test_api.py` using `pytest` + `httpx.AsyncClient`
- [ ] Test `GET /health` returns 200 with correct schema
- [ ] Test `POST /predict` with valid JPEG returns correct JSON structure
- [ ] Test `POST /predict` with invalid file type returns 422
- [ ] Test `POST /predict` with file > 20MB returns 413
- [ ] Test `GET /metadata` returns expected keys

---

## File Structure After This Phase

```
backend/
  __init__.py
  main.py             # FastAPI app
  inference.py        # InferenceEngine class
  schemas.py          # Pydantic models
  middleware.py       # Request ID, timing, CORS
scripts/
  push_to_hub.py
  promote_model.py
  log_analysis.py
.github/
  workflows/
    test.yml
    build.yml
    deploy.yml
docker-compose.yml
docker-compose.dev.yml
Dockerfile            # Updated multi-stage
.dockerignore
tests/
  test_api.py
```

---

## Definition of Done

- [ ] `docker-compose up` starts all services with no manual steps
- [ ] `GET http://localhost:8000/health` returns `{"status": "healthy"}`
- [ ] `POST http://localhost:8000/predict` with a test X-ray returns valid JSON
- [ ] All GitHub Actions workflows pass on a test PR
- [ ] Model is visible on HuggingFace Hub with a filled-out model card
- [ ] Gradio demo is live on HF Spaces with a public URL to share

---

## References

- FastAPI documentation: https://fastapi.tiangolo.com
- MLflow documentation: https://mlflow.org/docs/latest
- HuggingFace Hub documentation: https://huggingface.co/docs/hub
- GitHub Actions documentation: https://docs.github.com/en/actions
