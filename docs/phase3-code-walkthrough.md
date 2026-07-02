# Phase 3 + Phase 5 + Phase 7 — Complete Code Walkthrough

This document explains every file created, what it does, why it exists,
and how it connects to the rest of the system.

---

## The Big Picture

Before Phase 3, the project was just `app.py` — a Gradio interface that loaded
a model and generated reports. That is fine for demos but not for production.

Phase 3 converts it into a real system:

```
Before:
  app.py → model → report (Gradio only)

After:
  FastAPI backend  ←→  Celery workers  ←→  Redis
       ↕                                      ↕
  Gradio frontend      MLflow tracking    FAISS + ClinicalBERT
       ↕
  DICOM support (Phase 5)
       ↕
  LangChain Agent (Phase 7)
```

---

## File-by-File Explanation

---

### `backend/inference.py`

**What it does:**
Loads the model ONCE when the server starts and keeps it in memory.
Exposes a `predict()` method that takes image bytes and returns a report.

**Why it exists:**
In the original `app.py`, the model loads at module import time — fine for
a single script. But in FastAPI, you need a controlled object that:
- Loads the model exactly once (not on every request)
- Can report whether it loaded successfully
- Can be tested independently

**Key concept — InferenceEngine class:**
```python
class InferenceEngine:
    def __init__(self):
        self._load()          # runs once at startup

    def predict(self, image_bytes: bytes) -> dict:
        # converts bytes → PIL image → tensor → report
        return {"report": "...", "processing_time_ms": 89}
```

**Key concept — BASE_DIR:**
```python
BASE_DIR = Path(__file__).resolve().parent.parent
```
This finds the project root regardless of where you run the script from.
`__file__` is `backend/inference.py`, `.parent` is `backend/`, `.parent.parent` is the root.

**Module-level singleton:**
```python
engine = InferenceEngine()   # created once when Python imports this file
```
FastAPI imports this file once → engine loads once → all requests share it.

---

### `backend/main.py`

**What it does:**
The FastAPI application. Defines all HTTP endpoints, adds middleware,
configures rate limiting, Prometheus metrics, and OpenTelemetry.

**Why FastAPI:**
- Auto-generates Swagger docs at `/docs` (what you used to test)
- Validates request/response shapes automatically via Pydantic
- Async by default — handles many requests without blocking
- Much faster than Flask for ML APIs

**Endpoints explained:**

```
GET  /health        → is the server alive? is the model loaded?
GET  /metadata      → what model is this? what was it trained on?
GET  /metrics       → Prometheus counters (requests, latency, errors)
POST /predict       → upload JPEG/PNG → get report
POST /predict/dicom → upload .dcm file → get report + patient info
POST /preview/dicom → upload .dcm file → get windowed image as JPEG
POST /agent/index   → feed reports into FAISS index
POST /agent/query   → ask a clinical question → get answer from RAG agent
GET  /task/{id}     → check status of async Celery job
```

**Async inference — why it matters:**
```python
# WRONG — blocks the entire server while model runs:
result = engine.predict(data)

# RIGHT — runs in background thread, server stays responsive:
result = await loop.run_in_executor(_executor, engine.predict, data)
```
FastAPI is async but PyTorch is synchronous. `run_in_executor` puts the
blocking work on a thread so other requests can still be handled.

**File validation:**
```python
MAX_FILE_SIZE = 20 * 1024 * 1024   # 20MB
```
Rejects files that are too large (prevents memory exhaustion attacks)
and rejects wrong file types (prevents processing garbage data).

**CORS middleware:**
```python
app.add_middleware(CORSMiddleware, allow_origins=["*"])
```
Allows the Gradio frontend (running on port 7860) to call the API
(running on port 8000). Without this, browsers block cross-origin requests.

---

### `backend/schemas.py`

**What it does:**
Defines the exact shape of every API response using Pydantic models.

**Why it exists:**
Without schemas, you could return any random dict and the API would accept it.
With schemas, FastAPI validates that every response has the right fields and types.

```python
class PredictionResponse(BaseModel):
    report: str                # must be a string
    confidence: float          # must be a float
    processing_time_ms: int    # must be an integer
    model_version: str         # must be a string
```

If your code tries to return `{"report": 123}` (wrong type),
FastAPI raises an error immediately instead of sending bad data to the client.

---

### `backend/middleware.py`

**What it does:**
Two pieces of code that run on EVERY request, before and after the handler.

**RequestIdMiddleware:**
```python
request_id = str(uuid.uuid4())   # e.g. "738f5dd2-c011-4227-8e8a-e71da8e29d95"
response.headers["X-Request-ID"] = request_id
```
Attaches a unique ID to every request. When something breaks, you can
search logs for that ID to see exactly what happened. You saw this in
the response headers during testing.

**TimingMiddleware:**
```python
start = time.perf_counter()
response = await call_next(request)
elapsed_ms = int((time.perf_counter() - start) * 1000)
response.headers["X-Processing-Time-Ms"] = str(elapsed_ms)
```
Measures how long each request took. You saw `x-processing-time-ms: 91`
in your test — that 91ms was the DenseNet + Transformer inference time.

**Middleware chain — how it works:**
```
Request arrives
    → RequestIdMiddleware.dispatch() starts
        → TimingMiddleware.dispatch() starts
            → actual route handler runs
        → TimingMiddleware adds timing header
    → RequestIdMiddleware adds request ID header
Response sent
```

---

### `backend/celery_app.py`

**What it does:**
Configures Celery — a task queue system for running jobs asynchronously.

**Why it exists:**
Imagine 100 users upload X-rays at the same time. If each request waits
for inference (89ms each), the 100th user waits 8.9 seconds. With Celery:
- All 100 requests return instantly with a task_id
- Workers process them in parallel
- Users poll `/task/{id}` for results

**How it works:**
```
User → POST /predict (async_mode=true)
     → FastAPI puts job in Redis queue
     → returns {"task_id": "abc123"} immediately

[Background]
Celery Worker → picks job from Redis
             → runs engine.predict()
             → stores result in Redis

User → GET /task/abc123
     → FastAPI reads result from Redis
     → returns {"report": "..."}
```

**Redis is the broker:**
Redis is just a fast in-memory database. Celery uses it as a mailbox:
- FastAPI writes jobs to Redis
- Workers read jobs from Redis
- Workers write results back to Redis

---

### `backend/tasks.py`

**What it does:**
Defines the actual Celery task functions that workers execute.

```python
@celery_app.task(bind=True, name="backend.tasks.run_inference")
def run_inference(self, image_bytes_hex: str) -> dict:
    image_bytes = bytes.fromhex(image_bytes_hex)
    return engine.predict(image_bytes)
```

**Why hex encoding:**
Celery serializes tasks as JSON. Raw bytes are not JSON-serializable.
Converting to hex string (`bytes.fromhex`) makes them JSON-safe.

**Retry logic:**
```python
self.retry(exc=exc, countdown=2, max_retries=2)
```
If inference fails (e.g. GPU OOM), Celery automatically retries
2 times with a 2-second delay before giving up.

---

### `backend/telemetry.py`

**What it does:**
Sets up OpenTelemetry — the industry standard for observability.

**What observability means:**
In production, when something is slow or broken, you need to know:
- Which request was slow?
- Which part of the code was slow? (loading image? running model? serializing?)
- How does latency change over time?

OpenTelemetry answers these with "traces" — a tree of "spans":
```
Request: POST /predict  (91ms total)
  ├── validate_file     (1ms)
  ├── run_in_executor   (88ms)
  │     ├── load_image  (5ms)
  │     ├── transform   (3ms)
  │     └── model.generate (80ms)
  └── serialize_response (2ms)
```

**Why it's opt-in:**
OpenTelemetry requires an external collector (Jaeger, Grafana Tempo).
Setting `OTEL_ENABLED=false` (default) means it's a no-op — zero overhead
until you actually set up the infrastructure.

---

### `backend/rag_agent.py`

**What it does:**
The full Phase 7 clinical AI agent. Three components work together:

#### Component 1: ClinicalBERT Encoder
```python
def encode_texts(texts: list[str]) -> np.ndarray:
    # converts text → 768-dimensional float vector
    # "pleural effusion" → [0.23, -0.41, 0.87, ...]
```
ClinicalBERT is BERT trained on MIMIC-III hospital notes. It understands
medical language. "Effusion" and "fluid accumulation" become similar vectors.
General BERT would not understand these terms in clinical context.

#### Component 2: FAISS Index
```python
def build_index(reports: list[str]):
    embeddings = encode_texts(reports)   # shape: (N, 768)
    index = faiss.IndexFlatL2(768)
    index.add(embeddings)                # stores all vectors in RAM
```
FAISS (Facebook AI Similarity Search) stores the ClinicalBERT vectors.
When a query arrives:
```python
q_emb = encode_texts([query])           # query → vector
_, indices = index.search(q_emb, k=3)   # find 3 nearest vectors
```
It finds similar reports in microseconds using vector math,
not keyword matching. "heart enlargement" will find "cardiomegaly" reports
because they mean the same thing in the vector space.

#### Component 3: LangChain Agent (activates when langchain is installed)
```
User question
    → Agent thinks: "I need to search for similar cases"
    → Agent calls tool: search_similar_cases("cardiomegaly elderly")
    → FAISS returns 3 similar reports
    → Agent thinks: "Now I can answer"
    → Agent calls tool: explain_finding("cardiomegaly")
    → Agent combines everything into a final answer
```
The agent has 3 tools:
- `search_similar_cases` — searches FAISS index
- `explain_finding` — looks up clinical explanation
- `summarise_report` — condenses a long report

Without LangChain: just returns the raw FAISS results (what you tested).
With LangChain: reasons over results and gives a synthesized answer.

---

### `src/data/dicom_loader.py`

**What it does:**
Converts a DICOM file into a PyTorch tensor ready for the model.

**DICOM format explained:**
A DICOM file is not just an image. It contains:
- Pixel data in Hounsfield Units (HU), not 0-255
- HU = -1000 (air), 0 (water), +400 (bone), +3000 (dense bone)
- Patient metadata (age, sex, scanner settings)
- Acquisition parameters (slice thickness, pixel spacing)

**Windowing explained:**
Your monitor shows 256 shades. HU range is -1000 to +3000 = 4000 values.
You need to pick WHICH 256 values to show — this is the window.

```python
def apply_window(hu_array, window_center, window_width):
    lo = window_center - window_width / 2    # e.g. 40 - 200 = -160
    hi = window_center + window_width / 2    # e.g. 40 + 200 = +240
    clipped = np.clip(hu_array, lo, hi)      # values outside range → black or white
    normalized = (clipped - lo) / (hi - lo) * 255   # map to 0-255
```

Different windows reveal different anatomy:
```
Lung window      (center=-600, width=1500): air spaces, pneumothorax visible
Mediastinum      (center=40,   width=400):  heart, vessels, pleural fluid
Bone window      (center=400,  width=1800): rib fractures, vertebrae
```

**Photometric interpretation:**
Some scanners store images inverted (MONOCHROME1 = bright is dark).
The loader detects this and flips the values automatically.

**Full pipeline:**
```
.dcm file
  → pydicom.dcmread()        parse the file
  → pixel_array * slope + intercept   convert to HU
  → apply_window()           map HU to 0-255
  → PIL.Image                wrap as image
  → transforms.ToTensor()    convert to float tensor
  → Normalize(imagenet)      apply ImageNet normalization
  → (1, 3, 224, 224)         ready for DenseNet
```

---

### `src/data/dicom_metadata.py`

**What it does:**
Extracts and normalizes patient information from DICOM tags.

**DICOM age format is unusual:**
```
"045Y" → 45 years
"006M" → 6 months = 0.5 years
"002W" → 2 weeks
```
The `_parse_age()` function handles all these formats.

**Normalization:**
```python
age_normalized = age_years / 100.0   # 45 years → 0.45
sex_encoded = {"M": 0, "F": 1, "O": 2}[sex]
```
Values are normalized to [0,1] so they work well as neural network inputs.

**Why None handling matters:**
Not all DICOM files have all fields. A 1990s scanner might not store PatientAge.
The extractor returns `None` for missing fields instead of crashing.
The model then uses a zero vector for unknown metadata.

---

### `src/models/metadata_encoder.py`

**What it does:**
Converts patient age and sex into a 256-dimensional vector that can be
added to the image embedding before the decoder.

**Architecture:**
```
age (float 0-1) → Linear(1→32) → ReLU → Linear(32→64) → 64-d vector
sex (int 0/1/2) → Embedding(3, 64)                     → 64-d vector
                  concatenate → [age_emb, sex_emb]      → 128-d vector
                  Linear(128→256)                       → 256-d vector
```

**Why Embedding for sex:**
Sex is categorical (3 classes), not continuous. An embedding layer learns
a separate 64-d representation for each class, which works better than
using 0/1/2 as raw numbers in a Linear layer.

**How it integrates with the model (when retrained):**
```python
image_features = self.encoder(image)         # (batch, 256)
metadata_vec   = self.metadata_encoder(age, sex)  # (batch, 256)
combined       = image_features + metadata_vec    # element-wise add
# combined goes into the decoder instead of image_features alone
```

---

### `backend/celery_app.py` + `docker-compose.yml` — Redis

**What Redis is:**
Redis is an in-memory key-value store. Think of it as a Python dict
that lives in its own process and survives server restarts.

```python
# Celery writes to Redis:
redis["task:abc123"] = {"status": "pending"}

# Worker reads from Redis:
job = redis.lpop("celery_queue")

# Worker writes result to Redis:
redis["task:abc123"] = {"status": "done", "result": {"report": "..."}}
```

**Why in-memory:**
Redis keeps everything in RAM, making reads/writes microsecond-fast.
It's used as a job queue (not long-term storage) so RAM is fine.

---

### `Dockerfile` — Multi-stage build

**Why two stages:**
```dockerfile
# Stage 1: Builder — has gcc, pip, build tools
FROM python:3.11-slim AS builder
RUN pip install --prefix=/install -r requirements.txt
# This stage is DISCARDED after build

# Stage 2: Runtime — clean image, no build tools
FROM python:3.11-slim AS runtime
COPY --from=builder /install /usr/local   # only copy installed packages
```

Build tools (gcc, make, etc.) are needed to compile some Python packages
but not needed to RUN them. Discarding stage 1 cuts image size from
~1.2GB to ~500MB — faster to download, smaller attack surface.

---

### `docker-compose.yml`

**What it does:**
Defines 5 services that work together, all started with one command:
`docker-compose up`

```
api      — FastAPI server (port 8000)
worker   — Celery worker (processes inference jobs from Redis)
redis    — Message broker + result backend
mlflow   — Experiment tracking UI (port 5000)
frontend — Gradio interface (port 7860)
```

**Health checks:**
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
```
Docker restarts the container automatically if `/health` fails 3 times.
This is how services stay up 24/7 without manual intervention.

**Depends_on with condition:**
```yaml
api:
  depends_on:
    redis:
      condition: service_healthy
```
FastAPI won't start until Redis is confirmed healthy. Without this,
FastAPI might start before Redis is ready and crash immediately.

---

### GitHub Actions Workflows

**test.yml — runs on every Pull Request:**
```
You create a PR
  → GitHub automatically:
      1. Checks out your code
      2. Installs Python 3.11
      3. Caches pip packages (faster on repeated runs)
      4. Runs: pytest tests/ -v
      5. Blocks merge if tests fail
```
This prevents broken code from ever reaching the main branch.

**build.yml — runs on merge to main:**
```
Code merges to main
  → GitHub automatically:
      1. Builds Docker image
      2. Starts container
      3. Checks: curl http://localhost:8000/health
      4. If healthy → tags image as "latest"
      5. Pushes to GitHub Container Registry (ghcr.io)
```

**deploy.yml — runs when you create a version tag:**
```
git tag v1.0.0 && git push --tags
  → GitHub automatically:
      1. Runs push_to_hub.py → uploads model to HuggingFace
      2. Triggers Render.com redeploy (optional)
```

---

### `scripts/promote_model.py`

**What it does:**
Checks MLflow for the latest model in "Staging" and promotes it to
"Production" only if quality thresholds are met.

```python
if bleu4 >= 0.10 and auc_roc >= 0.80:
    client.transition_model_version_stage(
        name="xray-report-generator",
        version=version,
        stage="Production"
    )
```

**Why this matters:**
Without this, a worse model could accidentally replace a better one.
This gate ensures only improved models reach production.

---

### `scripts/log_analysis.py`

**What it does:**
Reads JSON log files and prints a dashboard:
```
Total requests  : 1423
Errors          : 12  (0.8%)
Avg latency     : 94 ms
Top 3 pathologies:
  cardiomegaly                   342
  no acute cardiopulmonary       289
  pleural effusion               201
Requests per hour (last 10):
  2026-04-03 09  ████████████ 124
  2026-04-03 10  ████████████████████ 203
```

---

### `app.py` — Updated Gradio UI (Phase 5)

**What changed:**
The original Gradio interface had one input (image) and one output (report).

New interface adds:
```python
dicom_input    = gr.File(file_types=[".dcm"])      # DICOM file upload
age_input      = gr.Number(label="Patient Age")    # 0-120
sex_input      = gr.Radio(["Male","Female","Not Specified"])
window_input   = gr.Dropdown(["Lung","Mediastinum","Bone","Auto"])
meta_output    = gr.Textbox(label="DICOM Metadata")  # shows extracted tags
```

**DICOM handling in app.py:**
```python
if dicom_file is not None:
    pil_image, meta_info, _ = _load_dicom_to_pil(dicom_bytes, preset_key)
    image = pil_image   # treat windowed DICOM as regular image from here
```
The DICOM is converted to a PIL image using the chosen window preset,
then the rest of inference runs exactly as before. This is called
"abstraction" — the model doesn't know or care that it came from a DICOM.

---

## How Everything Connects — Full Request Flow

```
1. User uploads chest X-ray to Gradio UI (port 7860)

2. Gradio calls predict_report(image, age, sex, window, dicom_file)

3. If DICOM:
   dicom_loader.py → parse pixels → apply window → PIL image
   dicom_metadata.py → extract age/sex from DICOM tags

4. PIL image → _transform → (1, 3, 224, 224) tensor

5. InferenceEngine.predict_from_tensor(tensor)
   ImageEncoder (DenseNet-121):
     → features.denseblock1-4 → 1024 feature maps
     → AdaptiveAvgPool2d → (1024,)
     → Linear → (256,)  ← image embedding

   TextDecoder (Transformer):
     → starts with <start> token
     → each step: attention over image embedding → predict next token
     → stops at <end> token
     → ["no", "acute", "cardiopulmonary", "abnormality"] → join → report

6. Return {"report": "...", "processing_time_ms": 89}

7. User asks follow-up via /agent/query:
   ClinicalBERT encodes question → 768-d vector
   FAISS searches index → 3 similar reports
   (LangChain agent reasons over results when installed)
   → answer returned
```

---

## Key Concepts Summary

| Concept | What it is | Why it matters |
|---------|-----------|----------------|
| InferenceEngine | Singleton that holds the model | Load once, serve thousands |
| Pydantic schemas | Type-safe response models | Catch bugs before they reach users |
| Middleware | Code that wraps every request | Logging, timing, IDs without touching route code |
| Thread pool executor | Runs blocking code without blocking async | Server stays responsive during inference |
| Celery + Redis | Async job queue | Handle traffic spikes without timeouts |
| FAISS | Vector similarity search | Find similar reports in microseconds |
| ClinicalBERT | Medical-domain text encoder | Understands "effusion" = "fluid accumulation" |
| Windowing | Map HU values to 0-255 | Make DICOM pixels visible and model-ready |
| Multi-stage Docker | Build then discard build tools | Smaller, safer production image |
| GitHub Actions | Automated test + build + deploy | No manual work when shipping new versions |
| OpenTelemetry | Distributed tracing | See where time is spent in production |
