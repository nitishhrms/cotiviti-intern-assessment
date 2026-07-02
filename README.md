# Chest X-Ray Report Generation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange?logo=gradio)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface)

**Automated chest X-ray report generation using DenseNet-121 + Transformer decoder,
with a clinical RAG agent, DICOM support, and a full MLOps backend.**

[Demo](#gradio-ui) · [API Docs](#api-endpoints) · [RAG Agent](#rag-clinical-agent) · [HuggingFace](#huggingface-model)

</div>

---

## What This Project Does

| Input | Output |
|-------|--------|
| Chest X-ray (JPEG / PNG / DICOM) | Generated radiology report |
| Clinical question | Answer with reasoning trace (RAG agent) |

The system has four parts:

```
X-ray image
    └─► DenseNet-121 encoder  →  256-d feature vector
                                        │
                                        ▼
                           Transformer decoder  →  report text
                                        │
                                        ▼
                           RAG Agent (ClinicalBERT + ChromaDB + LangChain)
                                        │
                                        ▼
                           Clinical question answering
```

---

## Architecture

<details>
<summary><b>Image Encoder</b></summary>

- DenseNet-121 pretrained on ImageNet
- AdaptiveAvgPool2d → flattened to 1024-d
- Linear projection to 256-d embedding

</details>

<details>
<summary><b>Text Decoder</b></summary>

- 3-layer Transformer Decoder
- 8 attention heads, 256-d embeddings
- Greedy autoregressive decoding up to 100 tokens
- Trained on Indiana University Chest X-ray Collection (~3,900 studies)

</details>

<details>
<summary><b>RAG Clinical Agent</b></summary>

- **Embeddings**: ClinicalBERT (`medicalai/ClinicalBERT`)
- **Vector store**: ChromaDB (persistent, metadata filtering) with FAISS fallback
- **Agent framework**: LangChain ReAct loop
- **LLM**: vLLM (set `VLLM_BASE_URL`) or built-in DemoLLM for local testing
- **Tools**: `search_similar_cases`, `explain_finding`, `summarise_report`

</details>

<details>
<summary><b>MLOps Stack</b></summary>

| Component | Tool |
|-----------|------|
| API | FastAPI + Uvicorn |
| Async tasks | Celery + Redis |
| Metrics | Prometheus |
| Tracing | OpenTelemetry (OTLP → Jaeger / Grafana Tempo) |
| Experiment tracking | MLflow |
| Container | Docker + Docker Compose |
| CI/CD | GitHub Actions |

</details>

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the FastAPI backend

```bash
uvicorn backend.main:app --reload
# API available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### 3. Start the Gradio UI

```bash
python app.py
# UI available at http://localhost:7860
```

### 4. (Optional) Start Celery worker for async inference

```bash
celery -A backend.celery_app worker --loglevel=info
```

---

## Gradio UI

The UI has four tabs:

| Tab | What it does |
|-----|-------------|
| **Report Generation** | Upload JPEG/PNG or DICOM → get a generated report |
| **Clinical Agent** | Ask clinical questions → ReAct reasoning trace + answer |
| **Knowledge Base** | Add reports, semantic search, manage ChromaDB |
| **System & Telemetry** | Live health, Prometheus metrics, OpenTelemetry status |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/metadata` | Model architecture info |
| `GET` | `/metrics` | Prometheus metrics |
| `POST` | `/predict` | Inference from JPEG/PNG |
| `POST` | `/predict/dicom` | Inference from DICOM file |
| `POST` | `/preview/dicom` | Windowed DICOM preview (JPEG) |
| `GET` | `/task/{task_id}` | Poll async Celery task |
| `POST` | `/agent/query` | RAG agent question answering |
| `POST` | `/agent/search` | Semantic search over reports |
| `POST` | `/agent/index` | Add reports to vector store |
| `GET` | `/agent/stats` | ChromaDB collection stats |

Full interactive docs: `http://localhost:8000/docs`

---

## RAG Clinical Agent

The agent uses a **ReAct (Reason + Act)** loop:

```
Question: "What does cardiomegaly indicate?"

Step 1 →  Thought: search for similar cases
          Tool:    search_similar_cases("cardiomegaly findings")
          Obs:     [3 similar reports from ChromaDB]

Step 2 →  Thought: get clinical explanation
          Tool:    explain_finding("cardiomegaly")
          Obs:     "Enlargement of cardiac silhouette >50% of thoracic width..."

Final  →  Synthesised answer from both observations
```

### Add reports to the knowledge base

```bash
curl -X POST http://localhost:8000/agent/index \
  -H "Content-Type: application/json" \
  -d '{"reports": ["Cardiomegaly noted. Bilateral pleural effusions."], "metadatas": [{"view": "PA"}]}'
```

### Query the agent

```bash
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does cardiomegaly mean?", "return_trace": true}'
```

---

## DICOM Support

Supported window presets:

| Preset | Center | Width |
|--------|--------|-------|
| Lung | -600 HU | 1500 HU |
| Mediastinum | 40 HU | 400 HU |
| Bone | 400 HU | 1800 HU |
| Auto | from DICOM tags | from DICOM tags |

```bash
curl -X POST http://localhost:8000/predict/dicom \
  -F "dicom_file=@chest.dcm" \
  -F "window_preset=mediastinum"
```

---

## Docker

```bash
# Full stack (API + Redis + Celery)
docker-compose up

# Development with hot reload
docker-compose -f docker-compose.dev.yml up
```

---

## HuggingFace Model

> Model weights and vocab are available on HuggingFace:

[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-Space-yellow?logo=huggingface)](https://huggingface.co/spaces/nitishhrms/xray-report-generation)

> Live demo running on HuggingFace Spaces:
> **[https://huggingface.co/spaces/nitishhrms/xray-report-generation](https://huggingface.co/spaces/nitishhrms/xray-report-generation)**

To push model weights to HuggingFace Hub:

```bash
python scripts/push_to_hub.py \
  --repo-id nitishhrms/xray-report-generation \
  --token hf_your_token_here
```

---

## Project Structure

```
.
├── app.py                    # Gradio UI (4 tabs)
├── backend/
│   ├── main.py               # FastAPI app + all endpoints
│   ├── inference.py          # Model architecture + InferenceEngine
│   ├── rag_agent.py          # ClinicalBERT + ChromaDB + LangChain agent
│   ├── schemas.py            # Pydantic request/response models
│   ├── middleware.py         # Request ID + timing middleware
│   ├── telemetry.py          # OpenTelemetry setup
│   ├── celery_app.py         # Celery configuration
│   └── tasks.py              # Async inference tasks
├── src/
│   ├── data/
│   │   ├── dicom_loader.py   # DICOM parsing + windowing
│   │   └── dicom_metadata.py # Patient metadata extraction
│   └── models/
│       └── metadata_encoder.py  # Age/sex conditioning module
├── scripts/
│   ├── push_to_hub.py        # Upload model to HuggingFace
│   ├── promote_model.py      # MLflow model promotion
│   └── log_analysis.py       # Log parsing utilities
├── tests/
│   ├── test_api.py
│   └── test_dicom_loader.py
├── docs/                     # Phase-by-phase documentation
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Disclaimer

> This project is for **educational and research purposes only**.
> The model is not FDA-cleared and must not be used for clinical diagnosis or patient care.
