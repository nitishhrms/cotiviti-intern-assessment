# API Reference (Surface Level)

The system exposes a REST API built with **FastAPI**. Full interactive documentation
is auto-generated and served at **`/docs`** (Swagger UI) and **`/redoc`** when the
backend is running:

```bash
uvicorn backend.main:app --reload
# Swagger:  http://localhost:8000/docs
```

## Endpoints

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness probe and model-loaded status |
| `GET` | `/metadata` | Model architecture details (vocab size, dimensions, dataset) |
| `GET` | `/metrics` | Prometheus metrics (request counts, latency histogram) |

### Inference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Generate a report from a JPEG or PNG image |
| `POST` | `/predict/dicom` | Generate a report from a DICOM file, with windowing |
| `POST` | `/preview/dicom` | Return a windowed DICOM preview image (JPEG) |
| `GET`  | `/task/{task_id}` | Poll the status and result of an async (Celery) job |

### Agent (RAG)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST`   | `/agent/query` | Grounded clinical question answering (optional reasoning trace) |
| `POST`   | `/agent/search` | Semantic search over indexed reports |
| `POST`   | `/agent/index` | Add reports to the vector store |
| `DELETE` | `/agent/index` | Delete the vector store collection |
| `GET`    | `/agent/stats` | Vector store statistics (backend and count) |

## Notes

- **Sync / async:** `/predict` and `/predict/dicom` accept `async_mode=true` to queue
  the job via Celery; poll `/task/{task_id}` for the result.
- **Response shape:** inference responses include `report`, `confidence`,
  `processing_time_ms`, and `model_version` (DICOM responses also include
  `window_applied` and `patient_info`).
- **Limits:** uploads are capped at 20 MB; accepted types are `.jpg`, `.png`, `.dcm`.
- **Middleware:** every request passes through Request-ID, timing, CORS, and
  (optional) rate-limiting middleware.

> See also: [`deliverables/hld_architecture.png`](../deliverables/hld_architecture.png)
> for the high-level design of how these endpoints map to the services behind them.
