# Deployment Guide

## Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `HF_TOKEN` | HuggingFace API token (Settings → Access Tokens) |
| `RENDER_DEPLOY_HOOK` | Render.com deploy webhook URL (optional) |

## Required GitHub Variables

| Variable | Description |
|----------|-------------|
| `HF_REPO_ID` | e.g. `your-username/xray-report-generator` |

## Local Development

```bash
# Start all services with hot reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# API:     http://localhost:8000
# MLflow:  http://localhost:5000
# Gradio:  http://localhost:7860
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379/0` | Celery broker URL |
| `INFERENCE_WORKERS` | `2` | Thread pool size for async inference |
| `OTEL_ENABLED` | `false` | Enable OpenTelemetry tracing |
| `OTEL_SERVICE_NAME` | `xray-report-api` | Service name in traces |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | OTLP collector endpoint |
| `VLLM_BASE_URL` | _(empty)_ | vLLM server URL for the RAG agent |
| `VLLM_MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | Model served by vLLM |
| `RATE_LIMIT` | `30/minute` | slowapi rate limit per IP |
| `MLFLOW_MODEL_NAME` | `xray-report-generator` | MLflow registered model name |
| `MIN_BLEU4` | `0.10` | Promotion threshold for BLEU-4 |
| `MIN_AUC_ROC` | `0.80` | Promotion threshold for AUC-ROC |

## Celery Workers

Start a worker separately (outside Docker):
```bash
celery -A backend.celery_app worker --loglevel=info --concurrency=2
```

## vLLM (Clinical Agent)

To enable the LangChain agent with fast inference:
```bash
# Start vLLM server (requires GPU)
vllm serve mistralai/Mistral-7B-Instruct-v0.2 --port 8001

# Then set the env var
export VLLM_BASE_URL=http://localhost:8001
```
