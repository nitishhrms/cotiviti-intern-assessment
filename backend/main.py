import asyncio
import io
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from backend.inference import engine, MODEL_VERSION
from backend.middleware import RequestIdMiddleware, TimingMiddleware
from backend.schemas import (
    DicomPredictionResponse,
    ErrorResponse,
    HealthResponse,
    MetadataResponse,
    PredictionResponse,
)
from backend.telemetry import configure_telemetry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("xray_api")

app = FastAPI(
    title="X-Ray Report Generation API",
    description="Production API for automated chest X-ray report generation.",
    version=MODEL_VERSION,
)

configure_telemetry(app)

app.add_middleware(RequestIdMiddleware)
app.add_middleware(TimingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=int(os.getenv("INFERENCE_WORKERS", "2")))

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    _RATE_LIMIT = os.getenv("RATE_LIMIT", "30/minute")
    _RATE_LIMITING = True
    logger.info("Rate limiting enabled: %s", _RATE_LIMIT)
except ImportError:
    _RATE_LIMITING = False
    logger.warning("slowapi not installed — rate limiting disabled.")

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

    _inference_counter = Counter("inference_count_total", "Total inference requests")
    _inference_latency = Histogram(
        "inference_latency_seconds", "Inference latency",
        buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )
    _error_counter = Counter("error_count_total", "Total errors")
    _PROMETHEUS = True
except ImportError:
    _PROMETHEUS = False

_START_TIME = time.time()

MAX_FILE_SIZE = 20 * 1024 * 1024
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "application/dicom", "application/octet-stream"}
DICOM_MAGIC = b"DICM"


def _is_dicom(data: bytes) -> bool:
    return len(data) > 132 and data[128:132] == DICOM_MAGIC


def _validate_upload(data: bytes, filename: str = ""):
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 20 MB limit.")
    ext = Path(filename).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png", ".dcm", ""):
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. Accepted: .jpg, .png, .dcm",
        )


async def _run_in_executor(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, fn, *args)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    return HealthResponse(
        status="healthy" if engine.is_loaded else "degraded",
        model_loaded=engine.is_loaded,
        version=MODEL_VERSION,
        uptime_seconds=round(time.time() - _START_TIME, 1),
    )


@app.get("/metadata", response_model=MetadataResponse, tags=["system"])
async def metadata():
    cfg = engine.config or {}
    return MetadataResponse(
        model_version=MODEL_VERSION,
        vocab_size=len(engine.word2idx) if engine.word2idx else 0,
        training_dataset="Indiana University Chest X-ray Collection",
        embed_dim=cfg.get("embed_dim", 256),
        num_heads=cfg.get("num_heads", 8),
        num_layers=cfg.get("num_layers", 3),
        max_len=cfg.get("max_len", 100),
    )


@app.get("/metrics", tags=["system"])
async def metrics():
    if not _PROMETHEUS:
        return JSONResponse(
            {"error": "prometheus_client not installed"},
            status_code=503,
        )
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(
    request: Request,
    file: UploadFile = File(...),
    async_mode: bool = Form(False),
):
    if _RATE_LIMITING:
        await limiter._check_request_limit(request, _RATE_LIMIT)

    data = await file.read()
    _validate_upload(data, file.filename or "")

    if async_mode:
        try:
            from backend.tasks import run_inference
            task = run_inference.delay(data.hex())
            return JSONResponse({"task_id": task.id, "status": "queued"})
        except Exception:
            logger.warning("Celery unavailable — falling back to sync inference.")

    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        if _PROMETHEUS:
            _inference_counter.inc()
        result = await _run_in_executor(engine.predict, data)
        if _PROMETHEUS:
            _inference_latency.observe(result["processing_time_ms"] / 1000)
        return PredictionResponse(**result)
    except Exception as exc:
        if _PROMETHEUS:
            _error_counter.inc()
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/dicom", response_model=DicomPredictionResponse, tags=["inference"])
async def predict_dicom(
    request: Request,
    dicom_file: UploadFile = File(...),
    patient_age: Optional[float] = Form(None),
    patient_sex: Optional[str] = Form(None),
    window_preset: str = Form("mediastinum"),
    async_mode: bool = Form(False),
):
    if _RATE_LIMITING:
        await limiter._check_request_limit(request, _RATE_LIMIT)

    data = await dicom_file.read()
    _validate_upload(data, dicom_file.filename or "")

    if not _is_dicom(data) and not (dicom_file.filename or "").lower().endswith(".dcm"):
        raise HTTPException(status_code=422, detail="File does not appear to be a valid DICOM.")

    if async_mode:
        try:
            from backend.tasks import run_dicom_inference
            task = run_dicom_inference.delay(data.hex(), window_preset)
            return JSONResponse({"task_id": task.id, "status": "queued"})
        except Exception:
            logger.warning("Celery unavailable — falling back to sync DICOM inference.")

    try:
        from src.data.dicom_loader import dicom_to_tensor

        tensor, patient_info, preset_used = await _run_in_executor(
            dicom_to_tensor, data, window_preset
        )
        if patient_age is not None:
            patient_info["age_normalized"] = min(patient_age / 100.0, 1.0)
        if patient_sex is not None:
            sex_map = {"male": 0, "m": 0, "female": 1, "f": 1}
            patient_info["sex_encoded"] = sex_map.get(patient_sex.lower(), 2)

        if not engine.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded.")

        result = await _run_in_executor(engine.predict_from_tensor, tensor)
        return DicomPredictionResponse(
            report=result["report"],
            confidence=result["confidence"],
            processing_time_ms=result["processing_time_ms"],
            model_version=result["model_version"],
            window_applied=preset_used,
            patient_info=patient_info,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("DICOM inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/preview/dicom", tags=["inference"])
async def preview_dicom(
    dicom_file: UploadFile = File(...),
    window_preset: str = Form("mediastinum"),
):
    data = await dicom_file.read()
    _validate_upload(data, dicom_file.filename or "")

    try:
        from src.data.dicom_loader import load_dicom, apply_window, WINDOW_PRESETS, auto_window

        result = await _run_in_executor(load_dicom, data)
        pixel_array = result["pixel_array"]
        ds = result["ds"]

        if window_preset == "auto":
            params = auto_window(ds)
        else:
            params = WINDOW_PRESETS.get(window_preset, WINDOW_PRESETS["mediastinum"])

        import numpy as np
        from PIL import Image as PILImage

        windowed = apply_window(pixel_array, params["center"], params["width"])
        if windowed.ndim == 3:
            windowed = windowed[0]

        pil_img = PILImage.fromarray(windowed, mode="L")
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        return Response(buf.getvalue(), media_type="image/jpeg")
    except Exception as exc:
        logger.exception("DICOM preview failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/task/{task_id}", tags=["inference"])
async def get_task_result(task_id: str):
    try:
        from celery.result import AsyncResult
        from backend.celery_app import celery_app

        result = AsyncResult(task_id, app=celery_app)
        if result.state == "PENDING":
            return {"task_id": task_id, "status": "pending"}
        elif result.state == "STARTED":
            return {"task_id": task_id, "status": "running"}
        elif result.state == "SUCCESS":
            return {"task_id": task_id, "status": "success", "result": result.result}
        elif result.state == "FAILURE":
            return {"task_id": task_id, "status": "failure", "error": str(result.result)}
        return {"task_id": task_id, "status": result.state}
    except ImportError:
        raise HTTPException(status_code=503, detail="Celery not installed.")


class AgentQueryRequest(BaseModel):
    question: str
    report_context: str = ""
    filters: dict = {}
    k: int = 3
    return_trace: bool = False


class IndexRequest(BaseModel):
    reports: list[str]
    metadatas: list[dict] = []


@app.post("/agent/query", tags=["agent"])
async def agent_query(body: AgentQueryRequest):
    try:
        from backend.rag_agent import query_agent_with_trace
        result = await _run_in_executor(
            query_agent_with_trace, body.question, body.report_context
        )
        if body.return_trace:
            return {
                "question": body.question,
                "answer": result["answer"],
                "langchain_active": result["langchain_active"],
                "steps": result["steps"],
            }
        return {"answer": result["answer"], "question": body.question}
    except Exception as exc:
        logger.exception("Agent query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/agent/search", tags=["agent"])
async def semantic_search(body: AgentQueryRequest):
    try:
        from backend.rag_agent import retrieve_similar_reports
        results = await _run_in_executor(
            retrieve_similar_reports, body.question, body.k,
            body.filters or None,
        )
        return {"results": results, "query": body.question}
    except Exception as exc:
        logger.exception("Semantic search failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/agent/index", tags=["agent"])
async def build_rag_index(body: IndexRequest):
    try:
        from backend.rag_agent import build_index
        await _run_in_executor(build_index, body.reports, body.metadatas or None)
        return {"status": "index built", "num_reports": len(body.reports)}
    except Exception as exc:
        logger.exception("Index build failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/agent/index", tags=["agent"])
async def delete_rag_index():
    try:
        from backend.rag_agent import delete_collection
        await _run_in_executor(delete_collection)
        return {"status": "collection deleted"}
    except Exception as exc:
        logger.exception("Collection delete failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/agent/stats", tags=["agent"])
async def rag_stats():
    try:
        from backend.rag_agent import _CHROMA_AVAILABLE, _get_chroma, _faiss_index
        if _CHROMA_AVAILABLE:
            _, collection = _get_chroma()
            return {
                "store": "chromadb",
                "num_reports": collection.count(),
                "collection": collection.name,
            }
        return {
            "store": "faiss",
            "num_reports": _faiss_index.ntotal if _faiss_index else 0,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
