"""
API integration tests using pytest + httpx.AsyncClient.

Run with:
    pytest tests/test_api.py -v --tb=short
"""

import io
import pytest
import pytest_asyncio
from PIL import Image
from httpx import AsyncClient, ASGITransport

from backend.main import app


def _make_dummy_jpeg(size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=(128, 128, 128))
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_large_bytes(mb: int = 21) -> bytes:
    return b"0" * (mb * 1024 * 1024)


@pytest.fixture(scope="module")
def dummy_jpeg():
    return _make_dummy_jpeg()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_returns_200():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert "model_loaded" in body
    assert "version" in body
    assert "uptime_seconds" in body


# ---------------------------------------------------------------------------
# /metadata
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_metadata_returns_expected_keys():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/metadata")
    assert r.status_code == 200
    body = r.json()
    for key in ("model_version", "vocab_size", "training_dataset", "embed_dim"):
        assert key in body


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_predict_valid_jpeg(dummy_jpeg):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/predict",
            files={"file": ("test.jpg", dummy_jpeg, "image/jpeg")},
        )
    # May return 200 (model loaded) or 503 (model not in test env) — both are valid
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        body = r.json()
        assert "report" in body
        assert "processing_time_ms" in body


@pytest.mark.asyncio
async def test_predict_invalid_file_type():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_predict_oversized_file():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/predict",
            files={"file": ("big.jpg", _make_large_bytes(21), "image/jpeg")},
        )
    assert r.status_code == 413


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_metrics_endpoint_responds():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/metrics")
    # 200 if prometheus_client installed, 503 if not — both acceptable
    assert r.status_code in (200, 503)
