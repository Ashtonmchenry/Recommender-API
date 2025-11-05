from __future__ import annotations

import os
import time
import asyncio
from pathlib import Path
from typing import Iterable, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, Response, JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest, Gauge

from recommender.avro_utils import load_parsed_schema, assert_valid


app = FastAPI(title="Recommender API")
reqs = Counter("recommend_requests_total", "requests", ["status"])
lat = Histogram("recommend_latency_seconds", "latency")

DEFAULT_RECOMMENDATIONS: List[int] = [50, 172, 1, 99, 87]
RESPONSE_SCHEMA = load_parsed_schema(Path("recommender/avro-schemas/RecoResponse.avsc"))

# Configurable limits (env-controlled)
MAX_CONCURRENCY     = int(os.getenv("MAX_CONCURRENCY", "16"))
ACQUIRE_TIMEOUT_SEC = float(os.getenv("ACQUIRE_TIMEOUT_SEC", "0.01"))

# Backpressure metrics
INFLIGHT = Gauge("http_inflight_requests", "In-flight HTTP requests")
REJECTS  = Counter("http_backpressure_rejections_total", "Requests rejected due to backpressure")

# Bounded concurrency
_sem = asyncio.BoundedSemaphore(MAX_CONCURRENCY)


def _recommend_for_user(user_id: int, k: int, model_path: str | None = None) -> List[int]:
    """Lookup recommendations for a user.

    This indirection makes it trivial to monkeypatch during tests and keeps the
    FastAPI route thin. When an actual model loader is wired in, this helper is
    the only place that needs to change.
    """

    # Until a production model is wired in, fall back to a deterministic stub.
    return DEFAULT_RECOMMENDATIONS[:k]


def _as_plain_text(items: Iterable[int]) -> str:
    """Serialize ids for plaintext responses."""

    return ",".join(str(i) for i in items)


@app.get("/health", response_model=dict)
def health() -> dict:
    return {"status": "ok", "version": os.getenv("MODEL_VERSION", "v0.1")}


# Health
@app.get("/healthz", response_class=PlainTextResponse)
def healthz() -> str:
    return "ok"


# For JSON results, FastAPI serializes lists/dicts (response_model needed):
@app.get("/recommend")
def recommend(user_id: int, k: int = 10):
    items = _recommend_for_user(user_id=user_id, k=k)
    payload = {
        "user_id": int(user_id),
        "items": [int(i) for i in items],
        "model_version": os.getenv("MODEL_VERSION", "v0.2"),
        "generated_at": int(time.time()),
    }
    # Enforce Avro schema at the service boundary
    assert_valid(RESPONSE_SCHEMA, payload)
    return payload

# For endpoints that return plain text ids:
@app.get("/recommend/plain", response_class=PlainTextResponse)
def recommend_plain(user_id: int, k: int = 10) -> str:
    items = _recommend_for_user(user_id=user_id, k=k)
    # Validate the same structure even for the plaintext variant (safety)
    _ = assert_valid(RESPONSE_SCHEMA, {
        "user_id": int(user_id),
        "items": [int(i) for i in items],
        "model_version": os.getenv("MODEL_VERSION", "v0.2"),
        "generated_at": int(time.time()),
    })
    return _as_plain_text(items)


# Metrics (Prometheus)
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.middleware("http")
async def backpressure_guard(request: Request, call_next):
    try:
        # non-blocking-ish acquire (tiny timeout) -> reject fast if saturated
        await asyncio.wait_for(_sem.acquire(), timeout=ACQUIRE_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        REJECTS.inc()
        return JSONResponse({"detail": "Backpressure: server busy"}, status_code=503)

    INFLIGHT.inc()
    try:
        with lat.time():  # reuse your Histogram to time total request latency
            resp = await call_next(request)
        # optional: label your request counter
        reqs.labels(status=str(getattr(resp, "status_code", 200))).inc()
        return resp
    finally:
        INFLIGHT.dec()
        _sem.release()