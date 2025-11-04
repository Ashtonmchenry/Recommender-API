from __future__ import annotations

import os
from typing import Iterable, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest


app = FastAPI(title="Recommender API")
reqs = Counter("recommend_requests_total", "requests", ["status"])
lat = Histogram("recommend_latency_seconds", "latency")

DEFAULT_RECOMMENDATIONS: List[int] = [50, 172, 1, 99, 87]


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
    return {"user_id": user_id, "items": items}

# For endpoints that return plain text ids:
@app.get("/recommend/plain", response_class=PlainTextResponse)
def recommend_plain(user_id: int, k: int = 10) -> str:
    items = _recommend_for_user(user_id=user_id, k=k)
    return _as_plain_text(items)


# Metrics (Prometheus)
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)