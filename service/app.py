from __future__ import annotations

import os
from typing import Iterable, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest


app = FastAPI(title="Recommender API", version=os.getenv("MODEL_VERSION", "0.1"))
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


@app.get("/healthz", response_class=PlainTextResponse)
def healthz() -> PlainTextResponse:
    return PlainTextResponse("ok")


@app.get("/recommend")
@lat.time()
def recommend(user_id: int, k: int = 20) -> dict:
    try:
        items = _recommend_for_user(user_id, k, os.getenv("MODEL_PATH"))
        reqs.labels("200").inc()
        return {"user_id": user_id, "items": list(items)}
    except Exception as exc:  # pragma: no cover - defensive; surfaced via tests
        reqs.labels("500").inc()
        raise HTTPException(500, str(exc))


@app.get("/recommend/plain")
@lat.time()
def recommend_plain(user_id: int, k: int = 20) -> PlainTextResponse:
    try:
        items = _recommend_for_user(user_id, k, os.getenv("MODEL_PATH"))
        reqs.labels("200").inc()
        return PlainTextResponse(_as_plain_text(items), media_type="text/plain")
    except Exception as exc:  # pragma: no cover - defensive
        reqs.labels("500").inc()
        raise HTTPException(500, str(exc))


@app.get("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}