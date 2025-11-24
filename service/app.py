from __future__ import annotations

"""FastAPI application exposing the recommender service with hot swapping."""

import asyncio
import json
import logging
import os
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, Info, generate_latest

from recommender.avro_utils import assert_valid, load_parsed_schema
from recommender.pipeline.serialize import load_model

# ----------------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------------
logger = logging.getLogger("service.app")
if not logger.handlers:
    # Configure root logger once (uvicorn injects its own handlers in prod)
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# ----------------------------------------------------------------------------
# FastAPI app + base metrics
# ----------------------------------------------------------------------------
app = FastAPI(title="Recommender API")

# Request/latency metrics: buckets allow 95p calculations in Prometheus queries.
reqs = Counter("recommend_requests_total", "Recommendation requests", ["status"])
lat = Histogram(
    "recommend_latency_seconds",
    "End-to-end recommendation latency",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 2, 5),
)

# Model lifecycle + error tracking metrics.
MODEL_SWITCHES = Counter(
    "model_switch_total",
    "Number of hot swaps performed",
    ["version"],
)
MODEL_LOADS = Counter(
    "model_registry_load_total",
    "Model artifacts loaded from registry",
    ["version"],
)
MODEL_ERRORS = Counter(
    "model_inference_errors_total",
    "Errors encountered during recommendation scoring",
)
MODEL_INFO = Info(
    "model_active_info",
    "Metadata for the currently active model",
)
MODEL_AGE = Gauge(
    "model_active_age_seconds",
    "Age (seconds) of the currently active model artifact",
)

# ----------------------------------------------------------------------------
# Static configuration / defaults
# ----------------------------------------------------------------------------
DEFAULT_RECOMMENDATIONS: list[int] = [50, 172, 1, 99, 87]
RESPONSE_SCHEMA = load_parsed_schema(Path("recommender/avro-schemas/RecoResponse.avsc"))

MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "16"))
ACQUIRE_TIMEOUT_SEC = float(os.getenv("ACQUIRE_TIMEOUT_SEC", "0.01"))

MODEL_REGISTRY_ROOT = Path(os.getenv("MODEL_REGISTRY_ROOT", "model_registry"))
MODEL_ARTIFACT_NAME = os.getenv("MODEL_ARTIFACT_NAME", "model.joblib")
MODEL_METADATA_NAME = os.getenv("MODEL_METADATA_NAME", "meta.yaml")
INITIAL_MODEL_VERSION = os.getenv("MODEL_VERSION", "v0.2")

# Concurrency / backpressure metrics
INFLIGHT = Gauge("http_inflight_requests", "In-flight HTTP requests")
REJECTS = Counter("http_backpressure_rejections_total", "Requests rejected due to backpressure")

# Limit the number of concurrent requests we execute.
_sem = asyncio.BoundedSemaphore(MAX_CONCURRENCY)

# ----------------------------------------------------------------------------
# Model registry utilities
# ----------------------------------------------------------------------------

def _parse_created_at(raw: str | None) -> float | None:
    """Parse ISO-8601 timestamps from metadata and return epoch seconds."""

    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()

def _parse_metadata_text(text: str) -> dict[str, Any]:
    """Best-effort YAML/JSON parser without requiring PyYAML."""

    if not text.strip():
        return {}

    try:  # Prefer YAML if available.
        import yaml  # type: ignore
    except ModuleNotFoundError:
        pass
    else:  # pragma: no cover - exercised when PyYAML present
        try:
            parsed = yaml.safe_load(text)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            return parsed

    # JSON fallback
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    result: dict[str, Any] = {}
    for line in text.splitlines():
        if not line or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        val = value.strip().strip('"')
        if not val:
            result[key] = ""
            continue
        if val.lower() in {"null", "none"}:
            result[key] = None
            continue
        try:
            result[key] = json.loads(val)
        except json.JSONDecodeError:
            result[key] = val
    return result


@dataclass
class ModelBundle:
    """In-memory representation of a model artifact."""

    version: str
    path: Path
    model: Any
    metadata: dict[str, Any]
    inverse_item_map: dict[int, int]
    cold_start_ranking: list[int]

    def recommend(self, user_id: int, k: int) -> list[int]:
        """Return top-k recommendations for ``user_id`` using the cached model."""

        k = max(0, min(int(k), len(self.cold_start_ranking)))
        if k == 0:
            return []

        uidx = self.model.user_map.get(int(user_id))
        if uidx is None:
            # Cold-start: rely on popularity proxy computed at load time.
            return self.cold_start_ranking[:k]

        # Score via dot product between user and item latent factors.
        scores = self.model.user_factors[uidx] @ self.model.item_factors.T
        top_indices = np.argpartition(scores, -k)[-k:]
        ordered = top_indices[np.argsort(-scores[top_indices])]
        return [int(self.inverse_item_map[int(i)]) for i in ordered[:k]]
class ModelRegistry:
    """Thread-safe loader/cache for model artifacts in ``model_registry/``."""

    def __init__(self, root: Path, artifact_name: str, metadata_name: str) -> None:
        self._root = root
        self._artifact_name = artifact_name
        self._metadata_name = metadata_name
        self._lock = threading.RLock()
        self._bundles: dict[str, ModelBundle] = {}
        self._active_version: str | None = None

    # ---- helpers -----------------------------------------------------------------
    def _bundle_dir(self, version: str) -> Path:
        return self._root / version

    def _load_metadata(self, version_dir: Path) -> dict[str, Any]:
        meta_path = version_dir / self._metadata_name
        if not meta_path.exists():
            return {}
        try:
            text = meta_path.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - defensive path
            logger.warning("Failed to read metadata %s: %s", meta_path, exc)
            return {}
        try:
            return _parse_metadata_text(text)
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning("Failed to parse metadata %s: %s", meta_path, exc)
            return {}

    def _materialize_bundle(self, version: str) -> ModelBundle:
        version_dir = self._bundle_dir(version)
        artifact_path = version_dir / self._artifact_name
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact missing: {artifact_path}")

        logger.info("Loading model version %s from %s", version, artifact_path)
        import sys

        from recommender import train as train_module

        if "__main__" in sys.modules:
            setattr(sys.modules["__main__"], "ALSModel", train_module.ALSModel)
        model = load_model(str(artifact_path))
        inverse_map = {internal: external for external, internal in model.item_map.items()}
        # Popularity proxy for cold-start scenarios.
        cold_scores = np.linalg.norm(model.item_factors, axis=1)
        cold_order = np.argsort(-cold_scores)
        cold_ids = [int(inverse_map[int(i)]) for i in cold_order]

        metadata = self._load_metadata(version_dir)
        bundle = ModelBundle(
            version=version,
            path=artifact_path,
            model=model,
            metadata=metadata,
            inverse_item_map=inverse_map,
            cold_start_ranking=cold_ids,
        )
        MODEL_LOADS.labels(version=version).inc()
        return bundle

    # ---- public API ---------------------------------------------------------------
    def get(self, version: str | None = None) -> ModelBundle:
        """Return a ``ModelBundle`` for ``version`` (or the active version)."""

        with self._lock:
            target = version or self._active_version
            if not target:
                raise LookupError("No active model version configured")
            cached = self._bundles.get(target)
            if cached is not None:
                return cached

        with self._lock:
            # Double-checked locking: ensure only one thread loads each version.
            cached = self._bundles.get(target)
            if cached is None:
                cached = self._materialize_bundle(target)
                self._bundles[target] = cached
            return cached

    def activate(self, version: str) -> ModelBundle:
        """Mark ``version`` as active and update instrumentation."""

        bundle = self.get(version)
        with self._lock:
            self._active_version = bundle.version
        MODEL_SWITCHES.labels(version=bundle.version).inc()
        self._update_model_metrics(bundle)
        return bundle

    def active_version(self) -> str | None:
        with self._lock:
            return self._active_version

    def _update_model_metrics(self, bundle: ModelBundle) -> None:
        metadata = bundle.metadata or {}
        info_payload = {
            "version": bundle.version,
            "artifact": bundle.path.name,
        }
        metrics = metadata.get("metrics", {})
        if "hr@20" in metrics:
            info_payload["hr_at_20"] = f"{metrics['hr@20']}"
        if "ndcg@20" in metrics:
            info_payload["ndcg_at_20"] = f"{metrics['ndcg@20']}"
        MODEL_INFO.info(info_payload)

        created_at = metadata.get("created_at") or metadata.get("trained_at")
        epoch = _parse_created_at(created_at)
        if epoch is None:
            MODEL_AGE.set(0)
        else:
            MODEL_AGE.set(max(0.0, time.time() - epoch))


# Instantiate the registry and attempt to load the default version once at import.
_REGISTRY = ModelRegistry(MODEL_REGISTRY_ROOT, MODEL_ARTIFACT_NAME, MODEL_METADATA_NAME)
try:
    _REGISTRY.activate(INITIAL_MODEL_VERSION)
except (FileNotFoundError, LookupError) as exc:
    logger.warning("Unable to load initial model '%s': %s", INITIAL_MODEL_VERSION, exc)

def configure_registry(
    root: Path | str,
    version: str | None = None,
    artifact_name: str | None = None,
) -> ModelRegistry:
    """Utility used by tests/scripts to reconfigure the registry on the fly."""

    global _REGISTRY
    _REGISTRY = ModelRegistry(Path(root), artifact_name or MODEL_ARTIFACT_NAME, MODEL_METADATA_NAME)
    if version:
        try:
            _REGISTRY.activate(version)
        except Exception as exc:  # pragma: no cover - helper used in tests
            logger.warning("Failed to activate %s from %s: %s", version, root, exc)
    return _REGISTRY

# ----------------------------------------------------------------------------
# Recommendation helpers
# ----------------------------------------------------------------------------
def _recommend_for_user(
    user_id: int,
    k: int,
    model_version: str | None = None,
) -> tuple[list[int], str | None]:
    """Generate recommendations and return (items, resolved_model_version)."""

    try:
        bundle = _REGISTRY.get(model_version)
    except LookupError:
        logger.warning("No active model; falling back to defaults")
        return DEFAULT_RECOMMENDATIONS[:k], None
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive logging path
        MODEL_ERRORS.inc()
        logger.exception("Unexpected error loading model", exc_info=exc)
        raise HTTPException(status_code=500, detail="Failed to load model") from exc

    try:
        return bundle.recommend(user_id, k), bundle.version
    except Exception as exc:  # pragma: no cover - defensive logging path
        MODEL_ERRORS.inc()
        logger.exception("Recommendation failure for model %s", bundle.version)
        raise HTTPException(status_code=500, detail="Failed to score recommendation") from exc

def _as_plain_text(items: Iterable[int]) -> str:
    """Serialize ids for plaintext responses."""

    return ",".join(str(i) for i in items)

# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/health", response_model=dict)
def health() -> dict:
    """Lightweight health endpoint consumed by Cloud Run / probes."""

    version = _REGISTRY.active_version() or INITIAL_MODEL_VERSION
    return {"status": "ok", "version": version}


@app.get("/healthz", include_in_schema=False, response_class=PlainTextResponse)
@app.get("/healthz/", include_in_schema=False, response_class=PlainTextResponse)
def healthz() -> str:
    return "ok"

@app.get("/recommend")
def recommend(user_id: int, k: int = 10, model_version: str | None = None):
    items, resolved_version = _recommend_for_user(user_id=user_id, k=k, model_version=model_version)
    payload = {
        "user_id": int(user_id),
        "items": [int(i) for i in items],
        "model_version": resolved_version or INITIAL_MODEL_VERSION,
        "generated_at": int(time.time()),
    }
    # Enforce Avro schema at the service boundary
    assert_valid(RESPONSE_SCHEMA, payload)
    return payload

@app.get("/recommend/plain", response_class=PlainTextResponse)
def recommend_plain(user_id: int, k: int = 10, model_version: str | None = None) -> str:
    items, resolved_version = _recommend_for_user(user_id=user_id, k=k, model_version=model_version)
    _ = assert_valid(
        RESPONSE_SCHEMA,
        {
            "user_id": int(user_id),
            "items": [int(i) for i in items],
            "model_version": resolved_version or INITIAL_MODEL_VERSION,
            "generated_at": int(time.time()),
        },
    )
    return _as_plain_text(items)

@app.post("/switch")
def switch(model_version: str):
    """Hot-swap the active model version without redeploying the service."""

    try:
        bundle = _REGISTRY.activate(model_version)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive logging path
        raise HTTPException(status_code=500, detail="Failed to switch model") from exc
    return {"status": "ok", "model_version": bundle.version, "metadata": bundle.metadata}

@app.get("/metrics")
def metrics():
    """Prometheus exposition endpoint."""

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.middleware("http")
async def backpressure_guard(request: Request, call_next):
    """Apply bounded concurrency and surface backpressure metrics."""

    try:
        # Non-blocking-ish acquire (tiny timeout) -> reject fast if saturated
        await asyncio.wait_for(_sem.acquire(), timeout=ACQUIRE_TIMEOUT_SEC)
    except TimeoutError:
        REJECTS.inc()
        return JSONResponse({"detail": "Backpressure: server busy"}, status_code=503)

    INFLIGHT.inc()
    try:
        with lat.time():  # reuse Histogram to time total request latency
            resp = await call_next(request)
        reqs.labels(status=str(getattr(resp, "status_code", 200))).inc()
        return resp
    finally:
        INFLIGHT.dec()
        _sem.release()
