# quality/avro_registry.py
from __future__ import annotations
import json
from typing import Iterable, Dict, Any, List, Tuple

try:  # pragma: no cover - optional dependency in unit tests
    from confluent_kafka.schema_registry import SchemaRegistryClient
except Exception:  # pragma: no cover - fallback to sentinel for tests
    SchemaRegistryClient = None  # type: ignore

from recommender.config import settings
from fastavro.validation import validate as avro_validate

def _client() -> SchemaRegistryClient:
    if SchemaRegistryClient is None:
        raise RuntimeError("confluent-kafka is required for Schema Registry validation")
    cfg = {"url": settings.schema_registry_url}
    if settings.schema_registry_key and settings.schema_registry_secret:
        cfg["basic.auth.user.info"] = f"{settings.schema_registry_key}:{settings.schema_registry_secret}"
    return SchemaRegistryClient(cfg)

def fetch_latest_schema(subject: str) -> dict:
    """
    Get the latest VALUE schema (as a Python dict) for a subject like:
      aerosparks.requests-value  or  aerosparks.reco_responses-value
    """
    sr = _client()
    latest = sr.get_latest_version(subject)
    # `latest.schema.schema_str` is a JSON string of the Avro schema
    return json.loads(latest.schema.schema_str)

def validate_records(subject: str, records: Iterable[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Validates each record against the latest Avro schema in the registry.
    Returns (ok, errors). If ok is False, at least one record failed.
    """
    schema = fetch_latest_schema(subject)
    errors: List[str] = []
    for i, rec in enumerate(records):
        try:
            avro_validate(rec, schema)   # raises on mismatch
        except Exception as e:
            errors.append(f"idx={i}: {e}")
    return (len(errors) == 0, errors)
