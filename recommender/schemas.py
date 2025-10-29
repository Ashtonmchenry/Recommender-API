# recommender/schemas.py
# ---------------------------------------------------------------------
# Avro schema helpers for decoding + validating Kafka events
# Works with Confluent Cloud's Schema Registry and Avro wire format.
# ---------------------------------------------------------------------

import os
import io
import json
import struct
from typing import Any, Dict, Optional, Tuple

from confluent_kafka.schema_registry import SchemaRegistryClient
from fastavro import parse_schema as avro_parse_schema
from fastavro import schemaless_reader
from fastavro.validation import validate as avro_validate


# ---------- Environment & SR client ----------

SR_URL    = os.getenv("SCHEMA_REGISTRY_URL")
SR_KEY    = os.getenv("SCHEMA_REGISTRY_KEY")
SR_SECRET = os.getenv("SCHEMA_REGISTRY_SECRET")

if not all([SR_URL, SR_KEY, SR_SECRET]):
    raise RuntimeError(
        "Missing Schema Registry credentials: "
        "SCHEMA_REGISTRY_URL / SCHEMA_REGISTRY_KEY / SCHEMA_REGISTRY_SECRET"
    )

_sr = SchemaRegistryClient({
    "url": SR_URL,
    "basic.auth.user.info": f"{SR_KEY}:{SR_SECRET}",
})

# ---------- Topic → subject pointers ----------
# If you add more topics, map them here (defaults to {topic}-value if missing)
SCHEMA_POINTERS: Dict[str, str] = {
    "aerosparks.watch":          "aerosparks.watch-value",
    "aerosparks.rate":           "aerosparks.rate-value",
    "aerosparks.reco_requests":  "aerosparks.reco_requests-value",
    "aerosparks.reco_responses": "aerosparks.reco_responses-value",
}

# ---------- Caches ----------
# Cache parsed fastavro schemas (by subject) and by schema id for wire-format decode
_SCHEMA_CACHE_BY_SUBJECT: Dict[str, Tuple[str, Dict[str, Any]]] = {}
_SCHEMA_CACHE_BY_ID: Dict[int, Dict[str, Any]] = {}


# ---------- Helpers ----------

def _subject_for_topic(topic: str) -> str:
    """Return the Schema Registry subject for a given topic's value schema."""
    return SCHEMA_POINTERS.get(topic, f"{topic}-value")


def _parsed_schema_for_subject(subject: str) -> Tuple[str, Dict[str, Any]]:
    """
    Get and cache the latest schema for a subject.
    Returns (schema_type, parsed_schema_dict_for_fastavro).
    """
    if subject in _SCHEMA_CACHE_BY_SUBJECT:
        return _SCHEMA_CACHE_BY_SUBJECT[subject]

    latest = _sr.get_latest_version(subject)  # -> RegisteredSchema
    # latest.schema is a Schema object: schema_str + schema_type (e.g., "AVRO")
    schema_type = getattr(latest.schema, "schema_type", "AVRO")
    schema_str  = latest.schema.schema_str
    schema_dict = json.loads(schema_str)

    parsed = avro_parse_schema(schema_dict)
    _SCHEMA_CACHE_BY_SUBJECT[subject] = (schema_type, parsed)
    return _SCHEMA_CACHE_BY_SUBJECT[subject]


def _parsed_schema_for_id(schema_id: int):
    """
    Return a fastavro parsed schema for a given schema id.
    First try the _sr_schema_by_id() helper (so tests can monkeypatch it),
    then fall back to the real client.
    """
    try:
        meta = _sr_schema_by_id(schema_id)           # <-- test can monkeypatch this
        return avro_parse_schema(json.loads(meta["schema"]))
    except Exception:
        # fallback to direct client call if helper is not patched/available
        if _sr is None:
            raise RuntimeError("Schema Registry client not configured")
        obj = _sr.get_schema(schema_id)              # Schema object
        return avro_parse_schema(json.loads(obj.schema_str))


def _maybe_decode_confluent_avro(raw: bytes) -> Optional[Dict[str, Any]]:
    """
    Detect & decode Confluent wire-format Avro payload.
    Format: [magic=0][schema_id:4 bytes big-endian][avro_payload...]
    Returns dict if Avro-framed, else None.
    """
    if not raw or len(raw) < 5:
        return None
    if raw[0] != 0:
        return None

    schema_id = struct.unpack(">I", raw[1:5])[0]
    parsed_schema = _parsed_schema_for_id(schema_id)

    buf = io.BytesIO(raw[5:])
    record = schemaless_reader(buf, parsed_schema)
    return record


def _schema_for_topic(topic: str) -> Tuple[str, Dict[str, Any]]:
    """
    Get parsed schema for the topic's value subject (latest).
    Returns (schema_type, parsed_schema_dict).
    """
    subject = _subject_for_topic(topic)
    return _parsed_schema_for_subject(subject)


def _normalize(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize common fields for downstream sinks.
    - Ensure a 'timestamp' field exists (prefer explicit timestamp fields).
    """
    if "timestamp" not in d:
        d["timestamp"] = d.get("ts") or d.get("time") or d.get("event_time")
    return d

# --- Compatibility shim so tests can monkeypatch this symbol ---
def _sr_schema_by_id(schema_id: int):
    """
    Return a dict with a 'schema' key (string JSON), same shape tests expect.
    In production we map the Confluent client response to this dict.
    """
    if _sr is None:
        raise RuntimeError("Schema Registry client not configured")
    obj = _sr.get_schema(schema_id)  # has .schema_str
    return {"schema": obj.schema_str}


# ---------- Public API ----------

def parse_event(topic: str, raw: bytes) -> Dict[str, Any]:
    # 1) Try wire format …
    avro_obj = _maybe_decode_confluent_avro(raw)
    if avro_obj is not None:
        res = _schema_for_topic(topic)
        parsed_schema = res[1] if isinstance(res, tuple) else res
        avro_validate(avro_obj, parsed_schema)
        return _normalize(avro_obj)

    # 2) Fallback: bytes are JSON text
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Unable to decode message as Avro or JSON: {e}")

    # **normalize first** so ts/time → timestamp is present for validation
    data = _normalize(data)

    res = _schema_for_topic(topic)
    parsed_schema = res[1] if isinstance(res, tuple) else res
    avro_validate(data, parsed_schema)
    return data


