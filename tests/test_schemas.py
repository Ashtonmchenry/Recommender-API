import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import io
import json
import struct

import pytest
from fastavro import parse_schema as avro_parse_schema
from fastavro import schemaless_writer

from recommender import schemas

WATCH_SCHEMA = {
    "type": "record",
    "name": "WatchEvent",
    "namespace": "aerosparks",
    "fields": [
        {"name": "event", "type": {"type": "enum", "name": "EventTypeWatch", "symbols": ["watch"]}},
        {"name": "user_id", "type": "long"},
        {"name": "movie_id", "type": "long"},
        {"name": "timestamp", "type": "string"},
        {"name": "platform", "type": ["null", "string"], "default": None},
    ],
}

RATE_SCHEMA = {
    "type": "record",
    "name": "RateEvent",
    "namespace": "aerosparks",
    "fields": [
        {"name": "event", "type": {"type": "enum", "name": "EventTypeRate", "symbols": ["rate"]}},
        {"name": "user_id", "type": "long"},
        {"name": "movie_id", "type": "long"},
        {"name": "rating", "type": "float"},
        {"name": "timestamp", "type": "string"},
    ],
}


def _wire(payload: dict, schema: dict, schema_id: int = 1234) -> bytes:
    """Build Confluent wire-format bytes: 0x00 + schema_id (big-endian) + Avro binary."""
    parsed = avro_parse_schema(schema)
    buf = io.BytesIO()
    buf.write(b"\x00")
    buf.write(struct.pack(">I", schema_id))
    schemaless_writer(buf, parsed, payload)
    return buf.getvalue()


def test_wire_format_watch_decodes_and_validates(monkeypatch):
    # Arrange
    topic = "aerosparks.watch"
    record = {
        "event": "watch",
        "user_id": 7,
        "movie_id": 99,
        "timestamp": "2025-10-22T01:23:45Z",
        "platform": "web",
    }
    payload = _wire(record, WATCH_SCHEMA, schema_id=444)

    # Monkeypatch SR lookup by ID to return our schema JSON
    def fake_sr_by_id(schema_id: int):
        assert schema_id == 444
        return {"schema": json.dumps(WATCH_SCHEMA)}

    monkeypatch.setattr(schemas, "_sr_schema_by_id", fake_sr_by_id)
    parsed_watch = avro_parse_schema(WATCH_SCHEMA)
    monkeypatch.setattr(schemas, "_schema_for_topic", lambda _topic: ("AVRO", parsed_watch))

    # Act
    out = schemas.parse_event(topic, payload)

    # Assert (normalized structure)
    assert out["event"] == "watch"
    assert out["user_id"] == 7
    assert out["movie_id"] == 99
    assert out["timestamp"] == "2025-10-22T01:23:45Z"
    assert out["platform"] == "web"


def test_json_path_validates_against_topic_schema(monkeypatch):
    # Arrange
    topic = "aerosparks.rate"
    good = {
        "event": "rate",
        "user_id": 1,
        "movie_id": 42,
        "rating": 4.5,
        "timestamp": "2025-10-22T01:23:45Z",
    }
    bad = {  # rating wrong type
        "event": "rate",
        "user_id": 1,
        "movie_id": 42,
        "rating": "bad",
        "timestamp": "2025-10-22T01:23:45Z",
    }
    parsed = avro_parse_schema(RATE_SCHEMA)

    # Monkeypatch topic schema resolver (bypass SR; use our local schema)
    monkeypatch.setattr(schemas, "_schema_for_topic", lambda _topic: parsed)

    # Act/Assert good
    out = schemas.parse_event(topic, json.dumps(good).encode())
    assert out["event"] == "rate"
    assert out["rating"] == 4.5

    # Act/Assert bad (should raise)
    with pytest.raises(Exception):
        schemas.parse_event(topic, json.dumps(bad).encode())


def test_normalization_maps_ts_to_timestamp(monkeypatch):
    topic = "aerosparks.watch"
    parsed = avro_parse_schema(WATCH_SCHEMA)
    monkeypatch.setattr(schemas, "_schema_for_topic", lambda _t: parsed)

    msg = {
        "event": "watch",
        "user_id": 3,
        "movie_id": 8,
        "ts": "2025-10-22T02:00:00Z",
        "platform": None,
    }
    out = schemas.parse_event(topic, json.dumps(msg).encode())
    assert out["timestamp"] == "2025-10-22T02:00:00Z"


from unittest.mock import MagicMock

import quality.avro_registry as avro_registry


def test_avro_validate_ok(monkeypatch):
    mock_client = MagicMock()
    mock_schema = MagicMock()
    mock_schema.schema.schema_str = json.dumps(
        {
            "type": "record",
            "name": "RecoResp",
            "fields": [{"name": "user_id", "type": "int"}],
        }
    )
    mock_client.get_latest_version.return_value = mock_schema
    monkeypatch.setattr(avro_registry, "_client", lambda: mock_client)

    ok, errs = avro_registry.validate_records("team.reco_responses-value", [{"user_id": 7}])
    assert ok
    assert errs == []


def test_avro_validate_bad(monkeypatch):
    mock_client = MagicMock()
    mock_schema = MagicMock()
    mock_schema.schema.schema_str = json.dumps(
        {
            "type": "record",
            "name": "RecoResp",
            "fields": [{"name": "user_id", "type": "int"}],
        }
    )
    mock_client.get_latest_version.return_value = mock_schema
    monkeypatch.setattr(avro_registry, "_client", lambda: mock_client)

    ok, errs = avro_registry.validate_records("team.reco_responses-value", [{"user_id": "oops"}])
    assert not ok
    assert len(errs) == 1
