# connects to Kafka + Schema Registry, builds Avro serializers, and sends real messages to the four topics

import datetime
import json
import os
import random
import sys
from typing import Any

from confluent_kafka import SerializingProducer
from confluent_kafka.schema_registry import Schema, SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer

# --- Schema Registry client ---
sr = SchemaRegistryClient(
    {
        "url": os.environ["SCHEMA_REGISTRY_URL"],
        "basic.auth.user.info": f"{os.environ['SCHEMA_REGISTRY_KEY']}:{os.environ['SCHEMA_REGISTRY_SECRET']}",
    }
)

# --- Avro schemas (match SR contracts) ---
SCHEMAS: dict[str, Schema] = {
    # watch (already working)
    "aerosparks.watch": Schema(
        schema_str=json.dumps(
            {
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
        ),
        schema_type="AVRO",
    ),
    # rate (corrected to match existing subject in SR)
    "aerosparks.rate": Schema(
        schema_str=json.dumps(
            {
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
        ),
        schema_type="AVRO",
    ),
    # reco_requests (match SR: event + timestamp string)
    "aerosparks.reco_requests": Schema(
        schema_str=json.dumps(
            {
                "type": "record",
                "name": "RecoRequest",
                "namespace": "aerosparks",
                "fields": [
                    {
                        "name": "event",
                        "type": {"type": "enum", "name": "EventTypeRecoReq", "symbols": ["reco_request"]},
                    },
                    {"name": "user_id", "type": "long"},
                    {"name": "k", "type": "int"},
                    {"name": "timestamp", "type": "string"},
                ],
            }
        ),
        schema_type="AVRO",
    ),
    # reco_responses (match SR: event + timestamp string)
    "aerosparks.reco_responses": Schema(
        schema_str=json.dumps(
            {
                "type": "record",
                "name": "RecoResponse",
                "namespace": "aerosparks",
                "fields": [
                    {
                        "name": "event",
                        "type": {"type": "enum", "name": "EventTypeRecoResp", "symbols": ["reco_response"]},
                    },
                    {"name": "user_id", "type": "long"},
                    {"name": "status", "type": "string"},
                    {"name": "latency_ms", "type": "int"},
                    {"name": "k", "type": "int"},
                    {"name": "movie_ids", "type": {"type": "array", "items": "long"}},
                    {"name": "timestamp", "type": "string"},
                ],
            }
        ),
        schema_type="AVRO",
    ),
}


def make_producer(topic: str) -> SerializingProducer:
    return SerializingProducer(
        {
            "bootstrap.servers": os.environ["KAFKA_BOOTSTRAP"],
            "security.protocol": "SASL_SSL",
            "sasl.mechanisms": "PLAIN",
            "sasl.username": os.environ["KAFKA_API_KEY"],
            "sasl.password": os.environ["KAFKA_API_SECRET"],
            "value.serializer": AvroSerializer(sr, SCHEMAS[topic]),
        }
    )


def now_ms() -> int:
    return int(datetime.datetime.now(datetime.UTC).timestamp() * 1000)


def iso_ts() -> str:
    import datetime

    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sample_watch() -> dict[str, Any]:
    return {
        "event": "watch",
        "user_id": random.randint(1, 1000),
        "movie_id": random.randint(1, 10000),
        "timestamp": iso_ts(),
        "platform": "web",
    }


# corrected: matches SR (event + timestamp string, rating float)
def sample_rate() -> dict[str, Any]:
    return {
        "event": "rate",
        "user_id": random.randint(1, 1000),
        "movie_id": random.randint(1, 10000),
        "rating": float(random.randint(1, 5)),  # float per contract
        "timestamp": iso_ts(),
    }


def sample_reco_req() -> dict[str, Any]:
    return {
        "event": "reco_request",
        "user_id": random.randint(1, 1000),
        "k": 5,
        "timestamp": iso_ts(),
    }


def sample_reco_resp() -> dict[str, Any]:
    k = 5
    base = random.randint(1, 10000)
    return {
        "event": "reco_response",
        "user_id": random.randint(1, 1000),
        "status": "ok",
        "latency_ms": random.randint(50, 200),
        "k": k,
        "movie_ids": [base + i for i in range(k)],
        "timestamp": iso_ts(),
    }


BUILDERS = {
    "aerosparks.watch": sample_watch,
    "aerosparks.rate": sample_rate,
    "aerosparks.reco_requests": sample_reco_req,
    "aerosparks.reco_responses": sample_reco_resp,
}


def produce_one(topic: str):
    p = make_producer(topic)
    val = BUILDERS[topic]()
    p.produce(topic, value=val)
    p.flush()
    print(f"✔ Produced → {topic}: {val}")


def main():
    args = sys.argv[1:]
    targets = list(SCHEMAS.keys()) if not args else [args[0]]
    for t in targets:
        if t not in SCHEMAS:
            raise SystemExit(f"Unknown topic: {t}\nChoose one of: {', '.join(SCHEMAS.keys())}")
        produce_one(t)


if __name__ == "__main__":
    main()
