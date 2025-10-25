# scripts/probe.py
import os, time, random, requests, json
from datetime import datetime
from confluent_kafka import SerializingProducer
from confluent_kafka.serialization import StringSerializer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer

TEAM = os.getenv("TEAM", "aerosparks")  # topic prefix
REQ_TOPIC = f"{TEAM}.reco_requests"
RESP_TOPIC = f"{TEAM}.reco_responses"

API = os.environ.get("RECO_API", "http://localhost:8080")

def utc_ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def sr_client() -> SchemaRegistryClient:
    url = os.environ["SCHEMA_REGISTRY_URL"]
    key = os.environ["SCHEMA_REGISTRY_KEY"]
    sec = os.environ["SCHEMA_REGISTRY_SECRET"]
    return SchemaRegistryClient({
        "url": url,
        "basic.auth.user.info": f"{key}:{sec}",
    })

def latest_schema(sr: SchemaRegistryClient, subject: str):
    """Return the latest Schema object for a subject."""
    return sr.get_latest_version(subject).schema

def make_producer(serializer) -> SerializingProducer:
    return SerializingProducer({
        "bootstrap.servers": os.environ["KAFKA_BOOTSTRAP"],
        "security.protocol": "SASL_SSL",
        "sasl.mechanisms": "PLAIN",
        "sasl.username": os.environ["KAFKA_API_KEY"],
        "sasl.password": os.environ["KAFKA_API_SECRET"],
        "key.serializer": StringSerializer("utf_8"),
        "value.serializer": serializer,
    })

def probe_once():
    sr = sr_client()

    # Build serializers from the **actual** latest SR subjects
    req_schema = latest_schema(sr, f"{REQ_TOPIC}-value")
    resp_schema = latest_schema(sr, f"{RESP_TOPIC}-value")

    req_ser = AvroSerializer(sr, req_schema)
    resp_ser = AvroSerializer(sr, resp_schema)

    prod_req = make_producer(req_ser)
    prod_resp = make_producer(resp_ser)

    # Pick a random user, send request event
    user = random.randint(1, 1000)
    req_event = {
        "event": "reco_request",
        "user_id": user,
        "k": 5,
        "timestamp": utc_ts(),
    }
    prod_req.produce(topic=REQ_TOPIC, key=str(user), value=req_event)
    prod_req.flush()
    print("→ sent", req_event)

    # Hit your API to get some ids (or fake them if call fails)
    try:
        r = requests.get(f"{API}/recommend/{user}?k=5", timeout=5)
        r.raise_for_status()
        # Your API returns CSV like "50,172,1"
        movie_ids = [int(x) for x in r.text.split(",") if x.strip()]
    except Exception as e:
        print("API call failed, using fallback ids:", e)
        movie_ids = list(range(100, 105))

    # Build and send a response event (make sure ids are ints → Avro long)
    resp_event = {
        "event": "reco_response",
        "user_id": user,
        "status": "ok",
        "latency_ms": random.randint(50, 150),
        "k": len(movie_ids),
        "movie_ids": movie_ids,
        "timestamp": utc_ts(),
    }
    prod_resp.produce(topic=RESP_TOPIC, key=str(user), value=resp_event)
    prod_resp.flush()
    print("→ sent", resp_event)

def main():
    # one-shot; change to a loop if you want it to run continuously
    probe_once()

if __name__ == "__main__":
    main()
