import os
import json
from confluent_kafka import Consumer, KafkaError

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP")
API_KEY = os.environ.get("KAFKA_API_KEY")
API_SECRET = os.environ.get("KAFKA_API_SECRET")
WATCH_TOPIC = os.environ.get("KAFKA_WATCH_TOPIC", "")

if not (BOOTSTRAP and API_KEY and API_SECRET and WATCH_TOPIC):
    raise SystemExit(
        "Missing one of KAFKA_BOOTSTRAP / KAFKA_API_KEY / "
        "KAFKA_API_SECRET / KAFKA_WATCH_TOPIC environment variables."
    )

conf = {
    "bootstrap.servers": BOOTSTRAP,
    "security.protocol": "SASL_SSL",
    "sasl.mechanisms": "PLAIN",
    "sasl.username": API_KEY,
    "sasl.password": API_SECRET,
    "group.id": "loop-analysis-consumer",
    "auto.offset.reset": "earliest",  # start from latest; change to 'earliest' if you want older data
}

consumer = Consumer(conf)
consumer.subscribe([WATCH_TOPIC])

out_path = "scripts/exports/kafka_watch_sample.jsonl"
max_messages = 2000

print(f"Consuming up to {max_messages} messages from topic {WATCH_TOPIC}...")
count = 0

with open(out_path, "w", encoding="utf-8") as f:
    while count < max_messages:
        msg = consumer.poll(2.0)
        if msg is None:
            print("No more messages (timeout). Stopping.")
            break
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                print("Reached end of partition.")
                break
            print("Kafka error:", msg.error())
            continue

        try:
            value = msg.value().decode("utf-8")
        except Exception as e:
            print("Unable to decode message value:", e)
            continue

        # If your producer already sent JSON strings, this will be valid JSON.
        # If not, we just store the raw string.
        try:
            obj = json.loads(value)
        except json.JSONDecodeError:
            obj = {"raw": value}

        f.write(json.dumps(obj) + "\n")
        count += 1

print(f"Wrote {count} messages to {out_path}")
consumer.close()
