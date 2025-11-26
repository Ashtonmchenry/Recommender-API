import os
import json
import random
import time
from confluent_kafka import Producer

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP")
API_KEY = os.environ.get("KAFKA_API_KEY")
API_SECRET = os.environ.get("KAFKA_API_SECRET")
WATCH_TOPIC = os.environ.get("KAFKA_WATCH_TOPIC")

print("BOOTSTRAP =", BOOTSTRAP)
print("WATCH_TOPIC =", WATCH_TOPIC)

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
}

def delivery_report(err, msg):
    if err is not None:
        print(f"Delivery failed for record {msg.key()}: {err}")
    else:
        # show only a few to avoid spam
        print(f"Produced record to {msg.topic()} [partition {msg.partition()}] @ offset {msg.offset()}")

print("Creating producer...")
producer = Producer(conf)
print("Producer created.")

popular_movies = list(range(1, 6))   # 5 popular items
tail_movies    = list(range(6, 51))  # 45 tail items

num_messages = 100

print(f"Producing {num_messages} synthetic watch events to topic {WATCH_TOPIC}...")

for i in range(num_messages):
    if random.random() < 0.7:
        movie_id = random.choice(popular_movies)
    else:
        movie_id = random.choice(tail_movies)

    event = {
        "user_id": random.randint(1, 30),
        "movie_id": movie_id,
        "ts": int(time.time() * 1000),
    }

    try:
        producer.produce(
            WATCH_TOPIC,
            json.dumps(event).encode("utf-8"),
            callback=delivery_report if i < 5 else None,  # first few log deliveries
        )
    except BufferError as e:
        print("Local buffer full, flushing:", e)
        producer.flush()
        producer.produce(WATCH_TOPIC, json.dumps(event).encode("utf-8"))

    if i % 20 == 0:
        producer.poll(0)  # serve delivery callbacks

print("Flushing producer…")
producer.flush()
print("Done producing.")
