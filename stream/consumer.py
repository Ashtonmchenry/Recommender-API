
# Full Batch Processor

from __future__ import annotations
from confluent_kafka import Consumer, KafkaError
from collections import deque
from typing import Deque
from recommender.config import settings

def main():
    c = Consumer({
        'bootstrap.servers': settings.kafka_bootstrap,
        'security.protocol': settings.kafka_security_protocol,
        'sasl.mechanism': settings.kafka_sasl_mechanism,
        'sasl.username': settings.kafka_api_key,
        'sasl.password': settings.kafka_api_secret,
        'group.id': settings.group_id,
        'enable.auto.commit': False,
        'auto.offset.reset': 'earliest',
        'queued.max.messages.kbytes': 20480  # 20MB
    })
    c.subscribe([settings.reco_requests_topic])
    inflight: Deque = deque(maxlen=1000)

    try:
        while True:
            msg = c.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                print("Kafka error", msg.error()); continue
            inflight.append(msg)
            # simulate work then commit
            # ... do work here ...
            c.commit(msg, asynchronous=True)
            # backpressure: pause if deque nearly full
            if len(inflight) > 800:
                c.pause(c.assignment())
            elif len(inflight) < 200:
                c.resume(c.assignment())
    finally:
        c.close()

if __name__ == "__main__":
    main()
