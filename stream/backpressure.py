from __future__ import annotations

from confluent_kafka import Consumer, TopicPartition


def with_backpressure(consumer: Consumer, handle, max_inflight: int = 100):
    """Poll loop with pause/resume based on inflight size reported by handler.

    The handler must return current inflight count after processing a message.
    """
    paused = False
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print(f"Kafka error: {msg.error()}")
            continue
        # user callback returns inflight count
        inflight = handle(msg)
        if inflight > max_inflight and not paused:
            consumer.pause([TopicPartition(msg.topic(), msg.partition())])
            paused = True
            print(f"Paused partitions due to backpressure: {msg.topic()}[{msg.partition()}]")
        elif paused and inflight <= max_inflight // 2:
            consumer.resume([TopicPartition(msg.topic(), msg.partition())])
            paused = False
            print("Resumed partitions after drain")
