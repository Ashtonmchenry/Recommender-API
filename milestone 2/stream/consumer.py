# connects to Kafka, deserializes & validates Avro (via schemas.py), batches records, and writes Parquet/CSV (and optionally S3). Also manages offsets/commits.

import os
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd
from confluent_kafka import Consumer, KafkaException
from recommender.schemas import parse_event  # our decoder+validator+normalizer


# ---- knobs (env overridable) -------------------------------------------------
BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP")
API_KEY   = os.environ.get("KAFKA_API_KEY")
API_SEC   = os.environ.get("KAFKA_API_SECRET")

GROUP_ID  = os.environ.get("KAFKA_GROUP", "ingestor")
TOPICS    = (os.environ.get("TOPICS") or
             f'{os.environ.get("WATCH_TOPIC","aerosparks.watch")},'
             f'{os.environ.get("RATE_TOPIC","aerosparks.rate")},'
             f'{os.environ.get("REQ_TOPIC","aerosparks.reco_requests")},'
             f'{os.environ.get("RESP_TOPIC","aerosparks.reco_responses")}').split(",")

AUTO_RESET   = os.environ.get("AUTO_OFFSET_RESET", "earliest")

# snapshot settings
SNAP_DIR     = Path(os.environ.get("SNAP_DIR", "snapshots"))
SNAP_FORMATS = set(os.environ.get("SNAP_FORMATS", "parquet,csv").split(","))  # any of: parquet,csv
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", "500"))    # flush when N records per-topic
BATCH_SECS   = float(os.environ.get("BATCH_SECS", "15"))   # or after N seconds since last flush

# -----------------------------------------------------------------------------

def _consumer_conf():
    if not (BOOTSTRAP and API_KEY and API_SEC):
        raise RuntimeError("Missing Kafka env (KAFKA_BOOTSTRAP/API_KEY/API_SECRET)")
    return {
        "bootstrap.servers": BOOTSTRAP,
        "security.protocol": "SASL_SSL",
        "sasl.mechanisms": "PLAIN",
        "sasl.username": API_KEY,
        "sasl.password": API_SEC,
        "group.id": GROUP_ID,
        "auto.offset.reset": AUTO_RESET,
        # commit explicitly after successful parse+buffer
        "enable.auto.commit": False,
    }


def _flush_topic(topic: str, buf: list[dict], file_seq: dict[str, int]) -> int:
    """Write one topic buffer to Parquet/CSV, return rows written."""
    if not buf:
        return 0

    df = pd.DataFrame(buf)

    # output path: snapshots/<topic>/dt=YYYY-MM-DD/batch-<ts>-<seq>.{parquet,csv}
    now = datetime.now(timezone.utc)
    date_dir = SNAP_DIR / topic / f"dt={now.strftime('%Y-%m-%d')}"
    date_dir.mkdir(parents=True, exist_ok=True)

    seq = file_seq[topic]
    file_seq[topic] += 1
    stem = date_dir / f"batch-{now.strftime('%Y%m%dT%H%M%S')}-{seq:05d}"

    if "parquet" in SNAP_FORMATS:
        df.to_parquet(f"{stem}.parquet", index=False)
    if "csv" in SNAP_FORMATS:
        df.to_csv(f"{stem}.csv", index=False)

    wrote = len(df)
    buf.clear()
    print(f"wrote {wrote} rows → {stem}.({'/'.join(sorted(SNAP_FORMATS))})")
    return wrote


def _flush_all(buffers: dict[str, list], file_seq: dict[str, int]) -> int:
    total = 0
    for t, b in buffers.items():
        total += _flush_topic(t, b, file_seq)
    return total


def main():
    c = Consumer(_consumer_conf())
    c.subscribe([t for t in TOPICS if t])

    print(f"Subscribed to: {TOPICS}   group={GROUP_ID}")

    buffers: dict[str, list[dict]] = defaultdict(list)
    last_flush = time.time()
    file_seq: dict[str, int] = defaultdict(int)

    try:
        while True:
            msg = c.poll(1.0)
            now = time.time()

            # time-based flush
            if (now - last_flush) >= BATCH_SECS:
                _flush_all(buffers, file_seq)
                last_flush = now

            if msg is None:
                continue
            if msg.error():
                # network/partition EOF etc—just log & continue
                print(f"[{msg.topic()}] error: {msg.error()}")
                continue

            try:
                # Parse + validate + normalize
                event = parse_event(msg.topic(), msg.value())

                # enrich with Kafka metadata for lineage/debug
                ts_type, ts_val = msg.timestamp()
                event["_kafka_topic"] = msg.topic()
                event["_kafka_partition"] = msg.partition()
                event["_kafka_offset"] = msg.offset()
                event["_kafka_ts"] = (
                    datetime.fromtimestamp(ts_val/1000, tz=timezone.utc).isoformat()
                    if ts_val and ts_val > 0 else None
                )

                # buffer
                buffers[msg.topic()].append(event)

                # size-based flush
                if len(buffers[msg.topic()]) >= BATCH_SIZE:
                    _flush_topic(msg.topic(), buffers[msg.topic()], file_seq)
                    last_flush = now

                # commit only after successful parse/buffer
                c.commit(msg, asynchronous=True)

            except Exception as e:
                # don’t commit bad messages; just log
                print(f"[{msg.topic()}] decode/validate failed at p{msg.partition()}@o{msg.offset()}: {e}")

    except KeyboardInterrupt:
        print("Stopping…")

    except KafkaException as ke:
        print(f"Kafka error: {ke}")

    finally:
        # final flush
        wrote = _flush_all(buffers, file_seq)
        print(f"final flush wrote {wrote} rows")
        c.close()


if __name__ == "__main__":
    main()
