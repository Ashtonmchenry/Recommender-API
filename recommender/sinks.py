# recommender/sinks.py
import datetime as dt
import os
from typing import Any

import pandas as pd

# Optional S3 upload
_S3_ENABLED = bool(os.getenv("S3_BUCKET"))
if _S3_ENABLED:
    import boto3

    _s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),  # optional (e.g., MinIO)
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )

OUT_DIR = os.getenv("OUT_DIR", "data")
FORMATS = [f.strip().lower() for f in os.getenv("OUTPUT_FORMATS", "parquet,csv").split(",") if f.strip()]


def _ts():
    return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _ensure_local_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _write_local(df: pd.DataFrame, topic: str, fmt: str, batch_tag: str) -> str | None:
    date_part = dt.datetime.utcnow().strftime("%Y-%m-%d")
    base = os.path.join(OUT_DIR, topic, f"date={date_part}")
    _ensure_local_dir(base + os.sep)
    if fmt == "parquet":
        path = os.path.join(base, f"{batch_tag}.parquet")
        df.to_parquet(path, index=False)
        return path
    if fmt == "csv":
        path = os.path.join(base, f"{batch_tag}.csv")
        df.to_csv(path, index=False)
        return path
    return None


def _upload_s3(local_path: str, topic: str):
    if not _S3_ENABLED:
        return
    bucket = os.environ["S3_BUCKET"]
    prefix = os.getenv("S3_PREFIX", "").strip("/")
    rel = os.path.relpath(local_path, OUT_DIR).replace("\\", "/")
    key = f"{prefix}/{rel}" if prefix else rel
    _s3.upload_file(local_path, bucket, key)


def write_batch(records: list[dict[str, Any]], topic: str) -> dict[str, str]:
    """
    Write a batch to all configured formats. Returns map {fmt: local_path}.
    """
    if not records:
        return {}
    df = pd.DataFrame(records)
    tag = _ts()
    outputs = {}
    for fmt in FORMATS:
        p = _write_local(df, topic, fmt, tag)
        if p:
            outputs[fmt] = p
            _upload_s3(p, topic)
    return outputs
