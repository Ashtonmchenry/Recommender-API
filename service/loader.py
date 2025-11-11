"""Data loading helpers for the service."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

_REQUIRED = ("user_id", "item_id", "rating", "timestamp")


def from_csv(path: str) -> pd.DataFrame:
    """Load ratings-like data from CSV and normalize column names."""

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    df = df.rename(
        columns={
            cols.get("userid", "userId"): "user_id",
            cols.get("movieid", "movieId"): "item_id",
            cols.get("rating", "rating"): "rating",
            cols.get("timestamp", "timestamp"): "timestamp",
        }
    )
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df


def from_kafka(batch: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Convert decoded Kafka events to a canonical DataFrame."""

    df = pd.DataFrame(list(batch))
    cols = {c.lower(): c for c in df.columns}
    ren: dict[str, str] = {}
    if "userid" in cols:
        ren[cols["userid"]] = "user_id"
    if "movieid" in cols:
        ren[cols["movieid"]] = "item_id"
    if "rating" in cols:
        ren[cols["rating"]] = "rating"
    if "timestamp" in cols:
        ren[cols["timestamp"]] = "timestamp"
    df = df.rename(columns=ren)

    required = {"user_id", "item_id", "timestamp"}
    if required - set(df.columns):
        raise ValueError("Kafka batch missing required keys for ingestion.")
    return df
