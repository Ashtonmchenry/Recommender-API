# loader.py
from __future__ import annotations
from typing import Iterable, Dict, Any
import pandas as pd

_REQUIRED = ("user_id", "item_id", "rating", "timestamp")

def from_csv(path: str) -> pd.DataFrame:
    """
    Load ratings-like data from CSV and normalize column names.
    Required logical columns: user_id, item_id, rating, timestamp.
    """
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

def from_kafka(batch: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a batch (list/iterable of dict) of decoded Kafka events to a DataFrame
    with canonical columns; extra keys are kept.
    """
    df = pd.DataFrame(list(batch))
    # tolerant rename if producers used camelCase
    cols = {c.lower(): c for c in df.columns}
    ren = {}
    if "userid" in cols: ren[cols["userid"]] = "user_id"
    if "movieid" in cols: ren[cols["movieid"]] = "item_id"
    if "rating" in cols: ren[cols["rating"]] = "rating"
    if "timestamp" in cols: ren[cols["timestamp"]] = "timestamp"
    df = df.rename(columns=ren)
    # only validate if those fields are present (watch events may not have rating)
    if {"user_id", "item_id", "timestamp"} - set(df.columns):
        raise ValueError("Kafka batch missing required keys for ingestion.")
    return df
