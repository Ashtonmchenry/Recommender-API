"""Shared ranking and online monitoring metrics for the service."""

from __future__ import annotations

import json
import math
from typing import Sequence, Set

import pandas as pd


# ---------- ranking metrics (per-user) ----------

def precision_at_k(recs: Sequence[int], truth: Set[int], k: int) -> float:
    recs_k = recs[:k]
    if not recs_k:
        return 0.0
    hits = sum(1 for x in recs_k if x in truth)
    return hits / float(min(k, len(recs)))


def recall_at_k(recs: Sequence[int], truth: Set[int], k: int) -> float:
    if not truth:
        return 0.0
    recs_k = recs[:k]
    hits = sum(1 for x in recs_k if x in truth)
    return hits / float(len(truth))


def dcg_at_k(recs: Sequence[int], truth: Set[int], k: int) -> float:
    score = 0.0
    for rank, item in enumerate(recs[:k], start=1):
        if item in truth:
            score += 1.0 / math.log2(rank + 1)
    return score


def ndcg_at_k(recs: Sequence[int], truth: Set[int], k: int) -> float:
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(k, len(truth)) + 1))
    if ideal == 0:
        return 0.0
    return dcg_at_k(recs, truth, k) / ideal


def apk(recs: Sequence[int], truth: Set[int], k: int) -> float:
    """Average Precision@k (for MAP)."""

    score = 0.0
    hits = 0
    for rank, item in enumerate(recs[:k], start=1):
        if item in truth:
            hits += 1
            score += hits / float(rank)
    return score / float(min(k, len(truth))) if truth else 0.0


# ---------- online proxy KPI ----------


def _coerce_timestamp(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    return parsed.view("int64") / 1_000_000_000


def proxy_kpi(responses: pd.DataFrame, watches: pd.DataFrame, horizon_min: int = 10) -> float:
    """% of recommendation requests that lead to a watch event within ``horizon_min`` minutes."""

    if responses.empty or watches.empty:
        return 0.0

    r = responses.copy()
    w = watches.copy()

    # Normalise column casing and provide fallback request ids if missing.
    r.columns = [c.strip().lower() for c in r.columns]
    w.columns = [c.strip().lower() for c in w.columns]
    if "request_id" not in r.columns:
        r.insert(0, "request_id", range(len(r)))

    if "movie_ids" not in r.columns:
        raise ValueError("responses is missing a movie_ids column")

    if r["movie_ids"].dtype == object:
        r["movie_ids"] = r["movie_ids"].apply(
            lambda s: json.loads(s) if isinstance(s, str) else s
        )

    ts_rec_candidates = [c for c in ("timestamp", "ts", "ts_rec") if c in r.columns]
    ts_watch_candidates = [c for c in ("timestamp", "ts", "ts_watch") if c in w.columns]
    if not ts_rec_candidates or not ts_watch_candidates:
        raise ValueError("timestamp columns missing from responses or watches")

    r["ts_rec"] = _coerce_timestamp(r[ts_rec_candidates[0]])
    w["ts_watch"] = _coerce_timestamp(w[ts_watch_candidates[0]])
    w = w.rename(columns={"movie_id": "watch_item"})

    r = r.explode("movie_ids").rename(columns={"movie_ids": "rec_item"})
    merged = r.merge(w, on="user_id", how="inner", suffixes=("_rec", "_watch"))
    merged = merged.dropna(subset=["ts_rec", "ts_watch", "rec_item", "watch_item"])

    window = horizon_min * 60
    dt = merged["ts_watch"] - merged["ts_rec"]
    merged = merged[(dt >= 0) & (dt <= window)]
    if merged.empty:
        return 0.0

    hits = (
        merged.assign(
            hit=(merged["rec_item"].astype(int) == merged["watch_item"].astype(int)).astype(int)
        )
        .groupby("request_id")["hit"]
        .max()
    )
    return float(hits.mean()) if not hits.empty else 0.0
