# Placeholder for custom metrics; using prometheus_client in app.py

# metrics.py
from __future__ import annotations
from typing import Iterable, Sequence, Set, Dict
import math
import numpy as np
import pandas as pd

# ---------- ranking metrics (per-user) ----------
def precision_at_k(recs: Sequence[int], truth: Set[int], k: int) -> float:
    recs_k = recs[:k]
    if not recs_k: return 0.0
    hits = sum(1 for x in recs_k if x in truth)
    return hits / float(min(k, len(recs)))

def recall_at_k(recs: Sequence[int], truth: Set[int], k: int) -> float:
    if not truth: return 0.0
    recs_k = recs[:k]
    hits = sum(1 for x in recs_k if x in truth)
    return hits / float(len(truth))

def dcg_at_k(recs: Sequence[int], truth: Set[int], k: int) -> float:
    s = 0.0
    for r, item in enumerate(recs[:k], start=1):
        if item in truth:
            s += 1.0 / math.log2(r + 1)
    return s

def ndcg_at_k(recs: Sequence[int], truth: Set[int], k: int) -> float:
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(k, len(truth)) + 1))
    if ideal == 0: return 0.0
    return dcg_at_k(recs, truth, k) / ideal

def apk(recs: Sequence[int], truth: Set[int], k: int) -> float:
    """Average Precision@k (for MAP)."""
    score = 0.0; hits = 0
    for r, item in enumerate(recs[:k], start=1):
        if item in truth:
            hits += 1
            score += hits / float(r)
    return score / float(min(k, len(truth))) if truth else 0.0

# ---------- online proxy KPI ----------
def proxy_kpi(responses: pd.DataFrame, watches: pd.DataFrame, horizon_min: int = 10) -> float:
    """
    % of response requests where the user watched any recommended item within horizon_min minutes.
    responses: columns [request_id, user_id, movie_ids, ts] where movie_ids is list[int] or JSON string.
    watches:   columns [user_id, movie_id, ts]
    """
    r = responses.copy()
    w = watches.copy()
    # movie_ids may be list or JSON-encoded string
    if r["movie_ids"].dtype == object and not isinstance(r["movie_ids"].iloc[0], list):
        import json
        r["movie_ids"] = r["movie_ids"].apply(lambda s: json.loads(s) if isinstance(s, str) else [])
    r = r.explode("movie_ids").rename(columns={"movie_ids": "rec_item"})
    # time window join (same user within horizon)
    merged = r.merge(w, on="user_id", suffixes=("_rec", "_watch"))
    dt = merged["ts_watch"] - merged["ts_rec"]
    merged = merged[(dt >= 0) & (dt <= horizon_min * 60)]
    # hit per request_id if any rec_item equals watched movie_id
    hits = merged.assign(hit=(merged["rec_item"] == merged["movie_id"]).astype(int)) \
                 .groupby("request_id")["hit"].max()
    return float(hits.mean()) if len(hits) else 0.0
