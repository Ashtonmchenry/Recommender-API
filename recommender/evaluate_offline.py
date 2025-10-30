# Compute HR@K / NDCG@K on chronological split

# recommender/evaluate_offline.py
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

import numpy as np
import pandas as pd

# Reuse your pipeline pieces
from . import transform
from .train import (
    train_baseline,        # trains ALS baseline and returns (model, metrics)
    reindex,               # reindex user_id/item_id -> uidx/iidx
    last_item_holdout,     # chronological split (last interaction per user)
    build_UI,              # build user–item CSR from train set
    seen_by_user,          # precompute "already seen" items per user
    hr_at_k, ndcg_at_k,    # ranking metrics
    K, MIN_INTERACTIONS, EVAL_USERS_CAP, RATING_THRESHOLD
)

def _evaluate_at_k(model, UI, test_df: pd.DataFrame, k: int) -> Dict[str, float]:
    """Compute HR@K and NDCG@K over a test set with columns ['uidx','true_item']."""
    users = test_df["uidx"].to_numpy()
    true_items = test_df["true_item"].to_numpy()
    if len(users) > EVAL_USERS_CAP:
        idx = np.random.RandomState(42).choice(len(users), size=EVAL_USERS_CAP, replace=False)
        users, true_items = users[idx], true_items[idx]

    seen = seen_by_user(UI)
    hits, ndcgs = [], []
    for u, t in zip(users, true_items):
        recs = model.recommend(u, UI, k=k, exclude_seen=True)
        hits.append(hr_at_k(recs, t))
        ndcgs.append(ndcg_at_k(recs, t))
    return {
        "hr@k": float(np.mean(hits)) if hits else 0.0,
        "ndcg@k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "users_evald": int(len(users)),
    }

def _bucket_by_activity(train_df: pd.DataFrame) -> pd.Series:
    """
    Buckets users by their training-set activity count.
    Returns a Series: index=uidx, value=bucket label.
    """
    counts = train_df.groupby("uidx").size()
    bins = [MIN_INTERACTIONS, 10, 20, 50, np.inf]
    labels = [f"[{MIN_INTERACTIONS}-10]", "[11-20]", "[21-50]", "50+"]
    # Align to all users that appear in train_df
    binned = pd.cut(counts, bins=bins, labels=labels, right=True, include_lowest=True)
    return binned

def run_offline_eval(input_csv: str, out_json: str, k: int) -> Dict[str, object]:
    # 1) Load + basic clean (your transform module)
    df = pd.read_csv(input_csv)
    df = transform.basic_clean(df)

    # 2) Chronological split prep
    df = reindex(df, MIN_INTERACTIONS)
    train_df, test_df = last_item_holdout(df)

    # 3) Build UI matrix (implicit positives via threshold)
    n_users = int(df["uidx"].max()) + 1
    n_items = int(df["iidx"].max()) + 1
    UI = build_UI(train_df, n_users, n_items, rating_threshold=RATING_THRESHOLD)

    # 4) Train model via your existing trainer (keeps one source of truth)
    model, top_metrics = train_baseline(df)  # already uses same split & params

    # 5) Re-evaluate explicitly at chosen K (for clarity/repro in this script)
    agg = _evaluate_at_k(model, UI, test_df, k=k)

    # 6) Subpopulation analysis by user activity in TRAIN
    bucket_map = _bucket_by_activity(train_df)
    # Only keep test rows whose user exists in bucket_map
    sub_rows = test_df.merge(bucket_map.rename("bucket"), left_on="uidx", right_index=True, how="left")
    sub_rows = sub_rows.dropna(subset=["bucket"]).copy()
    sub_buckets = {}
    for bucket, g in sub_rows.groupby("bucket"):
        sub_buckets[str(bucket)] = _evaluate_at_k(model, UI, g[["uidx", "true_item"]], k=k)

    # 7) Assemble results
    results = {
        "k": k,
        "threshold": RATING_THRESHOLD,
        "overall": {
            "hr@k": agg["hr@k"],
            "ndcg@k": agg["ndcg@k"],
            "users_evald": agg["users_evald"],
        },
        "subpops": sub_buckets,
        "model_metrics_from_train": top_metrics,  # what train_baseline already computed
        "schema": {
            "input_required_columns": ["user_id", "item_id", "rating", "timestamp"],
            "internal_columns": ["uidx", "iidx"],
        },
    }

    # 8) Persist JSON (good for your PDF deliverable)
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results

def main():
    ap = argparse.ArgumentParser(description="Offline evaluation with chronological split.")
    ap.add_argument("--input", required=True, help="CSV with user_id,item_id,rating,timestamp")
    ap.add_argument("--k", type=int, default=K, help="Top-K for ranking metrics")
    ap.add_argument("--out", default="model_registry/v0.1/offline_results.json", help="Where to write results JSON")
    args = ap.parse_args()

    res = run_offline_eval(args.input, args.out, k=args.k)
    # Pretty print to console as a quick summary
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
