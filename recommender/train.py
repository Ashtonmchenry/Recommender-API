# Minimal trainer stub: replace with ALS / CF / baseline training

# recommender/train.py
from __future__ import annotations
import argparse, math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from implicit.als import AlternatingLeastSquares

from .pipeline import transform
from . import serialize  # keep your existing helpers

# ---- hyperparams (can be CLI-overridden) ----
K = 20
MIN_INTERACTIONS = 5
EVAL_USERS_CAP = 5000
RATING_THRESHOLD = 3.5        # binarize: rating > threshold = positive
ALS_FACTORS = 64
ALS_REG = 0.01
ALS_ITERS = 20

# -------- utils (chronological split + indexing) --------
def _normalize_time(df: pd.DataFrame) -> pd.DataFrame:
    """Accept 'ts' or 'timestamp'; always return 'timestamp'."""
    if "timestamp" not in df.columns and "ts" in df.columns:
        df = df.rename(columns={"ts": "timestamp"})
    return df

def reindex(df: pd.DataFrame, min_interactions: int = MIN_INTERACTIONS) -> pd.DataFrame:
    """Keep users with enough interactions and add 0..N-1 indices."""
    counts = df.groupby("user_id").size()
    keep = set(counts[counts >= min_interactions].index)
    df = df[df["user_id"].isin(keep)].copy()
    uid_map = {u: i for i, u in enumerate(sorted(df["user_id"].unique()))}
    iid_map = {m: i for i, m in enumerate(sorted(df["item_id"].unique()))}
    df["uidx"] = df["user_id"].map(uid_map)
    df["iidx"] = df["item_id"].map(iid_map)
    return df

def last_item_holdout(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological last-item holdout per user.
    Requires columns: uidx, iidx (or item_id), timestamp.
    """
    df = _normalize_time(df)
    item_col = "iidx" if "iidx" in df.columns else "item_id"

    # Make sorting independent of any existing index
    tmp = df.reset_index(drop=True).sort_values(["uidx", "timestamp"])
    last_idx = tmp.groupby("uidx")["timestamp"].idxmax()

    test = (
        tmp.loc[last_idx, ["uidx", item_col]]
        .rename(columns={item_col: "true_item"})
        .reset_index(drop=True)
    )
    train = tmp.drop(index=last_idx).reset_index(drop=True)
    return train, test

def build_UI(train: pd.DataFrame, n_users: int, n_items: int, rating_threshold: float) -> sparse.csr_matrix:
    rows = train["uidx"].to_numpy()
    cols = train["iidx"].to_numpy()
    # implicit positives: 1 if rating > threshold else 0
    vals = (train["rating"].to_numpy() > rating_threshold).astype(float) if "rating" in train.columns else np.ones_like(rows, dtype=float)
    return sparse.coo_matrix((vals, (rows, cols)), shape=(n_users, n_items)).tocsr()

def seen_by_user(UI: sparse.csr_matrix) -> List[set]:
    return [set(UI[u].indices) for u in range(UI.shape[0])]

def hr_at_k(recs: Iterable[int], t: int) -> float:
    return 1.0 if t in recs else 0.0

def ndcg_at_k(recs: Iterable[int], t: int) -> float:
    for r, i in enumerate(recs, start=1):
        if i == t:
            return 1.0 / math.log2(r + 1)
    return 0.0

# -------- ALS model wrapper --------
@dataclass
class ALSModel:
    als: AlternatingLeastSquares
    user_map: dict[int, int]
    item_map: dict[int, int]
    user_factors: np.ndarray
    item_factors: np.ndarray

    def recommend(self, uidx: int, UI: sparse.csr_matrix, k: int, exclude_seen: bool = True) -> list[int]:
        # pass the 1×N row, not the full matrix
        user_row = UI[uidx]
        recs, _ = self.als.recommend(
            userid=uidx,
            user_items=user_row,
            N=k,
            filter_already_liked_items=exclude_seen,
        )
        return recs.tolist()

# -------- public API --------
def train_baseline(df: pd.DataFrame) -> Tuple[Any, Dict[str, float]]:
    """
    Train ALS on implicit positives with chronological evaluation.
    df columns required: user_id, item_id, rating, timestamp (or ts)
    Returns: (ALSModel, metrics)
    """
    df = _normalize_time(df)
    df = reindex(df, MIN_INTERACTIONS)
    train_df, test_df = last_item_holdout(df)

    n_users = int(df["uidx"].max()) + 1
    n_items = int(df["iidx"].max()) + 1
    UI = build_UI(train_df, n_users, n_items, rating_threshold=RATING_THRESHOLD)

    # ALS expects item-user matrix for fitting
    # IU = UI.T.tocsr()
    als = AlternatingLeastSquares(factors=ALS_FACTORS, regularization=ALS_REG, iterations=ALS_ITERS)
    als.fit(UI)

    model = ALSModel(
        als=als,
        user_map=dict(zip(df["user_id"], df["uidx"])),
        item_map=dict(zip(df["item_id"], df["iidx"])),
        user_factors=als.user_factors.copy(),
        item_factors=als.item_factors.copy(),
    )

    # Evaluate HR@20 / NDCG@20 on last-item holdout
    users = test_df["uidx"].to_numpy()
    true_items = test_df["true_item"].to_numpy()
    if len(users) > EVAL_USERS_CAP:
        idx = np.random.RandomState(42).choice(len(users), size=EVAL_USERS_CAP, replace=False)
        users, true_items = users[idx], true_items[idx]

    seen = seen_by_user(UI)
    hits, ndcgs = [], []
    for u, t in zip(users, true_items):
        recs = model.recommend(u, UI, k=K, exclude_seen=True)
        hits.append(hr_at_k(recs, t))
        ndcgs.append(ndcg_at_k(recs, t))

    metrics = {
        "hr@20": float(np.mean(hits)) if hits else 0.0,
        "ndcg@20": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "users_evald": int(len(users)),
        "params": {"factors": ALS_FACTORS, "reg": ALS_REG, "iters": ALS_ITERS, "thr": RATING_THRESHOLD},
    }
    return model, metrics

def main():
    global ALS_FACTORS, ALS_REG, ALS_ITERS, RATING_THRESHOLD

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with user_id,item_id,rating,timestamp (or ts)")
    ap.add_argument("--output", default="models/reco.joblib", help="Path to save model")
    ap.add_argument("--factors", type=int, default=ALS_FACTORS)
    ap.add_argument("--reg", type=float, default=ALS_REG)
    ap.add_argument("--iters", type=int, default=ALS_ITERS)
    ap.add_argument("--thr", type=float, default=RATING_THRESHOLD)
    args = ap.parse_args()

    ALS_FACTORS, ALS_REG, ALS_ITERS, RATING_THRESHOLD = args.factors, args.reg, args.iters, args.thr

    df = pd.read_csv(args.input)
    # keep your transform, but make it tolerant to ts/timestamp
    df = _normalize_time(df)
    df = transform.basic_clean(df)

    model, metrics = train_baseline(df)
    serialize.save_model(model, args.output)
    print("Training complete.\nMetrics:", metrics, "\nSaved:", args.output)

if __name__ == "__main__":
    main()

