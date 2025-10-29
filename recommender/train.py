# Minimal trainer stub: replace with ALS / CF / baseline training

"""
Trainer that uses your custom model in model/s4.py.

- Exposes train_baseline(df) so the rest of the code can call it.
- Provides a CLI:
    python -m recommender.train --input data/ratings.csv --output models/reco.joblib
"""
"""
Production trainer (no Colab deps)
Exports:
  - train_baseline(df) -> (model, metrics)
  - CLI: python -m recommender.train --input data/ratings.csv --output models/reco.joblib
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Any
from pathlib import Path
import argparse
import math

import numpy as np
import pandas as pd
from scipy import sparse

from pipeline import ingest, transform, serialize

# ---- hyperparams ----
K = 20
MIN_INTERACTIONS = 5
EVAL_USERS_CAP = 5000

# -------- utils (mirrors the logic in s4) --------
def reindex(df: pd.DataFrame, min_interactions: int = MIN_INTERACTIONS) -> pd.DataFrame:
    """Re-map original user and item IDs to zero-based integer indices (uidx, iidx) for efficient sparse matrix use."""

    counts = df.groupby("user_id").size()
    keep = set(counts[counts >= min_interactions].index) # Filter out users with fewer than min_interactions ratings.
    df = df[df["user_id"].isin(keep)].copy()

    # Build lookup maps (uid_map, iid_map).
    uid_map = {u: i for i, u in enumerate(sorted(df["user_id"].unique()))} # users
    iid_map = {m: i for i, m in enumerate(sorted(df["item_id"].unique()))} # items

    # Add two new integer columns to the DataFrame
    df["uidx"] = df["user_id"].map(uid_map)
    df["iidx"] = df["item_id"].map(iid_map)
    return df

def last_item_holdout(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a train/test split for evaluation based on time."""

    df = df.sort_values(["uidx", "timestamp"]) # most recent (latest timestamp) interaction, used as the test set
    test_idx = df.groupby("uidx")["timestamp"].idxmax()
    test = df.loc[test_idx, ["uidx", "iidx"]].rename(columns={"iidx": "true_item"})
    train = df.drop(index=test_idx, axis=0)
    return train, test

def build_UI(train: pd.DataFrame, n_users: int, n_items: int, use_ratings: bool = True) -> sparse.csr_matrix:
    """Build the user–item interaction matrix (sparse CSR format).
    Rows = users, columns = items, values = ratings"""

    rows = train["uidx"].to_numpy()
    cols = train["iidx"].to_numpy()
    vals = train["rating"].to_numpy() if use_ratings else np.ones_like(rows, float)
    return sparse.coo_matrix((vals, (rows, cols)), shape=(n_users, n_items)).tocsr()

def seen_by_user(UI: sparse.csr_matrix) -> List[set]:
    """Precompute which items each user has already interacted with.
    Returns a list of Python sets where seen[u] = set of item indices the user has rated.
    Exclude already-watched items when recommending"""
    return [set(UI[u].indices) for u in range(UI.shape[0])]

def hr_at_k(recs: Iterable[int], t: int) -> float:
    """Compute Hit Rate @ K for one user. Returns 1.0 if the true item t is in the top-K recommendations, else 0.0"""
    return 1.0 if t in recs else 0.0

def ndcg_at_k(recs: Iterable[int], t: int) -> float:
    """Compute Normalized Discounted Cumulative Gain @ K for one user. """
    for r, i in enumerate(recs, start=1):
        if i == t:
            return 1.0 / math.log2(r + 1)
    return 0.0

# -------- simple Item–Item CF model (cosine) --------
@dataclass
class ItemCFModel:
    """Minimal Item–Item Collaborative Filtering model class"""
    item_factors: sparse.csr_matrix  # normalized UI^T
    ui: sparse.csr_matrix            # user–item CSR
    item_norms: np.ndarray

    def recommend(self, user_idx: int, k: int = K, exclude: set | None = None) -> List[int]:
        exclude = exclude or set()
        scores = (self.ui[user_idx] @ self.item_factors).toarray().ravel()
        if exclude:
            scores[list(exclude)] = -np.inf
        top = np.argpartition(scores, -k)[-k:]
        return list(top[np.argsort(-scores[top])])

def _compute_item_cosine(UI: sparse.csr_matrix) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Compute item–item similarity (cosine) matrix"""
    # column (item) L2 norms
    norms = np.sqrt(UI.power(2).sum(axis=0)).A1 + 1e-9
    inv = sparse.diags(1.0 / norms)  # (n_items x n_items)
    # item_factors = (UI @ inv)^T  -> (n_items x n_users); store transposed for scoring
    # but we want (UI^T * D^-1)^T == D^-1 * UI^T so that (UI @ (D^-1 * UI^T)) gives cosine-ish scores
    item_factors = (inv @ UI.T).tocsr()  # (n_items x n_users)
    return item_factors, norms

def train_baseline(df: pd.DataFrame) -> Tuple[Any, Dict[str, float]]:
    """
    Train item–item CF using last-item holdout. Returns (model, metrics).
    Input df must have: user_id, item_id, rating, timestamp
    """
    # reindex & split
    df = reindex(df, MIN_INTERACTIONS)
    train_df, test_df = last_item_holdout(df)

    n_users = int(df["uidx"].max()) + 1
    n_items = int(df["iidx"].max()) + 1

    UI = build_UI(train_df, n_users, n_items, use_ratings=True)
    item_factors, norms = _compute_item_cosine(UI)
    model = ItemCFModel(item_factors=item_factors, ui=UI, item_norms=norms)

    # evaluate (HR@20 / NDCG@20)
    users = test_df["uidx"].to_numpy()
    true_items = test_df["true_item"].to_numpy()
    if len(users) > EVAL_USERS_CAP:
        idx = np.random.RandomState(42).choice(len(users), size=EVAL_USERS_CAP, replace=False)
        users, true_items = users[idx], true_items[idx]

    seen = seen_by_user(UI)
    hits, ndcgs = [], []
    for u, t in zip(users, true_items):
        recs = model.recommend(u, k=K, exclude=seen[u])
        hits.append(hr_at_k(recs, t))
        ndcgs.append(ndcg_at_k(recs, t))

    metrics = {
        "hr@20": float(np.mean(hits)) if hits else 0.0,
        "ndcg@20": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "users_evald": int(len(users)),
        "items": int(n_items),
        "users": int(n_users),
    }
    return model, metrics

# -------- CLI --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with user_id,item_id,rating,timestamp")
    ap.add_argument("--output", default="models/reco.joblib", help="Path to save model")
    args = ap.parse_args()

    df = ingest.from_csv("data/ratings.csv")
    df = transform.basic_clean(df)  # transform.py
    model, metrics = train_baseline(df)
    serialize.save_model(model, args.output)

    print("Training complete.")
    print("Metrics:", metrics)
    print("Saved model to:", "models/reco.joblib")

if __name__ == "__main__":
    main()
