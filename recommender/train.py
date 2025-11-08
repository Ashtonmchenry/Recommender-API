"""Training entry-point for the ALS-based recommender with registry publishing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy import sparse

from . import serialize  # keep your existing helpers
from .pipeline import transform

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
    """Chronological last-item holdout per user."""

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
    vals = (
        (train["rating"].to_numpy() > rating_threshold).astype(float)
        if "rating" in train.columns
        else np.ones_like(rows, dtype=float)
    )
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


# -------- registry + metadata helpers --------

def _sha256(path: Path) -> str:
    """Compute a content hash so downstream services can verify integrity."""

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dump_metadata(path: Path, payload: Dict[str, Any]) -> None:
    """Write metadata as YAML when available, JSON otherwise."""

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    with path.open("w", encoding="utf-8") as fh:  # pragma: no cover - exercised when PyYAML present
        yaml.safe_dump(payload, fh, sort_keys=False)


def _publish_to_registry(
    model_path: Path,
    registry: str,
    metadata: Dict[str, Any],
    explicit_version: str | None,
) -> Tuple[str, Dict[str, Any], Path]:
    """Copy the trained model into a versioned registry folder and persist metadata."""

    registry_root = Path(registry)
    registry_root.mkdir(parents=True, exist_ok=True)

    version = explicit_version or f"v{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
    version_dir = registry_root / version
    version_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = version_dir / model_path.name
    shutil.copy2(model_path, artifact_path)

    enriched = dict(metadata)
    enriched.update(
        {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "artifact": {
                "filename": artifact_path.name,
                "sha256": _sha256(artifact_path),
            },
        }
    )

    _dump_metadata(version_dir / "meta.yaml", enriched)

    return version, enriched, version_dir


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
    ap.add_argument("--registry", help="Directory (or mounted bucket) for versioned registry publishing")
    ap.add_argument("--registry-version", help="Explicit version tag to publish (e.g., v20241010)")
    ap.add_argument("--metadata-json", help="Optional path to dump metadata as JSON for automation")
    args = ap.parse_args()

    ALS_FACTORS, ALS_REG, ALS_ITERS, RATING_THRESHOLD = args.factors, args.reg, args.iters, args.thr

    df = pd.read_csv(args.input)
    df = _normalize_time(df)
    df = transform.basic_clean(df)

    model, metrics = train_baseline(df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialize.save_model(model, str(output_path))

    base_metadata: Dict[str, Any] = {
        "metrics": metrics,
        "hyperparams": {
            "factors": ALS_FACTORS,
            "reg": ALS_REG,
            "iters": ALS_ITERS,
            "thr": RATING_THRESHOLD,
        },
        "training": {
            "input_path": os.path.abspath(args.input),
            "rows": int(len(df)),
            "n_users": int(len(model.user_map)),
            "n_items": int(len(model.item_map)),
        },
        "artifact": {
            "filename": output_path.name,
            "sha256": _sha256(output_path),
        },
    }

    published_version = None
    registry_dir = None
    if args.registry:
        published_version, metadata, registry_dir = _publish_to_registry(
            model_path=output_path,
            registry=args.registry,
            metadata=base_metadata,
            explicit_version=args.registry_version,
        )
        base_metadata = metadata

    if args.metadata_json:
        metadata_path = Path(args.metadata_json)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(base_metadata, indent=2), encoding="utf-8")

    summary = {
        "saved": str(output_path),
        "registry_version": published_version,
        "registry_dir": str(registry_dir) if registry_dir else None,
        "metrics": metrics,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
