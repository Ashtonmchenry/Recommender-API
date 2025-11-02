"""Offline evaluation script that mirrors the ALS notebook workflow."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from .pipeline import transform
from .train import (
    MIN_INTERACTIONS,
    RATING_THRESHOLD,
    EVAL_USERS_CAP,
    build_UI,
    last_item_holdout,
    reindex,
    train_baseline,
)

DEFAULT_K_VALUES: Sequence[int] = (5, 10, 20)


def _precision_at_k(recs: Sequence[int], truth: Sequence[int], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for item in recs[:k] if item in truth)
    return hits / k


def _recall_at_k(recs: Sequence[int], truth: Sequence[int], k: int) -> float:
    if not truth:
        return 0.0
    hits = sum(1 for item in recs[:k] if item in truth)
    return hits / len(truth)


def _dcg_at_k(recs: Sequence[int], truth: Sequence[int], k: int) -> float:
    dcg = 0.0
    for rank, item in enumerate(recs[:k], start=1):
        if item in truth:
            dcg += 1.0 / np.log2(rank + 1)
    return dcg


def _ndcg_at_k(recs: Sequence[int], truth: Sequence[int], k: int) -> float:
    ideal_hits = min(len(truth), k)
    if ideal_hits == 0:
        return 0.0
    ideal_dcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    if ideal_dcg == 0:
        return 0.0
    return _dcg_at_k(recs, truth, k) / ideal_dcg


def _map_at_k(recs: Sequence[int], truth: Sequence[int], k: int) -> float:
    if not truth:
        return 0.0
    hits = 0
    score = 0.0
    truth_set = set(truth)
    for rank, item in enumerate(recs[:k], start=1):
        if item in truth_set:
            hits += 1
            score += hits / rank
    return score / min(len(truth_set), k)


@dataclass
class UserMetrics:
    precision: Dict[int, float]
    recall: Dict[int, float]
    ndcg: Dict[int, float]
    map: Dict[int, float]


def _evaluate_user(recs: Sequence[int], truth: Sequence[int], k_values: Sequence[int]) -> UserMetrics:
    precision, recall, ndcg, apk = {}, {}, {}, {}
    for k in k_values:
        precision[k] = _precision_at_k(recs, truth, k)
        recall[k] = _recall_at_k(recs, truth, k)
        ndcg[k] = _ndcg_at_k(recs, truth, k)
        apk[k] = _map_at_k(recs, truth, k)
    return UserMetrics(precision=precision, recall=recall, ndcg=ndcg, map=apk)


def _aggregate_metrics(metrics: Iterable[UserMetrics], k_values: Sequence[int]) -> Dict[str, Dict[str, float]]:
    metrics_list = list(metrics)
    if not metrics_list:
        return {
            "precision": {str(k): float("nan") for k in k_values},
            "recall": {str(k): float("nan") for k in k_values},
            "ndcg": {str(k): float("nan") for k in k_values},
            "map": {str(k): float("nan") for k in k_values},
        }

    def _mean(selector) -> Dict[str, float]:
        return {
            str(k): float(np.mean([selector(m, k) for m in metrics_list])) for k in k_values
        }

    return {
        "precision": _mean(lambda m, k: m.precision[k]),
        "recall": _mean(lambda m, k: m.recall[k]),
        "ndcg": _mean(lambda m, k: m.ndcg[k]),
        "map": _mean(lambda m, k: m.map[k]),
    }


def _collect_truth(test_df: pd.DataFrame) -> Mapping[int, List[int]]:
    truth: Dict[int, List[int]] = {}
    for row in test_df.itertuples(index=False):
        truth.setdefault(int(row.uidx), []).append(int(getattr(row, "true_item", row.iidx)))
    return truth


def _bucket_by_activity(train_df: pd.DataFrame) -> Dict[int, str]:
    counts = train_df.groupby("uidx").size()

    def _label(n: int) -> str:
        if n < 10:
            return "cold(<10)"
        if n < 50:
            return "warm(10-49)"
        return "hot(50+)"

    return {int(uidx): _label(int(count)) for uidx, count in counts.items()}


def run_offline_eval(
    input_csv: str,
    out_json: str | None = None,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> Dict[str, object]:
    df = pd.read_csv(input_csv)
    df = transform.basic_clean(df)
    if not {"user_id", "item_id", "timestamp"}.issubset(df.columns):
        raise ValueError("input must contain user_id, item_id, timestamp columns")
    if "rating" not in df.columns:
        df["rating"] = 1.0

    df = reindex(df, MIN_INTERACTIONS)
    train_df, test_df = last_item_holdout(df)
    if train_df.empty or test_df.empty:
        raise ValueError("Not enough data after filtering to run evaluation")

    n_users = int(df["uidx"].max()) + 1
    n_items = int(df["iidx"].max()) + 1
    UI = build_UI(train_df, n_users, n_items, rating_threshold=RATING_THRESHOLD)

    model, training_metrics = train_baseline(df)

    test_with_ratings = test_df.merge(
        df[["uidx", "iidx", "rating"]],
        left_on=["uidx", "true_item"],
        right_on=["uidx", "iidx"],
        how="left",
    )
    test_with_ratings = test_with_ratings.rename(columns={"rating": "rating_true"})
    test_with_ratings = test_with_ratings.dropna(subset=["rating_true"])
    test_with_ratings = test_with_ratings[test_with_ratings["rating_true"] >= RATING_THRESHOLD]

    truth = _collect_truth(test_with_ratings)
    if not truth:
        raise ValueError("No positive interactions found in holdout for evaluation")
    eval_users = sorted(truth.keys())
    if len(eval_users) > EVAL_USERS_CAP:
        rng = np.random.RandomState(42)
        eval_users = sorted(rng.choice(eval_users, size=EVAL_USERS_CAP, replace=False).tolist())

    user_metrics: Dict[int, UserMetrics] = {}
    max_k = max(k_values) if k_values else 0
    for u in eval_users:
        recs = model.recommend(u, UI, k=max_k, exclude_seen=True) if max_k else []
        user_metrics[u] = _evaluate_user(recs, truth.get(u, []), k_values)

    overall = _aggregate_metrics(user_metrics.values(), k_values)

    buckets = _bucket_by_activity(train_df)
    bucket_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for bucket in sorted(set(buckets.values())):
        members = [metrics for uid, metrics in user_metrics.items() if buckets.get(uid) == bucket]
        bucket_summary[bucket] = _aggregate_metrics(members, k_values)

    results: Dict[str, object] = {
        "k_values": [int(k) for k in k_values],
        "users_evaluated": len(eval_users),
        "overall": overall,
        "subpopulations": bucket_summary,
        "model_training_metrics": training_metrics,
        "schema": {
            "input_required_columns": ["user_id", "item_id", "timestamp"],
            "optional_columns": ["rating"],
        },
    }

    if out_json:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline evaluation for the ALS recommender")
    parser.add_argument("--input", required=True, help="CSV with user_id,item_id,rating,timestamp")
    parser.add_argument("--out", help="Optional path to persist JSON summary")
    parser.add_argument("--k-values", default="5,10,20", help="Comma-separated top-k cutoffs")
    args = parser.parse_args()

    k_values = tuple(int(x) for x in args.k_values.split(",") if x)
    results = run_offline_eval(args.input, args.out, k_values or DEFAULT_K_VALUES)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
