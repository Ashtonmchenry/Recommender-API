"""Compute online proxy metrics by matching rec responses to watch events."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, MutableMapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_movie_ids(value) -> list[int]:
    """Return a list of ints from whatever representation is provided."""
    if isinstance(value, list):
        return [int(v) for v in value]
    if pd.isna(value):
        return []
    if isinstance(value, (tuple, set)):
        return [int(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict) and "movie_ids" in parsed:
                parsed = parsed["movie_ids"]
        except json.JSONDecodeError:
            # fallback: assume comma / space separated
            if value.startswith("[") and value.endswith("]"):
                value = value[1:-1]
            parsed = [v for v in value.replace("|", ",").replace(";", ",").replace(" ", ",").split(",") if v]
        else:
            if isinstance(parsed, str):
                parsed = [parsed]
        if isinstance(parsed, list):
            return [int(v) for v in parsed]
        return []
    # default fallback
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return []


def _build_watch_index(watches: pd.DataFrame) -> MutableMapping[int, list[tuple]]:
    """Group watch events by user for fast lookups during scoring."""
    idx: dict[int, list[tuple]] = {}
    if watches.empty:
        return idx
    grouped = watches.groupby("user_id")
    for user, g in grouped:
        tuples = list(zip(g["ts"].astype(int), g["movie_id"].astype(int)))
        tuples.sort(key=lambda x: x[0])
        idx[int(user)] = tuples
    return idx


def _success_within_window(
    row: pd.Series,
    watch_index: MutableMapping[int, list[tuple]],
    horizon_seconds: int,
) -> int:
    start = int(row["ts"])
    end = start + horizon_seconds
    user = int(row["user_id"])
    candidates = set(row["movie_ids_list"])
    for ts, movie_id in watch_index.get(user, ()):  # type: ignore[arg-type]
        if ts < start:
            continue
        if ts > end:
            break
        if movie_id in candidates:
            return 1
    return 0


def _compute_latency_percentiles(latencies: Iterable[float]) -> dict[str, float]:
    arr = np.fromiter((float(x) for x in latencies if pd.notna(x)), dtype=float)
    if arr.size == 0:
        return {"p50": float("nan"), "p95": float("nan")}
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def evaluate_online(
    responses_csv: str,
    watches_csv: str,
    horizon_min: int = 10,
    k_values: Sequence[int] | None = None,
    out_json: str | None = None,
) -> dict[str, object]:
    """Evaluate online proxy KPI from logged responses and watch events."""

    if k_values is None:
        k_values = (5, 10, 20)

    responses = pd.read_csv(responses_csv)
    watches = pd.read_csv(watches_csv)

    required_resp_cols = {"user_id", "ts", "movie_ids"}
    missing = required_resp_cols - set(responses.columns)
    if missing:
        raise ValueError(f"responses_csv missing required columns: {sorted(missing)}")

    required_watch_cols = {"user_id", "movie_id", "ts"}
    missing = required_watch_cols - set(watches.columns)
    if missing:
        raise ValueError(f"watches_csv missing required columns: {sorted(missing)}")

    responses = responses.copy()
    responses["movie_ids_list"] = responses["movie_ids"].apply(_parse_movie_ids)
    responses["ts"] = responses["ts"].astype(int)

    horizon_seconds = int(horizon_min) * 60
    watch_index = _build_watch_index(watches)

    responses["success"] = responses.apply(
        _success_within_window,
        axis=1,
        watch_index=watch_index,
        horizon_seconds=horizon_seconds,
    )

    summary: dict[str, object] = {
        "horizon_minutes": int(horizon_min),
        "total_responses": int(len(responses)),
        "success_rate": float(responses["success"].mean()) if len(responses) else float("nan"),
    }

    if "latency_ms" in responses.columns:
        summary["latency_ms"] = _compute_latency_percentiles(responses["latency_ms"].to_numpy())

    success_at_k = {}
    for k in k_values:
        if k <= 0:
            continue
        tmp = responses.copy()
        tmp["movie_ids_list"] = tmp["movie_ids_list"].apply(lambda xs, kk=k: xs[:kk])
        tmp["success_k"] = tmp.apply(
            _success_within_window,
            axis=1,
            watch_index=watch_index,
            horizon_seconds=horizon_seconds,
        )
        success_at_k[str(k)] = float(tmp["success_k"].mean()) if len(tmp) else float("nan")

    summary["success_at_k"] = success_at_k

    if out_json:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute online proxy KPI from logs.")
    ap.add_argument("--responses", required=True, help="CSV of recommendation responses")
    ap.add_argument("--watches", required=True, help="CSV of watch events")
    ap.add_argument("--horizon", type=int, default=10, help="Success horizon in minutes")
    ap.add_argument(
        "--out",
        help="Optional path to persist summary JSON",
    )
    ap.add_argument(
        "--k-values",
        default="5,10,20",
        help="Comma-separated top-k cutoffs for success@k calculation",
    )
    args = ap.parse_args()

    k_values = [int(x) for x in args.k_values.split(",") if x]
    results = evaluate_online(
        responses_csv=args.responses,
        watches_csv=args.watches,
        horizon_min=args.horizon,
        k_values=k_values,
        out_json=args.out,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
