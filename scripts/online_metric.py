"""Compute the online proxy KPI from stored Kafka batch snapshots."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from service.metrics import proxy_kpi  # noqa: E402


def _iter_data_files(paths: Sequence[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_file():
            yield path
        elif path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.suffix.lower() in {".csv", ".parquet"}:
                    yield candidate
        else:
            raise FileNotFoundError(path)


def _load_table(paths: Sequence[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for file_path in _iter_data_files(paths):
        if file_path.suffix.lower() == ".csv":
            frame = pd.read_csv(file_path)
        else:
            frame = pd.read_parquet(file_path)
        frame.columns = [c.strip().lower() for c in frame.columns]
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--responses",
        type=Path,
        required=True,
        nargs="+",
        help="Path(s) to reco_responses batches (files or directories).",
    )
    parser.add_argument(
        "--watches",
        type=Path,
        required=True,
        nargs="+",
        help="Path(s) to watch batches (files or directories).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Time horizon in minutes for the engagement window.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print dataset sizes in addition to the KPI value.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    responses = _load_table(args.responses)
    watches = _load_table(args.watches)

    if responses.empty or watches.empty:
        print("No data found for responses or watches.")
        return 1

    score = proxy_kpi(responses, watches, horizon_min=args.horizon)

    if args.verbose:
        print(f"Loaded {len(responses)} response rows and {len(watches)} watch rows.")
    print(f"proxy_kpi@{args.horizon}min = {score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
