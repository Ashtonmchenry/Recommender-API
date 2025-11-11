"""
load CSV / Kafka data
"""

from collections.abc import Iterable
from typing import Any

import pandas as pd


def from_kafka(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Convert raw iterable of dict messages into a DataFrame."""
    return pd.DataFrame(records)


def from_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
