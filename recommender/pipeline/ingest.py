"""
load CSV / Kafka data
"""

from typing import Iterable, Dict, Any, List
import pandas as pd

def from_kafka(records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Convert raw iterable of dict messages into a DataFrame."""
    return pd.DataFrame(records)

def from_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
