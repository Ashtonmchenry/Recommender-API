"""Data cleansing helpers shared across training and pipeline flows."""

from __future__ import annotations

import pandas as pd

from ..features import basic_clean as _basic_clean


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Expose the notebook's ``basic_clean`` via the pipeline module path."""

    return _basic_clean(df)


__all__ = ["basic_clean"]
