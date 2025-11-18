from __future__ import annotations

from datetime import datetime
from typing import List, Tuple, Union

import pandas as pd


ParquetFilter = Tuple[str, str, object]
TimestampLike = Union[pd.Timestamp, datetime]


def _normalize_timestamp(value: TimestampLike) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    elif ts.tz is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def build_parquet_filters(
    ticker_type: str,
    strategy_type: str,
    start: TimestampLike | None = None,
    end: TimestampLike | None = None,
) -> List[ParquetFilter]:
    """Build parquet filters for dask.dataframe.read_parquet(filters=...).

    Parameters
    - ticker_type: "equity" | "all"
    - strategy_type: "simple" | "complex" | "all"
    - start: inclusive lower bound for `timestamp_ny`
    - end: inclusive upper bound for `timestamp_ny`

    Returns
    A list of tuple filters suitable for Dask/pyarrow predicate pushdown.
    """
    filters: list[ParquetFilter] = []

    # Ticker class
    if ticker_type == "equity":
        filters.append(("ticker_class", "==", "equity"))
    # else: no filter on ticker_class

    # Strategy type via prtType
    if strategy_type == "simple":
        filters.append(("prtType", ">=", 73))
        filters.append(("prtType", "<", 102))
    elif strategy_type == "complex":
        filters.append(("prtType", ">=", 102))
    elif strategy_type == "all":
        filters.append(("prtType", ">=", 73))
    else:
        raise ValueError(f"Unknown strategy_type: {strategy_type}")

    if start is not None:
        filters.append(("timestamp_ny", ">=", _normalize_timestamp(start)))

    if end is not None:
        filters.append(("timestamp_ny", "<=", _normalize_timestamp(end)))

    return filters


