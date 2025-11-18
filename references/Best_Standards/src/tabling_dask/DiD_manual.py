# Preparing data for DiD

from src.config import config_settings, initialize_main, DaskManager, get_logger

from dataclasses import dataclass
from datetime import date
import argparse
import math
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd
import dask.dataframe as dd

from src.tabling_dask.common import build_parquet_filters
from src.config.config_settings import PROCESSED_PATH, tables

@dataclass(frozen=True)
class ClosurePeriod:
    exchange: str
    start: date
    end: date
    label: str
    
CLOSURE_PERIODS: Sequence[ClosurePeriod] = (
    ClosurePeriod("CBOE", date(2020, 3, 16), date(2020, 6, 15), "covid_2020"),
    ClosurePeriod("PHLX", date(2020, 3, 17), date(2020, 6, 3), "covid_2020"),
    ClosurePeriod("BOX", date(2020, 3, 20), date(2020, 5, 4), "covid_2020"),
    ClosurePeriod("NYSE", date(2020, 3, 23), date(2020, 5, 26), "covid_2020"), 
)

def _window_bounds(window_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    min_start = min(period.start for period in CLOSURE_PERIODS)
    max_start = max(period.start for period in CLOSURE_PERIODS)
    
    # Read and parse timeline in one go
    timeline = pd.to_datetime([line.strip() for line in open("output/timeline.txt") if line.strip()])
    
    # Find indices and compute window bounds
    min_idx = timeline.searchsorted(pd.Timestamp(min_start), side='left')
    max_idx = timeline.searchsorted(pd.Timestamp(max_start), side='right') - 1
    
    start_idx = max(0, min_idx - window_days)
    end_idx = min(len(timeline) - 1, max_idx + window_days)
    
    return timeline[start_idx], timeline[end_idx]

def load_trades(window_days: int) -> pd.DataFrame:
    logger = get_logger(__name__)
    required_columns = [
        "timestamp_ny",
        "prtExch",
        "prtSize_agg",
        "quoted_spread",
        "prtDe",
        "prtIv",
        "moneyness",
        "time_to_expiry",
        "buy_sell_class",
        "okey_cp",
        "okey_tk",
    ]

    start_bound, end_bound = _window_bounds(window_days)
    logger.debug(
        "Date window bounds (business days): %s to %s",
        start_bound.date(),
        end_bound.date(),
    )

    filters = build_parquet_filters(
        ticker_type="equity",
        strategy_type="simple",
        start=start_bound,
        end=end_bound,
    )
    logger.info(
        "Loading processed trades for floor exchanges and aggregating before computation..."
    )
    logger.debug("Parquet filters (with date bounds): %s", filters)

    with DaskManager():
        try:
            ddf = dd.read_parquet(
                path=PROCESSED_PATH,
                engine=config_settings.parquet["engine"],
                columns=required_columns,
                filters=filters,
                split_row_groups="infer",
            )
        except Exception:  # pragma: no cover - logging adds context
            logger.exception("Failed to read processed parquet data")
            raise
        
    df = ddf.compute()
    
df = load_trades()

import pandas as pd
import numpy as np
from differences import ATTgt, aggte  # pip install differences
import matplotlib.pyplot as plt
    
did = pd.DataFrame()
    
did["Y"] = df["quoted_spread"]
did["G"] = df["prtExch"]
did["treated"] = df["prtExch"].isin(CLOSURE_PERIODS.exchange)


did[""]