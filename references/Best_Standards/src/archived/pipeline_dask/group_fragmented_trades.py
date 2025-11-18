# src/trades/group_fragmented_trades.py

from logging import Logger
import dask.dataframe as dd
from src import get_logger

def group_fragmented_trades(ddf: dd.DataFrame) -> dd.DataFrame:
    """
    Detects fragmented trades in a Dask DataFrame of options trades.
    """
    
    logger: Logger = get_logger(__name__)
    logger.debug("Starting fragmented trades detection")

    grouping_cols: list[str] = [
        "okey_tk", 
        "okey_xx", 
        "okey_cp",
        "uBid",
        "uAsk", 
        "uPrc",
        "prtExch",
        "prtPrice", 
        "prtType", 
        "timestamp_ny_round3",
        "tradingSession", 
    ]

    grouping_cols: list[str] = [col for col in grouping_cols if col in ddf.columns]
    ddf: dd.DataFrame = ddf.copy()
    
    # # OPTION 1 ----------------------------------------------------------------------------------------
    # if grouping_cols:
    #     agg_ddf = ddf.groupby(grouping_cols, observed=True).agg(
    #         fragment_count=("prtSize", "count"),
    #         prtSize_agg=("prtSize", "sum")
    #     ).reset_index()
    # ddf = ddf.merge(agg_ddf, on=grouping_cols, how="left")
    # # --------------------------------------------------------------------------------------------------
    
    # OPTION 2 ----------------------------------------------------------------------------------------
    # Compute count and sum by group in Dask
    count_ddf = ddf.groupby(grouping_cols, observed=True)["prtSize"].count().rename("fragment_count")
    sum_ddf = ddf.groupby(grouping_cols, observed=True)["prtSize"].sum().rename("prtSize_agg")

    # Reset index to make grouping columns regular columns for merging
    count_ddf = count_ddf.reset_index()
    sum_ddf = sum_ddf.reset_index()

    # Merge aggregated results back to the original dataframe
    ddf = ddf.merge(count_ddf, on=grouping_cols, how="left")
    ddf = ddf.merge(sum_ddf[grouping_cols + ["prtSize_agg"]], on=grouping_cols, how="left")
    # --------------------------------------------------------------------------------------------------

    # Drop duplicates by grouping columns
    ddf = ddf.drop_duplicates(subset=grouping_cols)

    logger.debug(f"Fragmented trades detection completed. Aggregated rows will be computed lazily.")

    return ddf