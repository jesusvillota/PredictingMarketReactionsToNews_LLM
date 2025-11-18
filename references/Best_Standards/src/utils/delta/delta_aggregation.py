from src import get_logger
import dask.dataframe as dd
import pandas as pd
from datetime import date, datetime
import numpy as np
from pathlib import Path

timeline_path = Path("FILE_MANAGER/timeline_official.txt")
with open(timeline_path, 'r') as file:
    timeline_dates: list[datetime.date] = [
            datetime.strptime(line.strip(), '%Y-%m-%d').date() 
            for line in file.readlines()
            ]
TIMELINE_SORTED: list[datetime.date] = sorted(timeline_dates)
TIMELINE_SET: set[datetime.date] = set(timeline_dates)

def delta_aggregation_dask_optimized(ddf: dd.DataFrame) -> pd.DataFrame:
    """
    Version that keeps more operations in Dask for larger datasets
    """
    logger = get_logger(__name__)

    logger.debug("Starting Dask-optimized delta_aggregation function.")

    logger.debug("Reclassifying trading_hours_class column.")
    ddf = temp_reclassify_trading_hours(ddf)

    logger.debug("Filtering for Equity tickers and after/before market hours.")
    equity_nonmkt = ddf[
        # (ddf['ticker_class'] == 'Equity') & 
        (ddf['trading_hours_class'].isin(['BeforeMarket', 'AfterMarket']))
    ][["okey_tk", "timestamp_ny", "trading_hours_class", "unsigned_delta", "signed_delta"]]

    logger.debug("Adding date column to Dask DataFrame.")
    equity_nonmkt = equity_nonmkt.assign(date=equity_nonmkt['timestamp_ny'].dt.date)

    # equity_nonmkt = equity_nonmkt.persist()

    logger.debug("Performing Dask aggregation (groupby and sum).")
    agg_result = equity_nonmkt.groupby(['okey_tk', 'date', 'trading_hours_class']).agg({
        'unsigned_delta': 'sum',
        'signed_delta': 'sum'
    }).reset_index()

    logger.debug("Computing aggregated result (triggering Dask compute). This may take a while...")
    pdf = agg_result.compute()
    logger.debug(f"Aggregated result shape: {pdf.shape}")
    
    if pdf.empty:
        logger.warning("Aggregated DataFrame is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    logger.debug("Pivoting DataFrame to separate after/before market data.")
    pivot_df = pdf.pivot_table(
        index=['okey_tk', 'date'], 
        columns='trading_hours_class',
        values=['unsigned_delta', 'signed_delta'],
        fill_value=0,
        aggfunc='sum'
    ).reset_index()

    # Flatten column names
    pivot_df.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] 
                       for col in pivot_df.columns]

    # Rename columns to match expected output
    column_mapping = {
        'AfterMarket_unsigned_delta': 'after_market_gross_delta_t',
        'AfterMarket_signed_delta': 'after_market_net_delta_t',
        'BeforeMarket_unsigned_delta': 'before_market_gross_delta_raw',
        'BeforeMarket_signed_delta': 'before_market_net_delta_raw'
    }
    pivot_df = pivot_df.rename(columns=column_mapping)

    logger.debug("Shifting before market data to next day and combining results.")
    # Handle the date shifting for before market data
    # This requires joining with the previous day's after market data
    results = []
    n_rows = 0
    for _, row in pivot_df.iterrows():
        ticker = row['okey_tk']
        date = row['date']
        # Get after market data for current date
        after_gross = row.get('after_market_gross_delta_t', 0)
        after_net = row.get('after_market_net_delta_t', 0)
        # Find before market data for next day
        before_gross = 0
        before_net = 0
        if date in TIMELINE_SET:
            try:
                current_idx = TIMELINE_SORTED.index(date)
                if current_idx + 1 < len(TIMELINE_SORTED):
                    next_date = TIMELINE_SORTED[current_idx + 1]
                    # Find before market data for this ticker on next_date
                    next_day_data = pivot_df[
                        (pivot_df['okey_tk'] == ticker) & 
                        (pivot_df['date'] == next_date)
                    ]
                    if not next_day_data.empty:
                        before_gross = next_day_data.iloc[0].get('before_market_gross_delta_raw', 0)
                        before_net = next_day_data.iloc[0].get('before_market_net_delta_raw', 0)
            except ValueError:
                logger.warning(f"Date {date} not found in TIMELINE_SORTED.")
                pass
        # Only include if we have some data
        if after_gross != 0 or after_net != 0 or before_gross != 0 or before_net != 0:
            results.append({
                'okey_tk': ticker,
                'date': date,
                'gross_delta': after_gross + before_gross,
                'net_delta': after_net + before_net,
                'after_market_gross_delta_t': after_gross,
                'before_market_gross_delta_t_plus_1': before_gross,
                'after_market_net_delta_t': after_net,
                'before_market_net_delta_t_plus_1': before_net
            })
            n_rows += 1
            if n_rows % 10000 == 0:
                logger.debug(f"Processed {n_rows} rows in result aggregation loop.")

    logger.debug(f"Dask-optimized delta_aggregation completed. Returning DataFrame with {len(results)} rows.")
    return pd.DataFrame(results)


def temp_reclassify_trading_hours(ddf: dd.DataFrame) -> dd.DataFrame:
    """
    Optimized trading hours classification using vectorized operations
    """
    from src import get_logger
    logger = get_logger(__name__)
    logger.debug("Reclassifying trading_hours_class in temp_reclassify_trading_hours.")
    # Drop existing column if present
    if 'trading_hours_class' in ddf.columns:
        logger.debug("Dropping existing trading_hours_class column.")
        ddf = ddf.drop(columns=['trading_hours_class'])
    
    def classify_partition(df):
        ts = df["timestamp_ny"]
        d = ts.dt.date
        h = ts.dt.hour
        m = ts.dt.minute
        is_early = d.isin(EARLY_CLOSE_DATES)
        
        # Classify using vectorized conditions
        before = (h < 9) | ((h == 9) & (m < 30))
        end_hour = np.where(is_early.to_numpy(), 13, 16)
        market = ((h > 9) | ((h == 9) & (m >= 30))) & (h < end_hour)
        
        # Default to AfterMarket when not before or market
        df['trading_hours_class'] = np.select(
            [before, market],
            ['BeforeMarket', 'MarketHours'],
            default='AfterMarket'
        )
        return df
    
    meta = ddf._meta.copy()
    meta['trading_hours_class'] = 'Unknown'
    ddf = ddf.map_partitions(classify_partition, meta=meta)
    logger.debug("Completed reclassification of trading_hours_class.")
    return ddf

def _load_early_close_dates(path: Path) -> set[datetime.date]:
    with path.open("r") as f:
        return {
            datetime.strptime(line.strip(), "%Y-%m-%d").date()
            for line in f
            if line.strip()
        }

# Resolve the dates file relative to this module (robust for workers)
# _EARLY_CLOSE_PATH = (Path(__file__).resolve().parent.parent / "early_closing" / "dates.txt")
# EARLY_CLOSE_DATES = _load_early_close_dates(_EARLY_CLOSE_PATH)
EARLY_CLOSE_DATES = _load_early_close_dates(Path("src/early_closing/dates.txt"))