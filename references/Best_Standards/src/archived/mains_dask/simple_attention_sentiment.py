# uv run src/mains/simple_attention_sentiment.py

"""
Calculate attention and sentiment measures for whale trades.

Attention: proportion of whale trades (size >= 200) relative to all trades
    Attention_{i,t} = # whale trades / # total trades

Sentiment: bull/bear ratio of whale trades
    Sentiment_{i,t} = (# T_bull - # T_bear) / (# T_bull + # T_bear)
    
    where:
    - T_bull = whale trades that are (buy call) OR (sell put)  [bullish positions]
    - T_bear = whale trades that are (sell call) OR (buy put)  [bearish positions]
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(PROJECT_ROOT))

import dask.dataframe as dd
from typing import Any

from src.config import config_settings, initialize_main, AdaptiveDaskManager, DaskManager
from src.config.config_settings import PROCESSED_PATH

if __name__ == '__main__':

    logger = initialize_main()
    
    # Output directory
    output_dir = PROJECT_ROOT / "_ATTENTION_SENTIMENT_DIRECTION_" / "Simple"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Reading Parquet files from {PROCESSED_PATH}")
    logger.info(f"Output directory: {output_dir}")
    
    with DaskManager() as dask_manager: 
    # with AdaptiveDaskManager() as dask_manager:        
        # Load data with proper filters
        ddf = dd.read_parquet(
            path=PROCESSED_PATH,
            engine=config_settings.parquet["engine"],
            filters=[
                ('ticker_class', '==', 'Equity'),
                ('prtType', '>=', 73),
                ('prtType', '<', 102),
            ],
            columns=[
                'okey_tk', 
                'okey_cp',
                'timestamp_ny', 
                'prtSize_agg',
                'buy_sell_class',
            ],
        )
        
        logger.info(f"Loaded Dask DataFrame with {ddf.npartitions} partitions")
        
        # Extract year_month for grouping
        logger.info(f"Creating year_month column...")
        ddf['year_month'] = ddf['timestamp_ny'].dt.strftime('%Y-%m')

        # Drop timestamp to save memory but keep buy_sell_class and okey_cp for sentiment calculation
        ddf = ddf[['okey_tk', 'year_month', 'prtSize_agg', 'buy_sell_class', 'okey_cp']]
        
        # Repartition for efficiency (optional: skip if partitions are already good)
        logger.info(f"Repartitioning data...")
        ddf = ddf.repartition(partition_size="100MB")
        
        # Create a flag for whale trades (prtSize_agg >= 200)
        logger.info(f"Creating whale trade flag...")
        ddf['is_whale'] = (ddf['prtSize_agg'] >= 200).astype(int)
        
        # Create bull/bear indicators for sentiment calculation based on the formula:
        # Bull: (buy AND call) OR (sell AND put) with size >= 200
        # Bear: (sell AND call) OR (buy AND put) with size >= 200
        logger.info(f"Creating bull/bear indicators for sentiment...")
        is_whale_size = ddf['prtSize_agg'] >= 200
        is_buy = ddf['buy_sell_class'] == 'Buy'
        is_sell = ddf['buy_sell_class'] == 'Sell'
        is_call = ddf['okey_cp'] == 'Call'
        is_put = ddf['okey_cp'] == 'Put'
        
        ddf['whale_bull'] = (is_whale_size & ((is_buy & is_call) | (is_sell & is_put))).astype(int)
        ddf['whale_bear'] = (is_whale_size & ((is_sell & is_call) | (is_buy & is_put))).astype(int)
        
        # Calculate attention and sentiment measures efficiently in a single pass
        logger.info(f"Calculating attention and sentiment measures...")
        
        # Group by ticker and year_month, calculate all metrics in one operation
        stats = (ddf
                .groupby(['okey_tk', 'year_month'])
                .agg({
                    'is_whale': ['sum', 'count'],  # sum gives whale count, count gives total count
                    'whale_bull': 'sum',           # sum gives bullish whale trade count
                    'whale_bear': 'sum'            # sum gives bearish whale trade count
                })
                .reset_index())
        
        # Flatten column names
        stats.columns = ['okey_tk', 'year_month', 'whale_count', 'total_count', 'whale_bull_count', 'whale_bear_count']
        
        # Calculate attention ratio
        stats['attention'] = stats['whale_count'] / stats['total_count']
        
        # Calculate sentiment ratio: (whale_bull - whale_bear) / (whale_bull + whale_bear)
        # Handle division by zero case when no whale trades exist
        whale_total = stats['whale_bull_count'] + stats['whale_bear_count']
        stats['sentiment'] = (stats['whale_bull_count'] - stats['whale_bear_count']) / whale_total
        stats['sentiment'] = stats['sentiment'].fillna(0)  # Set to 0 when no whale trades
        
        # Rename ticker column
        stats = stats.rename(columns={'okey_tk': 'ticker'})
        
        # Select final columns
        attention_result = stats[['ticker', 'year_month', 'whale_count', 'total_count', 'attention', 
                                'whale_bull_count', 'whale_bear_count', 'sentiment']]
        
        # Compute the result
        logger.info(f"Computing final result...")
        pdf_attention = attention_result.compute()
        
        # Save to Parquet
        output_file = output_dir / "attention_sentiment_direction.parquet"
        logger.info(f"Saving to {output_file}")
        
        pdf_attention.to_parquet(
            output_file,
            index=False,
            compression=config_settings.parquet["compression"],
            engine=config_settings.parquet["engine"],
        )
        
        logger.info(f"Saved attention and sentiment data with {len(pdf_attention)} ticker-month combinations")
        logger.info(f"Attention range: {pdf_attention['attention'].min():.4f} - {pdf_attention['attention'].max():.4f}")
        logger.info(f"Sentiment range: {pdf_attention['sentiment'].min():.4f} - {pdf_attention['sentiment'].max():.4f}")
    
    logger.info("Attention sentiment direction calculation completed successfully!")
    logger.info(f"Output file saved to: {output_file}")