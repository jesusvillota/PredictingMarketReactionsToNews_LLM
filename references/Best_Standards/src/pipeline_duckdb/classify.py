# src/pipeline_duckdb/classify.py

from pathlib import Path
from datetime import date, datetime
from typing import Set
import json


def _load_early_close_dates(path: Path) -> Set[date]:
    """Load early closing dates from file"""
    with path.open("r") as f:
        return {
            datetime.strptime(line.strip(), "%Y-%m-%d").date()
            for line in f
            if line.strip()
        }


def _load_equity_tickers(path: Path) -> Set[str]:
    """Load equity tickers from JSON file"""
    try:
        with path.open("r") as f:
            return set(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def get_classify_query(
    source_view: str, 
    target_view: str = "classified_data",
    equity_tickers_path: Path = Path("src/utils/tickers/EQTS.json"),
    early_close_dates_path: Path = Path("src/utils/early_closing/dates.txt")
) -> str:
    """
    Generate SQL query to classify trades.
    
    Adds classification columns:
    - moneyness_class_delta: ATM/ITM/OTM based on delta
    - moneyness_class_ratio: ATM/ITM/OTM based on strike/underlying ratio
    - trading_hours_class: before_market/market/after_market
    - moment_of_the_day: morning/midday/afternoon/overnight
    - ticker_class: equity/other
    - buy_sell_class: buy/sell/midpoint
    - bid_ask_proximity: closer_to_bid/same_distance/closer_to_ask
    - trade_type: simple/complex
    - time_to_expiry: lt_1w/1w_to_2w/2w_to_4w/1m_to_3m/3m_to_12m/gt_1y
    
    Category naming conventions:
    - All categories use snake_case with underscores
    - Time abbreviations: lt=less than, gt=greater than, w=weeks, m=months, y=years
    - Financial abbreviations kept in uppercase: ATM, ITM, OTM
    - All other categories use lowercase with underscores
    """
    
    # Load reference data
    equity_tickers = _load_equity_tickers(equity_tickers_path)
    early_close_dates = _load_early_close_dates(early_close_dates_path)
    
    # Build SQL lists for matching (convert tickers to strings to handle mixed types)
    equity_tickers_str = [str(ticker) for ticker in equity_tickers]
    equity_tickers_sql = ", ".join([f"'{ticker}'" for ticker in sorted(equity_tickers_str)])
    early_close_dates_sql = ", ".join([f"DATE '{d}'" for d in sorted(early_close_dates)])
    
    return f"""
        CREATE OR REPLACE VIEW {target_view} AS
        SELECT *,
            
            -- Moneyness classification (Delta-based)
            CASE 
                WHEN ABS(prtDe) >= 0.35 AND ABS(prtDe) <= 0.65 THEN 'ATM'
                WHEN ABS(prtDe) > 0.65 THEN 'ITM'
                WHEN ABS(prtDe) < 0.35 THEN 'OTM'
                ELSE 'unknown'
            END AS moneyness_class_delta,
            
            -- Moneyness classification (Ratio-based)
            CASE 
                WHEN moneyness >= 0.9 AND moneyness <= 1.2 THEN 'ATM'
                WHEN okey_cp = 'Call' AND moneyness < 0.9 THEN 'ITM'
                WHEN okey_cp = 'Call' AND moneyness > 1.2 THEN 'OTM'
                WHEN okey_cp = 'Put' AND moneyness > 1.2 THEN 'ITM'
                WHEN okey_cp = 'Put' AND moneyness < 0.9 THEN 'OTM'
                ELSE 'unknown'
            END AS moneyness_class_ratio,
            
            -- Trading hours classification
            CASE 
                WHEN (EXTRACT(HOUR FROM timestamp_ny) * 60 + EXTRACT(MINUTE FROM timestamp_ny)) < 570 
                    THEN 'before_market'
                WHEN (EXTRACT(HOUR FROM timestamp_ny) * 60 + EXTRACT(MINUTE FROM timestamp_ny)) >= 570 
                    AND (EXTRACT(HOUR FROM timestamp_ny) * 60 + EXTRACT(MINUTE FROM timestamp_ny)) < 
                        CASE 
                            WHEN CAST(timestamp_ny AS DATE) IN ({early_close_dates_sql}) THEN 780
                            ELSE 960
                        END
                    THEN 'market'
                ELSE 'after_market'
            END AS trading_hours_class,
            
            -- Moment of the day classification
            CASE 
                WHEN (EXTRACT(HOUR FROM timestamp_ny) * 60 + EXTRACT(MINUTE FROM timestamp_ny)) >= 570 
                    AND (EXTRACT(HOUR FROM timestamp_ny) * 60 + EXTRACT(MINUTE FROM timestamp_ny)) < 660 
                    THEN 'morning'
                WHEN (EXTRACT(HOUR FROM timestamp_ny) * 60 + EXTRACT(MINUTE FROM timestamp_ny)) >= 660 
                    AND (EXTRACT(HOUR FROM timestamp_ny) * 60 + EXTRACT(MINUTE FROM timestamp_ny)) < 780 
                    THEN 'midday'
                WHEN CAST(timestamp_ny AS DATE) NOT IN ({early_close_dates_sql})
                    AND (EXTRACT(HOUR FROM timestamp_ny) * 60 + EXTRACT(MINUTE FROM timestamp_ny)) >= 780 
                    AND (EXTRACT(HOUR FROM timestamp_ny) * 60 + EXTRACT(MINUTE FROM timestamp_ny)) < 
                        CASE 
                            WHEN CAST(timestamp_ny AS DATE) IN ({early_close_dates_sql}) THEN 780
                            ELSE 960
                        END
                    THEN 'afternoon'
                ELSE 'overnight'
            END AS moment_of_the_day,
            
            -- Ticker classification
            CASE 
                WHEN okey_tk IN ({equity_tickers_sql}) THEN 'equity'
                ELSE 'other'
            END AS ticker_class,
            
            -- buy/sell classification
            CASE 
                WHEN prtPrice != midpointNBBO AND prtPrice > midpointNBBO THEN 'buy'
                WHEN prtPrice != midpointNBBO AND prtPrice < midpointNBBO THEN 'sell'
                WHEN prtPrice = midpointNBBO AND prtPrice > midpointExch THEN 'buy'
                WHEN prtPrice = midpointNBBO AND prtPrice < midpointExch THEN 'sell'
                ELSE 'midpoint'
            END AS buy_sell_class,
            
            -- Bid/Ask proximity classification
            CASE 
                WHEN ABS(prtPrice - oBid) < ABS(prtPrice - oAsk) THEN 'closer_to_bid'
                WHEN ABS(prtPrice - oBid) = ABS(prtPrice - oAsk) THEN 'same_distance'
                WHEN ABS(prtPrice - oBid) > ABS(prtPrice - oAsk) THEN 'closer_to_ask'
                ELSE 'unknown'
            END AS bid_ask_proximity,
            
            -- Trade type classification
            CASE 
                WHEN prtType >= 102 THEN 'complex'
                ELSE 'simple'
            END AS trade_type,
            
            -- Time to expiry classification
            -- Abbreviations: lt=less than, gt=greater than, w=weeks, m=months, y=years
            CASE 
                WHEN dte < 7 THEN 'lt_1w'
                WHEN dte >= 7 AND dte < 14 THEN '1w_to_2w'
                WHEN dte >= 14 AND dte < 28 THEN '2w_to_4w'
                WHEN dte >= 28 AND dte < 91 THEN '1m_to_3m'
                WHEN dte >= 91 AND dte < 365 THEN '3m_to_12m'
                WHEN dte >= 365 THEN 'gt_1y'
                ELSE 'unknown'
            END AS time_to_expiry
            
        FROM {source_view}
    """
