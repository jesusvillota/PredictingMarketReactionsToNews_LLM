# tests/test_classify_duckdb.py

import duckdb
import pandas as pd
import pytest
from pathlib import Path
import sys
import tempfile
import os
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline_duckdb.classify import get_classify_query

@pytest.fixture
def db_connection():
    """Fixture for an in-memory DuckDB connection."""
    con = duckdb.connect(':memory:')
    yield con
    con.close()

@pytest.fixture
def mock_equity_tickers():
    """Creates a mock equity tickers JSON file."""
    tickers = ["AAPL", "MSFT", "GOOG"]
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as f:
        json.dump(tickers, f)
        f.flush()
        yield Path(f.name)
    os.remove(f.name)

@pytest.fixture
def mock_early_close_dates():
    """Creates a mock early close dates txt file."""
    dates = ["2023-11-24", "2023-12-24"]
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".txt") as f:
        f.write("\n".join(dates))
        f.flush()
        yield Path(f.name)
    os.remove(f.name)

def test_classify_query(db_connection, mock_equity_tickers, mock_early_close_dates):
    """Tests the get_classify_query for all classification columns."""
    mock_data = pd.DataFrame({
        # Moneyness (delta)
        'prtDe': [0.5, 0.8, 0.2, None, 0.5, 0.5], # ATM, ITM, OTM, unknown
        # Moneyness (ratio)
        'moneyness': [1.0, 0.8, 1.3, 1.3, 0.8, None], # ATM, ITM(C), OTM(C), ITM(P), OTM(P), unknown
        'okey_cp': ['Call', 'Call', 'Call', 'Put', 'Put', 'Call'],
        # Trading hours & Moment of day
        'timestamp_ny': [
            datetime(2023, 11, 24, 9, 0),   # before_market
            datetime(2023, 11, 24, 10, 0),  # market (early close), morning
            datetime(2023, 11, 24, 12, 30), # market (early close), midday
            datetime(2023, 11, 24, 14, 0),  # after_market (early close), overnight
            datetime(2023, 11, 25, 14, 0),  # market (normal day), afternoon
            datetime(2023, 11, 25, 20, 0),  # after_market (normal day), overnight
        ],
        # Ticker
        'okey_tk': ['AAPL', 'SPY', 'MSFT', 'TSLA', 'GOOG', 'AMZN'],
        # Buy/Sell & Bid/Ask Proximity
        'prtPrice':     [10.2, 10.8, 10.5, 10.0, 10.0, 10.0],
        'midpointNBBO': [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        'midpointExch': [10.0, 10.0, 10.0, 9.9, 10.1, 10.0],
        'oBid':         [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        'oAsk':         [11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
        # Trade Type
        'prtType': [1, 102, 200, 50, 101, 102],
        # DTE
        'dte': [5, 10, 20, 50, 100, 400],
        'id': range(6)
    })

    db_connection.register('source_view', mock_data)

    query = get_classify_query(
        'source_view', 
        'classified_view',
        equity_tickers_path=mock_equity_tickers,
        early_close_dates_path=mock_early_close_dates
    )
    db_connection.execute(query)
    result = db_connection.execute("SELECT * FROM classified_view ORDER BY id").fetchdf()

    # Assertions
    assert len(result) == 6

    # Moneyness Delta
    assert result['moneyness_class_delta'].tolist()[:4] == ['ATM', 'ITM', 'OTM', 'unknown']
    # Moneyness Ratio
    assert result['moneyness_class_ratio'].tolist() == ['ATM', 'ITM', 'OTM', 'ITM', 'OTM', 'unknown']
    # Trading Hours
    assert result['trading_hours_class'].tolist() == ['before_market', 'market', 'market', 'after_market', 'market', 'after_market']
    # Moment of Day
    assert result['moment_of_the_day'].tolist() == ['overnight', 'morning', 'midday', 'overnight', 'afternoon', 'overnight']
    # Ticker Class
    assert result['ticker_class'].tolist() == ['equity', 'other', 'equity', 'other', 'equity', 'other']
    # Buy/Sell Class
    assert result['buy_sell_class'].tolist() == ['buy', 'buy', 'buy', 'buy', 'sell', 'midpoint']
    # Bid/Ask Proximity
    assert result['bid_ask_proximity'].tolist() == ['closer_to_bid', 'closer_to_ask', 'same_distance', 'closer_to_bid', 'closer_to_bid', 'closer_to_bid']
    # Trade Type
    assert result['trade_type'].tolist() == ['simple', 'complex', 'complex', 'simple', 'simple', 'complex']
    # Time to Expiry
    assert result['time_to_expiry'].tolist() == [
        'less than a week', '1-2 weeks', '2-4 weeks', '1-3 months', '3-12 months', 'over a year'
    ]

if __name__ == "__main__":
    pytest.main()
