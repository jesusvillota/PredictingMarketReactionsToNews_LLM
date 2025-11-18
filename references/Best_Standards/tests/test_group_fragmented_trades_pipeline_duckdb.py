# tests/test_group_fragmented_trades_pipeline_duckdb.py

import duckdb
import pandas as pd
import pytest
from pathlib import Path
import sys
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline_duckdb.group_fragmented_trades import get_group_fragmented_query

@pytest.fixture
def db_connection():
    """Fixture for an in-memory DuckDB connection."""
    con = duckdb.connect(':memory:')
    yield con
    con.close()

def test_group_fragmented_trades_query(db_connection):
    """
    Tests the get_group_fragmented_query function to ensure it correctly
    groups fragmented trades.
    """
    # 1. Create mock data
    # Two groups of trades. Group 1 has 3 fragments, Group 2 has 2 fragments.
    # Group 3 is a single trade.
    base_timestamp = datetime(2023, 1, 1, 10, 0, 0)
    mock_data = pd.DataFrame({
        'okey_tk': ['AAPL', 'AAPL', 'MSFT', 'AAPL', 'MSFT', 'GOOG'],
        'okey_xx': [150.0, 150.0, 300.0, 150.0, 300.0, 2000.0],
        'okey_cp': ['Call', 'Call', 'Put', 'Call', 'Put', 'Call'],
        'uBid': [149.0, 149.0, 299.0, 149.0, 299.0, 1999.0],
        'uAsk': [151.0, 151.0, 301.0, 151.0, 301.0, 2001.0],
        'uPrc': [150.0, 150.0, 300.0, 150.0, 300.0, 2000.0],
        'prtExch': ['A', 'A', 'B', 'A', 'B', 'C'],
        'prtPrice': [5.0, 5.0, 10.0, 5.0, 10.0, 15.0],
        'prtType': [102, 102, 102, 102, 102, 1], # Group 1 & 2 are complex, 3 is simple
        'timestamp_ny_round3': [base_timestamp, base_timestamp, base_timestamp, base_timestamp, base_timestamp, base_timestamp],
        'tradingSession': ['Regular', 'Regular', 'Regular', 'Regular', 'Regular', 'Regular'],
        'prtSize': [10, 20, 5, 30, 15, 50],
        'other_col': [1, 2, 3, 4, 5, 6] # To ensure other columns are preserved
    })

    # 2. Register DataFrame as a view in DuckDB
    db_connection.register('source_data_view', mock_data)

    # 3. Generate and execute the grouping query
    query = get_group_fragmented_query(source_view='source_data_view', target_view='grouped_view')
    db_connection.execute(query)

    # 4. Fetch results and verify
    result_df = db_connection.execute("SELECT * FROM grouped_view ORDER BY okey_tk, okey_xx").fetchdf()

    # 5. Assertions
    assert len(result_df) == 3, "Should result in 3 unique groups."
    
    # Check group 1 (AAPL)
    aapl_group = result_df[result_df['okey_tk'] == 'AAPL']
    assert len(aapl_group) == 1
    assert aapl_group['fragment_count'].iloc[0] == 3
    assert aapl_group['prtSize_agg'].iloc[0] == 60 # 10 + 20 + 30
    assert 'other_col' in aapl_group.columns, "Other columns should be preserved."

    # Check group 2 (MSFT)
    msft_group = result_df[result_df['okey_tk'] == 'MSFT']
    assert len(msft_group) == 1
    assert msft_group['fragment_count'].iloc[0] == 2
    assert msft_group['prtSize_agg'].iloc[0] == 20 # 5 + 15

    # Check group 3 (GOOG)
    goog_group = result_df[result_df['okey_tk'] == 'GOOG']
    assert len(goog_group) == 1
    assert goog_group['fragment_count'].iloc[0] == 1
    assert goog_group['prtSize_agg'].iloc[0] == 50

    # Check that QUALIFY keeps only one row per group
    # The original table has 6 rows.
    assert len(db_connection.execute("SELECT * FROM source_data_view").fetchall()) == 6
    assert len(result_df) == 3

if __name__ == "__main__":
    pytest.main()
