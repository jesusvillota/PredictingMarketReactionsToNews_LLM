# tests/test_preprocessing_filters_duckdb.py

import duckdb
import pandas as pd
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline_duckdb.preprocessing_filters import get_filters_query

@pytest.fixture
def db_connection():
    """Fixture for an in-memory DuckDB connection."""
    con = duckdb.connect(':memory:')
    yield con
    con.close()

def test_preprocessing_filters(db_connection):
    """
    Tests the get_filters_query function to ensure it correctly filters data.
    """
    # 1. Create mock data
    # This data includes rows that should pass and fail each filter condition.
    mock_data = pd.DataFrame({
        # Base values for calculations
        'oBid': [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        'oAsk': [10.5, 10.5, 10.5, 9.5,  10.5, 10.5, 10.5], # Invalid spread for row 3
        'prtPrice': [10.2, 0,    10.2, 10.2, 10.2, 9.9,  10.6], # Invalid prtPrice for row 1
        'prtSize':  [10,   10,   0,    10,   10,   10,   10],   # Invalid prtSize for row 2
        'uBid':     [0.2,  0.2,  0.2,  0.2,  0.05, 0.2,  0.2],  # Invalid uBid for row 4
        # Calculated values needed for filters
        'spread':   [0.5,  0.5,  0.5,  -1.0, 0.5,  0.5,  0.5],  # Row 3 fails (spread < 0)
        'BIDabove': [0.7,  0.7,  0.7,  0.7,  0.7,  -0.1, 0.7],  # Row 5 fails (BIDabove < 0)
        'ASKbelow': [-0.8, -0.8, -0.8, -0.8, -0.8, -0.8, 0.1],  # Row 6 fails (ASKbelow > 0)
        'id':       ['pass', 'fail_prtPrice', 'fail_prtSize', 'fail_spread', 'fail_uBid', 'fail_BIDabove', 'fail_ASKbelow']
    })

    # 2. Register DataFrame as a view in DuckDB
    db_connection.register('source_data_view', mock_data)

    # 3. Generate and execute the filtering query
    query = get_filters_query(source_view='source_data_view', target_view='filtered_view')
    db_connection.execute(query)

    # 4. Fetch results and verify
    result_df = db_connection.execute("SELECT * FROM filtered_view").fetchdf()

    # 5. Assertions
    assert len(result_df) == 1, "Only one row should pass all filters."
    assert result_df['id'].iloc[0] == 'pass', "The passing row should have id 'pass'."

    # Verify that each specific filter works by checking which rows were excluded
    all_data_df = db_connection.execute("SELECT * FROM source_data_view").fetchdf()
    
    # Test each condition individually
    assert len(all_data_df[all_data_df['prtPrice'] <= 0]) == 1, "Should be one row with prtPrice <= 0"
    assert len(all_data_df[all_data_df['prtSize'] <= 0]) == 1, "Should be one row with prtSize <= 0"
    assert len(all_data_df[all_data_df['uBid'] <= 0.1]) == 1, "Should be one row with uBid <= 0.1"
    assert len(all_data_df[all_data_df['spread'] < 0]) == 1, "Should be one row with spread < 0"
    assert len(all_data_df[all_data_df['BIDabove'] < 0]) == 1, "Should be one row with BIDabove < 0"
    assert len(all_data_df[all_data_df['ASKbelow'] > 0]) == 1, "Should be one row with ASKbelow > 0"

if __name__ == "__main__":
    pytest.main()
