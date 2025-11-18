"""
Test timestamp_ny equivalence from two different sources.

This test validates that timestamp_ny (derived from UTC with DST handling)
is exactly equivalent to timestamp_ny_2 (derived from Chicago time + 1 hour).

Both New York and Chicago are in the same DST schedule, so:
- NY Time = UTC - 4 hours (EDT) or UTC - 5 hours (EST)
- Chicago Time = UTC - 5 hours (CDT) or UTC - 6 hours (CST)
- Therefore: NY Time = Chicago Time + 1 hour (always)
"""

import duckdb
import pandas as pd
import pytest
from pathlib import Path
import sys
import tempfile
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline_duckdb.create_new_vars import get_create_vars_1_query


@pytest.fixture
def db_connection():
    """Fixture for an in-memory DuckDB connection."""
    con = duckdb.connect(':memory:')
    yield con
    con.close()


@pytest.fixture
def mock_dst_csv():
    """Creates a mock DST csv file and returns its path."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".csv") as f:
        f.write("year,summer,winter\n")
        f.write("2021,2021-03-14 02:00:00,2021-11-07 02:00:00\n")
        f.write("2023,2023-03-12 02:00:00,2023-11-05 02:00:00\n")
        f.write("2024,2024-03-10 02:00:00,2024-11-03 02:00:00\n")
        f.flush()
        yield f.name
    os.remove(f.name)


@pytest.fixture(autouse=True)
def patch_dst_csv_path(monkeypatch, mock_dst_csv):
    """Monkeypatch get_dst_csv_path to use our mock file."""
    monkeypatch.setattr('src.pipeline_duckdb.create_new_vars.get_dst_csv_path', lambda: mock_dst_csv)


def test_timestamp_ny_equivalence_comprehensive(db_connection):
    """
    Test that timestamp_ny and timestamp_ny_2 are exactly equivalent.
    
    Tests multiple scenarios:
    - Winter/EST period (January, February, December)
    - Summer/EDT period (June, July, August)
    - Around DST transitions (March and November)
    - Different years (2021, 2023, 2024)
    """
    
    # Create test data with both UTC timestamp and Chicago timestamp
    # NY Time = Chicago Time + 1 hour (both follow same DST schedule)
    test_cases = [
        # Format: (timestamp_utc, timestamp_cst, description)
        # Winter/EST period - NY is UTC-5, Chicago is UTC-6
        ("2021-01-15 15:00:00", "2021-01-15 09:00:00", "Winter EST/CST"),
        ("2021-02-10 18:30:00", "2021-02-10 12:30:00", "February EST/CST"),
        ("2021-12-20 20:15:00", "2021-12-20 14:15:00", "December EST/CST"),
        
        # Summer/EDT period - NY is UTC-4, Chicago is UTC-5 (CDT)
        ("2021-06-15 16:00:00", "2021-06-15 11:00:00", "June EDT/CDT"),
        ("2021-07-04 19:45:00", "2021-07-04 14:45:00", "July EDT/CDT"),
        ("2021-08-25 14:20:00", "2021-08-25 09:20:00", "August EDT/CDT"),
        
        # Around DST transitions for 2021 (March 14, November 7)
        ("2021-03-14 06:00:00", "2021-03-14 00:00:00", "Before DST starts"),
        ("2021-03-14 07:00:00", "2021-03-14 02:00:00", "After DST starts"),
        ("2021-11-07 05:00:00", "2021-11-07 00:00:00", "Before DST ends"),
        ("2021-11-07 07:00:00", "2021-11-07 01:00:00", "After DST ends"),
        
        # 2023 test cases
        ("2023-03-12 06:30:00", "2023-03-12 00:30:00", "2023 Before DST"),
        ("2023-03-12 07:30:00", "2023-03-12 02:30:00", "2023 After DST"),
        ("2023-07-15 13:00:00", "2023-07-15 08:00:00", "2023 Summer"),
        
        # 2024 test cases
        ("2024-01-10 17:00:00", "2024-01-10 11:00:00", "2024 Winter"),
        ("2024-08-01 20:00:00", "2024-08-01 15:00:00", "2024 Summer"),
    ]
    
    # Prepare data
    timestamps_utc = [tc[0] for tc in test_cases]
    timestamps_cst = [tc[1] for tc in test_cases]
    descriptions = [tc[2] for tc in test_cases]
    
    # Create mock DataFrame with all required fields
    mock_data = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps_utc),
        'timestamp_cst': pd.to_datetime(timestamps_cst),
        'description': descriptions,
        # Add minimal required fields for the query
        'oAsk': [10.5] * len(test_cases),
        'oBid': [10.0] * len(test_cases),
        'ebid': [9.9] * len(test_cases),
        'eask': [10.6] * len(test_cases),
        'okey_xx': [100.0] * len(test_cases),
        'uPrc': [100.0] * len(test_cases),
        'okey_yr': [2023] * len(test_cases),
        'okey_mn': [12] * len(test_cases),
        'okey_dy': [31] * len(test_cases),
        'oBidM1': [9.8] * len(test_cases),
        'oAskM1': [10.3] * len(test_cases),
        'oBidM10': [9.5] * len(test_cases),
        'oAskM10': [10.0] * len(test_cases),
        'prtPrice': [10.25] * len(test_cases)
    })
    
    # Register data in DuckDB
    db_connection.register('source_view', mock_data)
    
    # Execute the query
    query = get_create_vars_1_query('source_view', 'vars_1_data')
    db_connection.execute(query)
    
    # Get results
    result = db_connection.execute("""
        SELECT 
            description,
            timestamp::VARCHAR AS timestamp_utc,
            timestamp_cst::VARCHAR AS timestamp_cst,
            timestamp_ny::VARCHAR AS timestamp_ny,
            timestamp_ny_2::VARCHAR AS timestamp_ny_2
        FROM vars_1_data
        ORDER BY timestamp
    """).fetchdf()
    
    # Print results for visual verification
    print("\n" + "="*100)
    print("Timestamp NY Equivalence Test Results")
    print("="*100)
    print(f"{'Description':<25} {'UTC':<20} {'CST':<20} {'NY (from UTC)':<20} {'NY (from CST)':<20}")
    print("-"*100)
    
    for idx, row in result.iterrows():
        print(f"{row['description']:<25} {row['timestamp_utc']:<20} {row['timestamp_cst']:<20} "
              f"{row['timestamp_ny']:<20} {row['timestamp_ny_2']:<20}")
    
    print("="*100 + "\n")
    
    # Convert to datetime for precise comparison
    result['timestamp_ny_dt'] = pd.to_datetime(result['timestamp_ny'])
    result['timestamp_ny_2_dt'] = pd.to_datetime(result['timestamp_ny_2'])
    
    # Main assertion: Both methods should produce identical results
    assert (result['timestamp_ny_dt'] == result['timestamp_ny_2_dt']).all(), \
        "timestamp_ny and timestamp_ny_2 should be exactly equal for all records"
    
    # Additional check: Verify the relationship holds (NY = CST + 1 hour)
    result['timestamp_cst_dt'] = pd.to_datetime(result['timestamp_cst'])
    expected_ny_from_cst = result['timestamp_cst_dt'] + pd.Timedelta(hours=1)
    
    assert (result['timestamp_ny_2_dt'] == expected_ny_from_cst).all(), \
        "timestamp_ny_2 should equal timestamp_cst + 1 hour"
    
    print("✓ All tests passed! timestamp_ny and timestamp_ny_2 are exactly equivalent.")


def test_timestamp_ny_equivalence_basic(db_connection):
    """
    Simple test with just a few cases to ensure basic functionality.
    """
    mock_data = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-06-01 14:00:00', '2023-01-01 15:00:00']),
        'timestamp_cst': pd.to_datetime(['2023-06-01 09:00:00', '2023-01-01 09:00:00']),
        'oAsk': [10.5, 20.5],
        'oBid': [10.0, 20.0],
        'ebid': [9.9, 19.9],
        'eask': [10.6, 20.6],
        'okey_xx': [100.0, 200.0],
        'uPrc': [100.0, 200.0],
        'okey_yr': [2023, 2024],
        'okey_mn': [12, 1],
        'okey_dy': [31, 31],
        'oBidM1': [9.8, 19.8],
        'oAskM1': [10.3, 20.3],
        'oBidM10': [9.5, 19.5],
        'oAskM10': [10.0, 20.0],
        'prtPrice': [10.25, 20.25]
    })
    
    db_connection.register('source_view', mock_data)
    
    query = get_create_vars_1_query('source_view', 'vars_1_data')
    db_connection.execute(query)
    
    result = db_connection.execute("""
        SELECT timestamp_ny, timestamp_ny_2 FROM vars_1_data
    """).fetchdf()
    
    # Both columns should exist
    assert 'timestamp_ny' in result.columns
    assert 'timestamp_ny_2' in result.columns
    
    # Both columns should be equal
    assert (result['timestamp_ny'] == result['timestamp_ny_2']).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

