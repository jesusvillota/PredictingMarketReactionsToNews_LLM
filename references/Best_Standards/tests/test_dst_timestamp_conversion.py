"""
Test DST-aware timestamp conversion from UTC to New York time.

This test verifies that the timestamp conversion logic correctly handles:
1. EDT period (UTC-4): Between 2nd Sunday in March and 1st Sunday in November
2. EST period (UTC-5): Outside the EDT period
"""

import duckdb
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline_duckdb.create_new_vars import get_create_vars_1_query


def test_dst_timestamp_conversion():
    """Test timestamp conversion for various dates in EDT and EST periods."""
    
    # Create test data with known UTC timestamps
    test_data = [
        # Format: (timestamp_utc, expected_ny_time, description)
        # 2021 dates (Summer: March 14 02:00 -> November 7 02:00)
        ("2021-01-15 15:00:00", "2021-01-15 10:00:00", "Winter/EST: UTC-5"),
        ("2021-03-14 06:00:00", "2021-03-14 01:00:00", "Before DST starts: UTC-5"),
        ("2021-03-14 07:00:00", "2021-03-14 03:00:00", "After DST starts: UTC-4"),
        ("2021-06-15 16:00:00", "2021-06-15 12:00:00", "Summer/EDT: UTC-4"),
        ("2021-11-07 05:00:00", "2021-11-07 01:00:00", "Before DST ends: UTC-4"),
        ("2021-11-07 07:00:00", "2021-11-07 02:00:00", "After DST ends: UTC-5"),
        ("2021-12-15 18:00:00", "2021-12-15 13:00:00", "Winter/EST: UTC-5"),
        
        # 2024 dates (Summer: March 10 02:00 -> November 3 02:00)
        ("2024-02-15 14:00:00", "2024-02-15 09:00:00", "2024 Winter/EST: UTC-5"),
        ("2024-03-10 06:00:00", "2024-03-10 01:00:00", "2024 Before DST: UTC-5"),
        ("2024-03-10 07:00:00", "2024-03-10 03:00:00", "2024 After DST: UTC-4"),
        ("2024-08-01 20:00:00", "2024-08-01 16:00:00", "2024 Summer/EDT: UTC-4"),
        ("2024-11-03 05:00:00", "2024-11-03 01:00:00", "2024 Before DST ends: UTC-4"),
        ("2024-11-03 07:00:00", "2024-11-03 02:00:00", "2024 After DST ends: UTC-5"),
    ]
    
    # Create a DuckDB connection
    con = duckdb.connect()
    
    try:
        # Create test data table
        con.execute("""
            CREATE TABLE raw_data AS
            SELECT 
                timestamp::TIMESTAMP AS timestamp,
                -- Add minimal required fields for the query
                1.0 AS oAsk,
                1.0 AS oBid,
                1.0 AS ebid,
                1.0 AS eask,
                1.0 AS okey_xx,
                1.0 AS uPrc,
                2021 AS okey_yr,
                1 AS okey_mn,
                1 AS okey_dy,
                1.0 AS oBidM1,
                1.0 AS oAskM1,
                1.0 AS oBidM10,
                1.0 AS oAskM10,
                1.0 AS prtPrice
            FROM (VALUES
                ('2021-01-15 15:00:00'),
                ('2021-03-14 06:00:00'),
                ('2021-03-14 07:00:00'),
                ('2021-06-15 16:00:00'),
                ('2021-11-07 05:00:00'),
                ('2021-11-07 07:00:00'),
                ('2021-12-15 18:00:00'),
                ('2024-02-15 14:00:00'),
                ('2024-03-10 06:00:00'),
                ('2024-03-10 07:00:00'),
                ('2024-08-01 20:00:00'),
                ('2024-11-03 05:00:00'),
                ('2024-11-03 07:00:00')
            ) AS t(timestamp)
        """)
        
        # Apply the transformation
        query = get_create_vars_1_query("raw_data", "vars_1_data")
        con.execute(query)
        
        # Get results
        results = con.execute("""
            SELECT 
                timestamp::VARCHAR AS timestamp_utc,
                timestamp_ny::VARCHAR AS timestamp_ny
            FROM vars_1_data
            ORDER BY timestamp
        """).fetchall()
        
        # Verify results
        print("\n" + "="*80)
        print("DST Timestamp Conversion Test Results")
        print("="*80)
        
        all_passed = True
        for i, (timestamp_utc, expected_ny, description) in enumerate(test_data):
            actual_utc, actual_ny = results[i]
            
            # Format for comparison (normalize to comparable format)
            expected_ny_normalized = expected_ny.replace(" ", "T")
            actual_ny_normalized = actual_ny.replace(" ", "T")
            
            passed = expected_ny_normalized in actual_ny_normalized
            status = "✓ PASS" if passed else "✗ FAIL"
            
            print(f"\n{status} | {description}")
            print(f"  UTC:      {actual_utc}")
            print(f"  Expected: {expected_ny}")
            print(f"  Actual:   {actual_ny}")
            
            if not passed:
                all_passed = False
                print(f"  ERROR: Mismatch!")
        
        print("\n" + "="*80)
        if all_passed:
            print("✓ All tests passed!")
        else:
            print("✗ Some tests failed!")
        print("="*80 + "\n")
        
        return all_passed
        
    finally:
        con.close()


if __name__ == "__main__":
    success = test_dst_timestamp_conversion()
    sys.exit(0 if success else 1)

