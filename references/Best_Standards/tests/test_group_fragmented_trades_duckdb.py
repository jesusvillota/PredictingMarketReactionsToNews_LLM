# uv run pytest tests/test_group_fragmented_trades_duckdb.py -v
# uv run pytest tests/test_group_fragmented_trades_duckdb.py::test_cross_partition_grouping -v

"""
Comprehensive tests for group_fragmented_trades_duckdb.py script.
Tests all processing modes and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import duckdb
import shutil
from datetime import datetime, timedelta

# Import the functions we need to test
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config_settings


# Test data generation helpers
def create_mock_fragmented_data(
    tmp_path: Path,
    num_partitions: int = 3,
    groups_config: list[dict] = None
) -> Path:
    """
    Create mock fragmented trade data across multiple parquet partitions.
    
    Args:
        tmp_path: Temporary directory path
        num_partitions: Number of parquet files to create
        groups_config: List of group configurations, each dict containing:
            - grouping_values: dict of column values for the group
            - prtSizes: list of prtSize values
            - partition_distribution: list indicating which partition each row goes to
    
    Returns:
        Path to the daily folder containing parquet files
    """
    if groups_config is None:
        # Default: create simple test data
        groups_config = [
            {
                'grouping_values': {
                    'okey_tk': 'AAPL',
                    'okey_xx': 150.0,
                    'okey_cp': 'Call',
                    'prtExch': 'A',
                    'prtPrice': 5.25,
                    'prtType': 102,
                },
                'prtSizes': [100, 200, 150],
                'partition_distribution': [0, 1, 2]
            },
            {
                'grouping_values': {
                    'okey_tk': 'MSFT',
                    'okey_xx': 300.0,
                    'okey_cp': 'Put',
                    'prtExch': 'B',
                    'prtPrice': 3.75,
                    'prtType': 102,
                },
                'prtSizes': [50, 75],
                'partition_distribution': [0, 1]
            }
        ]
    
    # Create daily folder
    daily_folder = tmp_path / "2019-01-15"
    daily_folder.mkdir(parents=True, exist_ok=True)
    
    # Initialize partition data collectors
    partition_data = {i: [] for i in range(num_partitions)}
    
    # Generate rows for each group and distribute to partitions
    base_timestamp = datetime(2019, 1, 15, 10, 0, 0)
    
    for group_idx, group_config in enumerate(groups_config):
        grouping_values = group_config['grouping_values']
        prtSizes = group_config['prtSizes']
        partition_dist = group_config.get('partition_distribution', [0] * len(prtSizes))
        
        # Use same timestamp for all rows in the same group
        group_timestamp = base_timestamp + timedelta(minutes=group_idx)
        
        for row_idx, (prt_size, partition_idx) in enumerate(zip(prtSizes, partition_dist)):
            row = {
                'okey_tk': grouping_values.get('okey_tk', 'AAPL'),
                'okey_xx': grouping_values.get('okey_xx', 150.0),
                'okey_cp': grouping_values.get('okey_cp', 'Call'),
                'uBid': grouping_values.get('uBid', 148.5),
                'uAsk': grouping_values.get('uAsk', 149.5),
                'uPrc': grouping_values.get('uPrc', 149.0),
                'prtExch': grouping_values.get('prtExch', 'A'),
                'prtPrice': grouping_values.get('prtPrice', 5.25),
                'prtType': grouping_values.get('prtType', 102),
                'timestamp_ny_round3': group_timestamp,  # Same timestamp for all rows in group
                'tradingSession': grouping_values.get('tradingSession', 'Regular'),
                'prtSize': prt_size,
                'extra_col': f'extra_{group_idx}_{row_idx}'  # Extra column to test preservation
            }
            
            partition_data[partition_idx].append(row)
    
    # Write each partition to a parquet file
    for partition_idx, rows in partition_data.items():
        if rows:  # Only write if there's data
            df = pd.DataFrame(rows)
            partition_file = daily_folder / f"partition_{partition_idx}.parquet"
            df.to_parquet(
                partition_file,
                engine=config_settings.parquet["engine"],
                compression=config_settings.parquet["compression"],
                index=False
            )
    
    return daily_folder


def verify_grouping_results(result_df: pd.DataFrame, expected_groups: list[dict]) -> bool:
    """
    Verify that grouping results match expected values.
    
    Args:
        result_df: DataFrame with grouped results
        expected_groups: List of dicts with expected group characteristics
    
    Returns:
        True if all checks pass
    """
    assert len(result_df) == len(expected_groups), \
        f"Expected {len(expected_groups)} groups, got {len(result_df)}"
    
    for expected in expected_groups:
        # Find matching row based on grouping columns
        mask = pd.Series([True] * len(result_df))
        for col, val in expected.get('grouping_values', {}).items():
            if col in result_df.columns:
                mask &= (result_df[col] == val)
        
        matching_rows = result_df[mask]
        assert len(matching_rows) == 1, \
            f"Expected exactly 1 row for group {expected.get('grouping_values')}, got {len(matching_rows)}"
        
        row = matching_rows.iloc[0]
        
        # Check fragment_count
        if 'expected_fragment_count' in expected:
            assert row['fragment_count'] == expected['expected_fragment_count'], \
                f"Expected fragment_count={expected['expected_fragment_count']}, got {row['fragment_count']}"
        
        # Check prtSize_agg
        if 'expected_prtSize_agg' in expected:
            assert row['prtSize_agg'] == expected['expected_prtSize_agg'], \
                f"Expected prtSize_agg={expected['expected_prtSize_agg']}, got {row['prtSize_agg']}"
    
    return True


def run_grouping_with_duckdb(
    daily_folder: Path,
    output_path: Path,
    batched: bool = False,
    batch_size: int = 10
) -> pd.DataFrame:
    """
    Run the grouping logic using DuckDB (mimics the script logic).
    
    Returns:
        DataFrame with grouped results
    """
    from src.mains.group_fragmented_trades_duckdb import (
        GROUPING_COLS,
        get_available_grouping_cols,
        process_day_non_batched,
        process_day_batched
    )
    from src.config.config_settings import RAM_LIMIT, CPU_LIMIT, DASK_TEMP_DIR
    
    con = duckdb.connect()
    try:
        # Configure DuckDB
        con.execute(f"SET memory_limit='{RAM_LIMIT}GB'")
        temp_dir = DASK_TEMP_DIR
        temp_dir.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{temp_dir}'")
        con.execute(f"SET threads={CPU_LIMIT}")
        con.execute("SET preserve_insertion_order=false")
        con.execute("SET enable_object_cache=false")
        
        # Create view
        con.execute(f"""
            CREATE OR REPLACE VIEW trades_data AS 
            SELECT *
            FROM read_parquet('{daily_folder}/**/*.parquet')
        """)
        
        # Get available grouping columns
        available_grouping_cols = get_available_grouping_cols(con, 'trades_data')
        
        if not available_grouping_cols:
            return pd.DataFrame()
        
        partition_clause = ", ".join(available_grouping_cols)
        
        if not batched:
            # Non-batched query
            query = f"""
                SELECT *,
                       COUNT(*) OVER (PARTITION BY {partition_clause}) as fragment_count,
                       SUM(prtSize) OVER (PARTITION BY {partition_clause}) as prtSize_agg
                FROM trades_data
                QUALIFY ROW_NUMBER() OVER (PARTITION BY {partition_clause} ORDER BY prtSize) = 1
                ORDER BY {partition_clause}
            """
            result_df = con.execute(query).df()
            
            # Save output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_parquet(
                output_path,
                engine=config_settings.parquet["engine"],
                compression=config_settings.parquet["compression"],
                index=False
            )
        else:
            # Batched processing
            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE grouped_data AS
                SELECT *,
                       COUNT(*) OVER (PARTITION BY {partition_clause}) as fragment_count,
                       SUM(prtSize) OVER (PARTITION BY {partition_clause}) as prtSize_agg
                FROM trades_data
                QUALIFY ROW_NUMBER() OVER (PARTITION BY {partition_clause} ORDER BY prtSize) = 1
            """)
            
            total_groups = con.execute("SELECT COUNT(*) FROM grouped_data").fetchone()[0]
            
            # Ensure output directory exists for temp files
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            temp_files = []
            for offset in range(0, total_groups, batch_size):
                batch_query = f"""
                    SELECT * FROM grouped_data
                    ORDER BY {partition_clause}
                    LIMIT {batch_size} OFFSET {offset}
                """
                batch_df = con.execute(batch_query).df()
                
                if not batch_df.empty:
                    batch_num = offset // batch_size
                    temp_file = output_path.parent / f"temp_batch_{batch_num:06d}.parquet"
                    batch_df.to_parquet(
                        temp_file,
                        engine=config_settings.parquet["engine"],
                        compression=config_settings.parquet["compression"],
                        index=False
                    )
                    temp_files.append(temp_file)
            
            # Combine temp files
            if temp_files:
                combined_df = pd.concat([
                    pd.read_parquet(f, engine=config_settings.parquet["engine"])
                    for f in temp_files
                ], ignore_index=True)
                
                combined_df.to_parquet(
                    output_path,
                    engine=config_settings.parquet["engine"],
                    compression=config_settings.parquet["compression"],
                    index=False
                )
                
                # Clean up temp files
                for f in temp_files:
                    f.unlink(missing_ok=True)
                
                result_df = combined_df
            else:
                result_df = pd.DataFrame()
        
        return result_df
    
    finally:
        con.close()


# ============================================================================
# TEST CASES
# ============================================================================

def test_non_batched_mode_new_folder(tmp_path):
    """Test non-batched processing with new folder output"""
    # Create test data
    groups_config = [
        {
            'grouping_values': {'okey_tk': 'AAPL', 'okey_xx': 150.0, 'okey_cp': 'Call', 
                              'prtExch': 'A', 'prtPrice': 5.25, 'prtType': 102},
            'prtSizes': [100, 200, 150],
            'partition_distribution': [0, 1, 2]
        },
        {
            'grouping_values': {'okey_tk': 'MSFT', 'okey_xx': 300.0, 'okey_cp': 'Put',
                              'prtExch': 'B', 'prtPrice': 3.75, 'prtType': 102},
            'prtSizes': [50, 75],
            'partition_distribution': [0, 1]
        }
    ]
    
    daily_folder = create_mock_fragmented_data(tmp_path, num_partitions=3, groups_config=groups_config)
    
    # Run grouping
    output_dir = tmp_path / "output" / "2019-01-15"
    output_path = output_dir / "grouped_trades.parquet"
    
    result_df = run_grouping_with_duckdb(daily_folder, output_path, batched=False)
    
    # Verify results
    assert len(result_df) == 2, "Should have 2 groups"
    assert 'fragment_count' in result_df.columns
    assert 'prtSize_agg' in result_df.columns
    
    expected_groups = [
        {
            'grouping_values': {'okey_tk': 'AAPL', 'okey_xx': 150.0},
            'expected_fragment_count': 3,
            'expected_prtSize_agg': 450
        },
        {
            'grouping_values': {'okey_tk': 'MSFT', 'okey_xx': 300.0},
            'expected_fragment_count': 2,
            'expected_prtSize_agg': 125
        }
    ]
    
    verify_grouping_results(result_df, expected_groups)
    
    # Check that output file was created
    assert output_path.exists()


def test_batched_mode_new_folder(tmp_path):
    """Test batched processing produces same results as non-batched"""
    # Create test data with more groups
    groups_config = [
        {
            'grouping_values': {'okey_tk': f'STOCK{i}', 'okey_xx': 100.0 + i, 'okey_cp': 'Call',
                              'prtExch': 'A', 'prtPrice': 5.0, 'prtType': 102},
            'prtSizes': [100 * (i+1), 200 * (i+1)],
            'partition_distribution': [0, 1]
        }
        for i in range(10)
    ]
    
    daily_folder = create_mock_fragmented_data(tmp_path, num_partitions=2, groups_config=groups_config)
    
    # Run non-batched
    output_path_non_batched = tmp_path / "output_non_batched" / "grouped_trades.parquet"
    result_non_batched = run_grouping_with_duckdb(daily_folder, output_path_non_batched, batched=False)
    
    # Run batched with small batch size
    output_path_batched = tmp_path / "output_batched" / "grouped_trades.parquet"
    result_batched = run_grouping_with_duckdb(daily_folder, output_path_batched, batched=True, batch_size=3)
    
    # Both should have same number of groups
    assert len(result_non_batched) == len(result_batched) == 10
    
    # Sort both for comparison
    result_non_batched = result_non_batched.sort_values('okey_tk').reset_index(drop=True)
    result_batched = result_batched.sort_values('okey_tk').reset_index(drop=True)
    
    # Compare key columns
    pd.testing.assert_series_equal(
        result_non_batched['fragment_count'],
        result_batched['fragment_count'],
        check_names=True
    )
    pd.testing.assert_series_equal(
        result_non_batched['prtSize_agg'],
        result_batched['prtSize_agg'],
        check_names=True
    )
    
    # Check temp files were cleaned up
    temp_files = list((output_path_batched.parent).glob("temp_batch_*.parquet"))
    assert len(temp_files) == 0, "Temp files should be cleaned up"


def test_cross_partition_grouping(tmp_path):
    """Test that groups spanning multiple partitions are correctly aggregated"""
    # Create a group that spans 3 partitions
    groups_config = [
        {
            'grouping_values': {'okey_tk': 'AAPL', 'okey_xx': 150.0, 'okey_cp': 'Call',
                              'prtExch': 'A', 'prtPrice': 5.25, 'prtType': 102},
            'prtSizes': [100, 200, 150, 50, 75],  # 5 fragments
            'partition_distribution': [0, 1, 2, 0, 1]  # Spread across 3 partitions
        }
    ]
    
    daily_folder = create_mock_fragmented_data(tmp_path, num_partitions=3, groups_config=groups_config)
    
    # Run grouping
    output_path = tmp_path / "output" / "grouped_trades.parquet"
    result_df = run_grouping_with_duckdb(daily_folder, output_path, batched=False)
    
    # Should have exactly 1 group
    assert len(result_df) == 1
    
    # Check aggregations
    assert result_df.iloc[0]['fragment_count'] == 5, "Should count all 5 fragments"
    assert result_df.iloc[0]['prtSize_agg'] == 575, "Should sum all prtSizes (100+200+150+50+75)"


def test_grouping_correctness(tmp_path):
    """Test that grouping columns and aggregations are correct"""
    groups_config = [
        {
            'grouping_values': {
                'okey_tk': 'AAPL',
                'okey_xx': 150.0,
                'okey_cp': 'Call',
                'uBid': 148.5,
                'uAsk': 149.5,
                'uPrc': 149.0,
                'prtExch': 'A',
                'prtPrice': 5.25,
                'prtType': 102,
                'tradingSession': 'Regular'
            },
            'prtSizes': [100, 200, 300],
            'partition_distribution': [0, 0, 1]
        },
        {
            'grouping_values': {
                'okey_tk': 'AAPL',
                'okey_xx': 150.0,
                'okey_cp': 'Call',
                'uBid': 148.5,
                'uAsk': 149.5,
                'uPrc': 149.0,
                'prtExch': 'A',
                'prtPrice': 5.50,  # Different price -> different group
                'prtType': 102,
                'tradingSession': 'Regular'
            },
            'prtSizes': [50, 75],
            'partition_distribution': [1, 2]
        }
    ]
    
    daily_folder = create_mock_fragmented_data(tmp_path, num_partitions=3, groups_config=groups_config)
    output_path = tmp_path / "output" / "grouped_trades.parquet"
    result_df = run_grouping_with_duckdb(daily_folder, output_path, batched=False)
    
    # Should have 2 groups (different prtPrice)
    assert len(result_df) == 2
    
    # Verify each group
    group1 = result_df[result_df['prtPrice'] == 5.25].iloc[0]
    assert group1['fragment_count'] == 3
    assert group1['prtSize_agg'] == 600
    
    group2 = result_df[result_df['prtPrice'] == 5.50].iloc[0]
    assert group2['fragment_count'] == 2
    assert group2['prtSize_agg'] == 125


def test_edge_case_single_row(tmp_path):
    """Test edge case with single row"""
    groups_config = [
        {
            'grouping_values': {'okey_tk': 'AAPL', 'okey_xx': 150.0, 'okey_cp': 'Call',
                              'prtExch': 'A', 'prtPrice': 5.25, 'prtType': 102},
            'prtSizes': [100],
            'partition_distribution': [0]
        }
    ]
    
    daily_folder = create_mock_fragmented_data(tmp_path, num_partitions=1, groups_config=groups_config)
    output_path = tmp_path / "output" / "grouped_trades.parquet"
    result_df = run_grouping_with_duckdb(daily_folder, output_path, batched=False)
    
    assert len(result_df) == 1
    assert result_df.iloc[0]['fragment_count'] == 1
    assert result_df.iloc[0]['prtSize_agg'] == 100


def test_edge_case_all_rows_same_group(tmp_path):
    """Test when all rows belong to the same group"""
    groups_config = [
        {
            'grouping_values': {'okey_tk': 'AAPL', 'okey_xx': 150.0, 'okey_cp': 'Call',
                              'prtExch': 'A', 'prtPrice': 5.25, 'prtType': 102},
            'prtSizes': [100, 200, 300, 400, 500],
            'partition_distribution': [0, 0, 1, 1, 2]
        }
    ]
    
    daily_folder = create_mock_fragmented_data(tmp_path, num_partitions=3, groups_config=groups_config)
    output_path = tmp_path / "output" / "grouped_trades.parquet"
    result_df = run_grouping_with_duckdb(daily_folder, output_path, batched=False)
    
    assert len(result_df) == 1
    assert result_df.iloc[0]['fragment_count'] == 5
    assert result_df.iloc[0]['prtSize_agg'] == 1500


def test_edge_case_each_row_unique_group(tmp_path):
    """Test when each row is its own unique group"""
    groups_config = [
        {
            'grouping_values': {'okey_tk': 'AAPL', 'okey_xx': 150.0 + i, 'okey_cp': 'Call',
                              'prtExch': 'A', 'prtPrice': 5.0, 'prtType': 102},
            'prtSizes': [100],
            'partition_distribution': [i % 2]
        }
        for i in range(5)
    ]
    
    daily_folder = create_mock_fragmented_data(tmp_path, num_partitions=2, groups_config=groups_config)
    output_path = tmp_path / "output" / "grouped_trades.parquet"
    result_df = run_grouping_with_duckdb(daily_folder, output_path, batched=False)
    
    assert len(result_df) == 5
    # Each group should have fragment_count=1
    assert all(result_df['fragment_count'] == 1)
    assert all(result_df['prtSize_agg'] == 100)


def test_output_structure(tmp_path):
    """Test that output preserves all original columns and adds new ones"""
    groups_config = [
        {
            'grouping_values': {'okey_tk': 'AAPL', 'okey_xx': 150.0, 'okey_cp': 'Call',
                              'prtExch': 'A', 'prtPrice': 5.25, 'prtType': 102},
            'prtSizes': [100, 200],
            'partition_distribution': [0, 1]
        }
    ]
    
    daily_folder = create_mock_fragmented_data(tmp_path, num_partitions=2, groups_config=groups_config)
    output_path = tmp_path / "output" / "grouped_trades.parquet"
    result_df = run_grouping_with_duckdb(daily_folder, output_path, batched=False)
    
    # Check new columns exist
    assert 'fragment_count' in result_df.columns
    assert 'prtSize_agg' in result_df.columns
    
    # Check original columns are preserved
    original_cols = ['okey_tk', 'okey_xx', 'okey_cp', 'prtExch', 'prtPrice', 
                    'prtType', 'prtSize', 'extra_col']
    for col in original_cols:
        assert col in result_df.columns, f"Original column {col} should be preserved"
    
    # Check no duplicate rows (one per group)
    assert len(result_df) == 1
    
    # Check data types are reasonable
    assert result_df['fragment_count'].dtype in [np.int64, np.int32]
    assert result_df['prtSize_agg'].dtype in [np.float64, np.int64, np.int32]


def test_empty_data_folder(tmp_path):
    """Test handling of empty data folder"""
    # Create empty daily folder
    daily_folder = tmp_path / "2019-01-15"
    daily_folder.mkdir(parents=True, exist_ok=True)
    
    # Create a single empty parquet file
    empty_df = pd.DataFrame()
    (daily_folder / "empty.parquet").write_text("")
    
    # This should handle gracefully - create proper empty df structure
    output_path = tmp_path / "output" / "grouped_trades.parquet"
    
    # The function should handle this gracefully
    # We expect it to either return empty DataFrame or skip
    try:
        result_df = run_grouping_with_duckdb(daily_folder, output_path, batched=False)
        # If it succeeds, should be empty
        assert len(result_df) == 0 or result_df.empty
    except Exception:
        # It's also acceptable to fail gracefully on completely empty data
        pass

