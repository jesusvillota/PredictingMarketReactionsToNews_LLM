# uv run pytest tests/test_check_missing_prtSize_agg.py

"""
Tests for check_missing_values.py script.
"""

import pytest
import pandas as pd
import dask.dataframe as dd
from pathlib import Path
import tempfile
import shutil
from logging import Logger

from src.config.logger import get_logger
from src.debugging.check_missing_prtSize_agg import check_date_for_missing_values


@pytest.fixture
def logger():
    """Create a logger for testing."""
    return get_logger(__name__)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup after test
    if temp_path.exists():
        shutil.rmtree(temp_path)


def test_check_date_no_missing_values(temp_dir, logger):
    """Test that function returns False when there are no missing values."""
    # Create test data without missing values
    df = pd.DataFrame({
        'prtSize_agg': [100, 200, 300, 400],
        'fragment_count': [1, 2, 3, 4],
        'other_column': ['a', 'b', 'c', 'd']
    })
    
    # Save to parquet
    date_folder = temp_dir / "2021-01-29"
    date_folder.mkdir(parents=True, exist_ok=True)
    df.to_parquet(date_folder / "part.0.parquet", engine='pyarrow')
    
    # Test the function
    result = check_date_for_missing_values(date_folder, logger)
    
    assert result == False, "Should return False when no missing values"


def test_check_date_with_missing_prtSize_agg(temp_dir, logger):
    """Test that function returns True when prtSize_agg has missing values."""
    # Create test data with missing values in prtSize_agg
    df = pd.DataFrame({
        'prtSize_agg': [100, None, 300, 400],
        'fragment_count': [1, 2, 3, 4],
        'other_column': ['a', 'b', 'c', 'd']
    })
    
    # Save to parquet
    date_folder = temp_dir / "2021-02-11"
    date_folder.mkdir(parents=True, exist_ok=True)
    df.to_parquet(date_folder / "part.0.parquet", engine='pyarrow')
    
    # Test the function
    result = check_date_for_missing_values(date_folder, logger)
    
    assert result == True, "Should return True when prtSize_agg has missing values"


def test_check_date_with_missing_fragment_count(temp_dir, logger):
    """Test that function returns True when fragment_count has missing values."""
    # Create test data with missing values in fragment_count
    df = pd.DataFrame({
        'prtSize_agg': [100, 200, 300, 400],
        'fragment_count': [1, None, 3, 4],
        'other_column': ['a', 'b', 'c', 'd']
    })
    
    # Save to parquet
    date_folder = temp_dir / "2021-03-08"
    date_folder.mkdir(parents=True, exist_ok=True)
    df.to_parquet(date_folder / "part.0.parquet", engine='pyarrow')
    
    # Test the function
    result = check_date_for_missing_values(date_folder, logger)
    
    assert result == True, "Should return True when fragment_count has missing values"


def test_check_date_with_missing_both_columns(temp_dir, logger):
    """Test that function returns True when both columns have missing values."""
    # Create test data with missing values in both columns
    df = pd.DataFrame({
        'prtSize_agg': [100, None, 300, 400],
        'fragment_count': [1, 2, None, 4],
        'other_column': ['a', 'b', 'c', 'd']
    })
    
    # Save to parquet
    date_folder = temp_dir / "2021-04-15"
    date_folder.mkdir(parents=True, exist_ok=True)
    df.to_parquet(date_folder / "part.0.parquet", engine='pyarrow')
    
    # Test the function
    result = check_date_for_missing_values(date_folder, logger)
    
    assert result == True, "Should return True when both columns have missing values"


def test_check_date_multiple_partitions(temp_dir, logger):
    """Test with multiple parquet partitions."""
    # Create multiple parquet files
    date_folder = temp_dir / "2021-05-20"
    date_folder.mkdir(parents=True, exist_ok=True)
    
    # First partition - no missing values
    df1 = pd.DataFrame({
        'prtSize_agg': [100, 200, 300],
        'fragment_count': [1, 2, 3],
    })
    df1.to_parquet(date_folder / "part.0.parquet", engine='pyarrow')
    
    # Second partition - has missing values
    df2 = pd.DataFrame({
        'prtSize_agg': [400, None, 600],
        'fragment_count': [4, 5, 6],
    })
    df2.to_parquet(date_folder / "part.1.parquet", engine='pyarrow')
    
    # Test the function
    result = check_date_for_missing_values(date_folder, logger)
    
    assert result == True, "Should detect missing values across multiple partitions"


def test_check_date_missing_columns(temp_dir, logger):
    """Test that function handles missing required columns gracefully."""
    # Create test data without required columns
    df = pd.DataFrame({
        'other_column': ['a', 'b', 'c', 'd'],
        'another_column': [1, 2, 3, 4]
    })
    
    # Save to parquet
    date_folder = temp_dir / "2021-06-30"
    date_folder.mkdir(parents=True, exist_ok=True)
    df.to_parquet(date_folder / "part.0.parquet", engine='pyarrow')
    
    # Test the function
    result = check_date_for_missing_values(date_folder, logger)
    
    assert result == False, "Should return False when required columns are missing"


def test_check_date_empty_folder(temp_dir, logger):
    """Test that function handles empty folders gracefully."""
    # Create empty date folder
    date_folder = temp_dir / "2021-07-15"
    date_folder.mkdir(parents=True, exist_ok=True)
    
    # Test the function - should handle error gracefully
    result = check_date_for_missing_values(date_folder, logger)
    
    assert result == False, "Should return False when folder is empty (error case)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

