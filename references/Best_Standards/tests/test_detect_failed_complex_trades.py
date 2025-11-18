# uv run pytest tests/test_detect_failed_complex_trades.py

import pytest
from pathlib import Path
import tempfile
import shutil
import subprocess
import sys

# Add project root to sys.path to allow importing from src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.logger import get_logger

@pytest.fixture
def temp_test_env():
    """Create a temporary directory for test data and configuration."""
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        
        # Create temporary output and complex trades daily paths
        output_path = temp_dir / "_OUTPUT_"
        complex_trades_path = temp_dir / "COMPLEX_TRADES_DAILY"
        
        output_path.mkdir()
        complex_trades_path.mkdir()
        
        # Yield paths to the test
        yield temp_dir, output_path, complex_trades_path

def run_script_in_subprocess(monkeypatch, temp_dir, output_path, complex_trades_path):
    """Helper function to run the script in a subprocess with a modified environment."""
    
    # Use monkeypatch to override config paths
    monkeypatch.setenv("OUTPUT_FOLDER", str(output_path))
    monkeypatch.setenv("COMPLEX_TRADES_DAILY_PATH", str(complex_trades_path))
    
    script_path = PROJECT_ROOT / "src" / "debugging" / "detect_failed_complex_trades.py"
    
    # Run the script as a subprocess
    result = subprocess.run(
        ["poetry", "run", "python", str(script_path)],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    
    return result

def test_all_dates_have_parquet(temp_test_env, monkeypatch):
    """Test case where all date folders have complex_trades.parquet."""
    temp_dir, output_path, complex_trades_path = temp_test_env
    
    # Create dummy date folders with parquet files
    (complex_trades_path / "2023-01-01").mkdir()
    (complex_trades_path / "2023-01-01" / "complex_trades.parquet").touch()
    (complex_trades_path / "2023-01-02").mkdir()
    (complex_trades_path / "2023-01-02" / "complex_trades.parquet").touch()
    
    # Run the script
    result = run_script_in_subprocess(monkeypatch, temp_dir, output_path, complex_trades_path)
    
    assert result.returncode == 0, f"Script failed with output:\n{result.stdout}\n{result.stderr}"
    
    # Check that the output file is empty
    output_file = output_path / "missing_complex_trades.txt"
    assert output_file.exists()
    assert output_file.read_text() == ""

def test_some_dates_missing_parquet(temp_test_env, monkeypatch):
    """Test case where some date folders are missing complex_trades.parquet."""
    temp_dir, output_path, complex_trades_path = temp_test_env
    
    # Create dummy date folders
    (complex_trades_path / "2023-01-01").mkdir()
    (complex_trades_path / "2023-01-01" / "complex_trades.parquet").touch()
    (complex_trades_path / "2023-01-02").mkdir()  # Missing parquet
    (complex_trades_path / "2023-01-03").mkdir()
    (complex_trades_path / "2023-01-03" / "complex_trades.parquet").touch()
    (complex_trades_path / "2023-01-04").mkdir()  # Missing parquet
    
    # Run the script
    result = run_script_in_subprocess(monkeypatch, temp_dir, output_path, complex_trades_path)
    
    assert result.returncode == 0, f"Script failed with output:\n{result.stdout}\n{result.stderr}"
    
    # Check the output file for missing dates
    output_file = output_path / "missing_complex_trades.txt"
    assert output_file.exists()
    content = output_file.read_text().strip().split('\n')
    assert sorted(content) == ["2023-01-02", "2023-01-04"]

def test_ignore_invalid_folder_names(temp_test_env, monkeypatch):
    """Test that folders with names not in YYYY-MM-DD format are ignored."""
    temp_dir, output_path, complex_trades_path = temp_test_env
    
    # Create valid and invalid folder names
    (complex_trades_path / "2023-01-01").mkdir()
    (complex_trades_path / "2023-01-01" / "complex_trades.parquet").touch()
    (complex_trades_path / "not-a-date").mkdir()
    (complex_trades_path / "2023-13-01").mkdir()  # Invalid month
    (complex_trades_path / "2023-01-02").mkdir() # Missing parquet
    
    # Run the script
    result = run_script_in_subprocess(monkeypatch, temp_dir, output_path, complex_trades_path)
    
    assert result.returncode == 0, f"Script failed with output:\n{result.stdout}\n{result.stderr}"
    
    # Check that only the valid missing date is reported
    output_file = output_path / "missing_complex_trades.txt"
    assert output_file.exists()
    content = output_file.read_text().strip().split('\n')
    assert content == ["2023-01-02"]

def test_no_date_folders_exist(temp_test_env, monkeypatch):
    """Test case where the COMPLEX_TRADES_DAILY directory is empty."""
    temp_dir, output_path, complex_trades_path = temp_test_env
    
    # Ensure the complex_trades_path is empty
    # It is empty by default from the fixture
    
    # Run the script
    result = run_script_in_subprocess(monkeypatch, temp_dir, output_path, complex_trades_path)
    
    assert result.returncode == 0, f"Script failed with output:\n{result.stdout}\n{result.stderr}"
    
    # Check that the output file is empty
    output_file = output_path / "missing_complex_trades.txt"
    assert output_file.exists()
    assert output_file.read_text() == ""

def test_complex_trades_path_does_not_exist(temp_test_env, monkeypatch):
    """Test case where COMPLEX_TRADES_DAILY_PATH does not exist."""
    temp_dir, output_path, complex_trades_path = temp_test_env
    
    # Remove the complex_trades_path directory
    shutil.rmtree(complex_trades_path)
    
    # Run the script
    result = run_script_in_subprocess(monkeypatch, temp_dir, output_path, complex_trades_path)
    
    # The script should exit with a non-zero status code
    assert result.returncode != 0
    assert "COMPLEX_TRADES_DAILY_PATH does not exist" in result.stdout

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
