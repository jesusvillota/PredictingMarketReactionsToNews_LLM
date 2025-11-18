# tests/test_create_new_vars_duckdb.py

import duckdb
import pandas as pd
import pytest
from pathlib import Path
import sys
import tempfile
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline_duckdb.create_new_vars import (
    get_create_vars_1_query,
    get_create_vars_1_step2_query,
    get_create_vars_2_query,
    get_create_vars_2_step2_query,
    get_dst_csv_path,
)

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
        f.write("2023,2023-03-12 02:00:00,2023-11-05 02:00:00\n")
        f.write("2024,2024-03-10 02:00:00,2024-11-03 02:00:00\n")
        f.flush()
        yield f.name
    os.remove(f.name)

# Monkeypatch get_dst_csv_path to use our mock file
@pytest.fixture(autouse=True)
def patch_dst_csv_path(monkeypatch, mock_dst_csv):
    monkeypatch.setattr('src.pipeline_duckdb.create_new_vars.get_dst_csv_path', lambda: mock_dst_csv)


def test_create_vars_1(db_connection):
    """Tests the get_create_vars_1_query function."""
    mock_data = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-06-01 14:00:00', '2023-01-01 15:00:00']), # One in EDT, one in EST
        'timestamp_cst': pd.to_datetime(['2023-06-01 09:00:00', '2023-01-01 09:00:00']), # CST/CDT = NY - 1 hour
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
    
    result = db_connection.execute("SELECT * FROM vars_1_data ORDER BY timestamp").fetchdf()

    assert 'spread' in result.columns
    assert 'midpointNBBO' in result.columns
    assert 'timestamp_ny' in result.columns
    assert 'expiration' in result.columns
    
    # result is ordered by timestamp, so iloc[0] is 2023-01-01 and iloc[1] is 2023-06-01
    # Check calculations
    assert pytest.approx(result['spread'].iloc[0]) == 0.5
    assert pytest.approx(result['midpointNBBO'].iloc[0]) == 20.25
    assert pytest.approx(result['spread'].iloc[1]) == 0.5
    assert pytest.approx(result['midpointNBBO'].iloc[1]) == 10.25
    assert pytest.approx(result['moneyness'].iloc[1]) == 1.0
    
    # Check timestamp conversion (EDT UTC-4, EST UTC-5)
    assert result['timestamp_ny'].iloc[0].hour == 10 # 15:00 UTC -> 10:00 EST
    assert result['timestamp_ny'].iloc[1].hour == 10 # 14:00 UTC -> 10:00 EDT

def test_create_vars_1_step2(db_connection):
    """Tests the get_create_vars_1_step2_query function."""
    mock_data = pd.DataFrame({
        'timestamp': [pd.to_datetime('2023-01-01 12:00:00')],
        'timestamp_ny': [pd.to_datetime('2023-01-01 07:00:00.123456')],
        'expiration': [pd.to_datetime('2023-01-11').date()],
        'midpriceM1': [100.5],
        'midpriceM10': [101.0],
        'prtPrice': [100.0]
    })
    db_connection.register('vars_1_data', mock_data)

    query = get_create_vars_1_step2_query('vars_1_data', 'vars_1_complete')
    db_connection.execute(query)
    
    result = db_connection.execute("SELECT * FROM vars_1_complete").fetchdf()

    assert 'dte' in result.columns
    assert 'time_to_maturity' in result.columns
    assert 'impactPriceM1' in result.columns
    
    assert result['dte'].iloc[0] == 10
    assert result['timestamp_ny_round3'].iloc[0].microsecond == 123000
    assert pytest.approx(result['impactPriceM1'].iloc[0]) == (100.5 - 100.0) / 100.0

def test_create_vars_2(db_connection):
    """Tests the get_create_vars_2_query function."""
    mock_data = pd.DataFrame({
        'okey_cp': ['Call', 'Put'],
        'prtPrice': [105.0, 95.0],
        'okey_xx': [100.0, 100.0],
        'prtSize_agg': [10, 5],
        'oAsk': [10.5, 20.5],
        'oBid': [10.0, 20.0],
        'midpointNBBO': [10.25, 20.25],
        'prtDe': [0.5, -0.5],
        'uPrc': [104.0, 96.0]
    })
    db_connection.register('grouped_data', mock_data)

    query = get_create_vars_2_query('grouped_data', 'vars_2_data')
    db_connection.execute(query)
    
    result = db_connection.execute("SELECT * FROM vars_2_data ORDER BY okey_cp").fetchdf()

    assert 'intrinsic_value' in result.columns
    assert 'trade_size_dollar' in result.columns
    assert 'leverage' in result.columns

    # Intrinsic value: Call = P - K, Put = K - P
    assert result['intrinsic_value'].iloc[0] == 5.0 # 105 - 100
    assert result['intrinsic_value'].iloc[1] == 5.0 # 100 - 95
    
    # Trade size
    assert result['trade_size_dollar'].iloc[0] == 105.0 * 10 * 100

def test_create_vars_2_step2(db_connection):
    """Tests the get_create_vars_2_step2_query function."""
    mock_data = pd.DataFrame({
        'intrinsic_value': [5.0, 2.0],
        'prtSize_agg': [10, 20]
    })
    db_connection.register('vars_2_data', mock_data)

    query = get_create_vars_2_step2_query('vars_2_data', 'vars_2_complete')
    db_connection.execute(query)
    
    result = db_connection.execute("SELECT * FROM vars_2_complete").fetchdf()

    assert 'notional_value' in result.columns
    assert result['notional_value'].iloc[0] == 5.0 * 10 * 100
    assert result['notional_value'].iloc[1] == 2.0 * 20 * 100

if __name__ == "__main__":
    pytest.main()
