"""Unit tests for stock data download functions in utils/financial.py."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from PMRTN.utils.financial import (
    FinancialUtilsError,
    _fetch_single_ticker_data,
    download_market_index,
    download_stock_returns,
    load_risk_free_rate,
)


class TestLoadRiskFreeRate:
    """Tests for load_risk_free_rate function."""

    def test_load_risk_free_rate_success(self):
        """Test successful loading of risk-free rate data."""
        # Create temporary ESTR data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = Path(f.name)
            # Write CSV with proper format
            f.write('datetime,TIME PERIOD,rf\n')
            f.write('2020-01-01,2020-01,0.5\n')
            f.write('2020-01-02,2020-01,0.6\n')
            f.write('2020-01-03,2020-01,0.55\n')

        try:
            # Load data
            rf_data = load_risk_free_rate(temp_path)

            # Verify structure
            assert isinstance(rf_data, pd.DataFrame)
            assert 'rf' in rf_data.columns
            assert len(rf_data) == 3
            assert rf_data.index.name == 'datetime'

            # Verify conversion to daily rate
            # Original: 0.5% annual -> 0.005 decimal -> daily rate
            expected_daily_rate = (1 + 0.005) ** (1 / 252) - 1
            assert rf_data['rf'].iloc[0] == pytest.approx(expected_daily_rate, abs=1e-6)

        finally:
            temp_path.unlink()

    def test_load_risk_free_rate_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FinancialUtilsError, match="not found"):
            load_risk_free_rate(Path('/nonexistent/file.csv'))

    def test_load_risk_free_rate_invalid_format(self):
        """Test error handling for invalid file format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = Path(f.name)
            # Write CSV with completely wrong structure - no datetime column
            f.write('col1,col2\n')
            f.write('a,b\n')

        try:
            with pytest.raises(FinancialUtilsError, match="Error loading"):
                load_risk_free_rate(temp_path)
        finally:
            temp_path.unlink()


class TestDownloadMarketIndex:
    """Tests for download_market_index function."""

    @patch('PMRTN.utils.financial.yf.download')
    def test_download_market_index_success(self, mock_download):
        """Test successful market index download."""
        # Create mock data
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        mock_data = pd.DataFrame({
            'Adj Close': [100, 101, 102, 103, 104],
            'Close': [100, 101, 102, 103, 104],
            'High': [101, 102, 103, 104, 105],
            'Low': [99, 100, 101, 102, 103],
            'Open': [100, 101, 102, 103, 104],
            'Volume': [1000000] * 5
        }, index=dates)
        mock_download.return_value = mock_data

        # Download market index
        start = pd.Timestamp('2020-01-01')
        end = pd.Timestamp('2020-01-05')
        result = download_market_index('^IBEX', start, end)

        # Verify structure
        assert isinstance(result, pd.DataFrame)
        assert 'r_market' in result.columns
        assert len(result) == 4  # First return is NaN and dropped
        assert result.index.name == 'datetime'

        # Verify returns calculation
        expected_return = (101 - 100) / 100
        assert result['r_market'].iloc[0] == pytest.approx(expected_return)

    @patch('PMRTN.utils.financial.yf.download')
    def test_download_market_index_no_data(self, mock_download):
        """Test error when no data is retrieved."""
        mock_download.return_value = pd.DataFrame()

        start = pd.Timestamp('2020-01-01')
        end = pd.Timestamp('2020-01-05')

        with pytest.raises(FinancialUtilsError, match="No data retrieved"):
            download_market_index('^IBEX', start, end)

    @patch('PMRTN.utils.financial.yf.download')
    def test_download_market_index_download_error(self, mock_download):
        """Test error handling when download fails."""
        mock_download.side_effect = Exception("Network error")

        start = pd.Timestamp('2020-01-01')
        end = pd.Timestamp('2020-01-05')

        with pytest.raises(FinancialUtilsError, match="Error downloading"):
            download_market_index('^IBEX', start, end)


class TestFetchSingleTickerData:
    """Tests for _fetch_single_ticker_data helper function."""

    @patch('PMRTN.utils.financial.yf.download')
    def test_fetch_single_ticker_success(self, mock_download):
        """Test successful ticker data fetch."""
        # Create mock price data
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        mock_data = pd.DataFrame({
            'Adj Close': [50, 51, 52, 53, 54],
            'Close': [50, 51, 52, 53, 54],
        }, index=dates)
        mock_download.return_value = mock_data

        # Create risk-free rate series
        rf_series = pd.Series(
            [0.0001] * 5,
            index=[d.date() for d in dates]
        )

        # Fetch ticker data
        start = pd.Timestamp('2020-01-01')
        end = pd.Timestamp('2020-01-05')
        ticker, returns, excess_returns, error = _fetch_single_ticker_data(
            'TEST.MC', start, end, rf_series
        )

        # Verify results
        assert ticker == 'TEST.MC'
        assert error is None
        assert returns is not None
        assert excess_returns is not None
        assert len(returns) == 4  # First return is NaN and dropped

    @patch('PMRTN.utils.financial.yf.download')
    def test_fetch_single_ticker_no_data(self, mock_download):
        """Test handling when no data is found for ticker."""
        mock_download.return_value = pd.DataFrame()

        rf_series = pd.Series([0.0001] * 5)

        start = pd.Timestamp('2020-01-01')
        end = pd.Timestamp('2020-01-05')
        ticker, returns, excess_returns, error = _fetch_single_ticker_data(
            'INVALID.MC', start, end, rf_series
        )

        # Verify error handling
        assert ticker == 'INVALID.MC'
        assert returns is None
        assert excess_returns is None
        assert error == "No data"

    @patch('PMRTN.utils.financial.yf.download')
    def test_fetch_single_ticker_download_error(self, mock_download):
        """Test error handling when download fails."""
        mock_download.side_effect = Exception("Connection timeout")

        rf_series = pd.Series([0.0001] * 5)

        start = pd.Timestamp('2020-01-01')
        end = pd.Timestamp('2020-01-05')
        ticker, returns, excess_returns, error = _fetch_single_ticker_data(
            'ERROR.MC', start, end, rf_series
        )

        # Verify error is captured
        assert ticker == 'ERROR.MC'
        assert returns is None
        assert excess_returns is None
        assert "Connection timeout" in error


class TestDownloadStockReturns:
    """Tests for download_stock_returns function."""

    @patch('PMRTN.utils.financial._fetch_single_ticker_data')
    def test_download_stock_returns_success(self, mock_fetch):
        """Test successful download of multiple tickers."""
        # Create mock risk-free rate data
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        rf_data = pd.DataFrame({
            'rf': [0.0001] * 5
        }, index=dates)

        # Create mock returns for two tickers
        returns_1 = pd.Series([0.01, 0.02, -0.01, 0.015], index=dates[1:])
        excess_1 = returns_1 - 0.0001

        returns_2 = pd.Series([0.005, 0.015, -0.005, 0.02], index=dates[1:])
        excess_2 = returns_2 - 0.0001

        # Setup mock to return different data for each ticker
        mock_fetch.side_effect = [
            ('TICKER1.MC', returns_1, excess_1, None),
            ('TICKER2.MC', returns_2, excess_2, None),
        ]

        # Download stock returns
        tickers = ['TICKER1.MC', 'TICKER2.MC']
        start = pd.Timestamp('2020-01-01')
        end = pd.Timestamp('2020-01-05')

        returns_df, successful, failed = download_stock_returns(
            tickers, start, end, rf_data, n_jobs=1
        )

        # Verify results
        assert isinstance(returns_df, pd.DataFrame)
        assert len(successful) == 2
        assert len(failed) == 0
        assert 'r_TICKER1.MC' in returns_df.columns
        assert 'r_TICKER1.MC_excess' in returns_df.columns
        assert 'r_TICKER2.MC' in returns_df.columns
        assert 'r_TICKER2.MC_excess' in returns_df.columns

    @patch('PMRTN.utils.financial._fetch_single_ticker_data')
    def test_download_stock_returns_mixed_success(self, mock_fetch):
        """Test download with some failures."""
        # Create mock risk-free rate data
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        rf_data = pd.DataFrame({
            'rf': [0.0001] * 5
        }, index=dates)

        # Setup mock: one success, one failure
        returns_1 = pd.Series([0.01, 0.02, -0.01, 0.015], index=dates[1:])
        excess_1 = returns_1 - 0.0001

        mock_fetch.side_effect = [
            ('GOOD.MC', returns_1, excess_1, None),
            ('BAD.MC', None, None, "Download failed"),
        ]

        # Download stock returns
        tickers = ['GOOD.MC', 'BAD.MC']
        start = pd.Timestamp('2020-01-01')
        end = pd.Timestamp('2020-01-05')

        returns_df, successful, failed = download_stock_returns(
            tickers, start, end, rf_data, n_jobs=1
        )

        # Verify results
        assert len(successful) == 1
        assert 'GOOD.MC' in successful
        assert len(failed) == 1
        assert 'BAD.MC' in failed
        assert 'r_GOOD.MC' in returns_df.columns
        assert 'r_BAD.MC' not in returns_df.columns

    def test_download_stock_returns_empty_tickers(self):
        """Test error with empty tickers list."""
        rf_data = pd.DataFrame({'rf': [0.0001]}, index=pd.date_range('2020-01-01', periods=1))
        start = pd.Timestamp('2020-01-01')
        end = pd.Timestamp('2020-01-05')

        with pytest.raises(FinancialUtilsError, match="Tickers list is empty"):
            download_stock_returns([], start, end, rf_data)

    def test_download_stock_returns_empty_rf_data(self):
        """Test error with empty risk-free rate data."""
        rf_data = pd.DataFrame()
        start = pd.Timestamp('2020-01-01')
        end = pd.Timestamp('2020-01-05')

        with pytest.raises(FinancialUtilsError, match="Risk-free rate data is empty"):
            download_stock_returns(['TEST.MC'], start, end, rf_data)


class TestIntegrationWithExistingFunctions:
    """Integration tests to ensure new functions work with existing code."""

    @patch('PMRTN.utils.financial.yf.download')
    def test_full_pipeline_mock(self, mock_download):
        """Test full pipeline from loading rf to downloading stocks."""
        # Create temporary ESTR file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = Path(f.name)
            f.write('datetime,TIME PERIOD,rf\n')
            f.write('2020-01-01,2020-01,0.5\n')
            f.write('2020-01-02,2020-01,0.6\n')
            f.write('2020-01-03,2020-01,0.55\n')

        try:
            # Load risk-free rate
            rf_data = load_risk_free_rate(temp_path)

            # Mock market index download
            market_dates = pd.date_range('2020-01-01', periods=3, freq='D')
            mock_market_data = pd.DataFrame({
                'Adj Close': [1000, 1010, 1020],
                'Close': [1000, 1010, 1020],
            }, index=market_dates)

            # Mock stock download
            mock_stock_data = pd.DataFrame({
                'Adj Close': [50, 51, 52],
                'Close': [50, 51, 52],
            }, index=market_dates)

            mock_download.side_effect = [mock_market_data, mock_stock_data]

            # Download market index
            market_data = download_market_index(
                '^IBEX',
                pd.Timestamp('2020-01-01'),
                pd.Timestamp('2020-01-03')
            )

            # Combine with risk-free rate
            combined_data = rf_data.join(market_data, how='inner')
            combined_data['r_market_excess'] = combined_data['r_market'] - combined_data['rf']

            # Verify combined data
            assert 'rf' in combined_data.columns
            assert 'r_market' in combined_data.columns
            assert 'r_market_excess' in combined_data.columns

        finally:
            temp_path.unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
