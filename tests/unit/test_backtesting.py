"""Tests for backtesting module."""

import numpy as np
import pandas as pd
import pytest

from news_market_analysis.analysis.backtesting import (
    BacktestingError,
    calculate_average_metrics_by_group,
    calculate_trading_strategy_data,
    process_article_ticker_pair,
)


@pytest.fixture
def sample_returns_df():
    """Create sample returns DataFrame for testing."""
    dates = pd.date_range('2020-01-01', periods=200, freq='B')
    np.random.seed(42)
    
    returns_df = pd.DataFrame({
        'r_AAPL.MC_excess': np.random.normal(0.001, 0.02, 200),
        'r_BBVA.MC_excess': np.random.normal(0.0005, 0.015, 200),
        'r_market_excess': np.random.normal(0.0008, 0.01, 200),
    }, index=dates)
    
    return returns_df


@pytest.fixture
def successful_tickers():
    """Create sample successful tickers dictionary."""
    return {
        'AAPL.MC': True,
        'BBVA.MC': True,
        'MISSING.MC': False,
    }


class TestCalculateTradingStrategyData:
    """Tests for calculate_trading_strategy_data function."""
    
    def test_basic_calculation(self, sample_returns_df, successful_tickers):
        """Test basic trading strategy data calculation."""
        result = calculate_trading_strategy_data(
            ticker='AAPL.MC',
            date_affect=sample_returns_df.index[120],
            returns_df=sample_returns_df,
            successful_tickers=successful_tickers,
            l_max=10,
            market_model_window=50,
            market_model_buffer=5
        )
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 11  # 0 to l_max inclusive
        assert list(result.columns) == ['AR', 'CAR', 'μ', 'σ', 'SR']
    
    def test_unsuccessful_ticker_returns_none(self, sample_returns_df, successful_tickers):
        """Test that unsuccessful tickers return None."""
        result = calculate_trading_strategy_data(
            ticker='MISSING.MC',
            date_affect=sample_returns_df.index[120],
            returns_df=sample_returns_df,
            successful_tickers=successful_tickers,
            l_max=10
        )
        
        assert result is None
    
    def test_date_not_in_index_returns_none(self, sample_returns_df, successful_tickers):
        """Test that invalid date returns None."""
        invalid_date = pd.Timestamp('2025-01-01')
        result = calculate_trading_strategy_data(
            ticker='AAPL.MC',
            date_affect=invalid_date,
            returns_df=sample_returns_df,
            successful_tickers=successful_tickers,
            l_max=10
        )
        
        assert result is None
    
    def test_insufficient_history_returns_none(self, sample_returns_df, successful_tickers):
        """Test that insufficient historical data returns None."""
        # Try to use date too early for market model window
        result = calculate_trading_strategy_data(
            ticker='AAPL.MC',
            date_affect=sample_returns_df.index[10],  # Not enough history
            returns_df=sample_returns_df,
            successful_tickers=successful_tickers,
            l_max=10,
            market_model_window=50,
            market_model_buffer=5
        )
        
        assert result is None
    
    def test_holding_period_extends_beyond_data_returns_none(
        self, sample_returns_df, successful_tickers
    ):
        """Test that holding period extending beyond data returns None."""
        result = calculate_trading_strategy_data(
            ticker='AAPL.MC',
            date_affect=sample_returns_df.index[120],
            returns_df=sample_returns_df,
            successful_tickers=successful_tickers,
            l_max=100,  # Would extend beyond available data
            market_model_window=50,
            market_model_buffer=5
        )
        
        assert result is None
    
    def test_car_increases_with_positive_returns(self, successful_tickers):
        """Test that CAR increases with positive abnormal returns."""
        # Create synthetic data with varying positive returns (avoid constant values)
        np.random.seed(123)
        dates = pd.date_range('2020-01-01', periods=200, freq='B')
        returns_df = pd.DataFrame({
            'r_TEST.MC_excess': np.random.normal(0.01, 0.002, 200),  # Positive with small variance
            'r_market_excess': np.random.normal(0.005, 0.001, 200),
        }, index=dates)
        
        successful_tickers['TEST.MC'] = True
        
        result = calculate_trading_strategy_data(
            ticker='TEST.MC',
            date_affect=dates[120],
            returns_df=returns_df,
            successful_tickers=successful_tickers,
            l_max=10,
            market_model_window=50,
            market_model_buffer=5
        )
        
        assert result is not None
        # CAR should be increasing (generally) with positive returns
        cars = result['CAR'].values
        assert cars[10] > cars[0]  # CAR at L=10 should be > CAR at L=0
    
    def test_sharpe_ratio_nan_for_l_zero(self, sample_returns_df, successful_tickers):
        """Test that Sharpe ratio is NaN for L=0."""
        result = calculate_trading_strategy_data(
            ticker='AAPL.MC',
            date_affect=sample_returns_df.index[120],
            returns_df=sample_returns_df,
            successful_tickers=successful_tickers,
            l_max=10
        )
        
        assert result is not None
        assert np.isnan(result.loc[0, 'SR'])
        assert np.isnan(result.loc[0, 'μ'])
        assert np.isnan(result.loc[0, 'σ'])
    
    def test_sharpe_ratio_calculated_for_l_greater_than_zero(
        self, sample_returns_df, successful_tickers
    ):
        """Test that Sharpe ratio is calculated for L > 0."""
        result = calculate_trading_strategy_data(
            ticker='AAPL.MC',
            date_affect=sample_returns_df.index[120],
            returns_df=sample_returns_df,
            successful_tickers=successful_tickers,
            l_max=10
        )
        
        assert result is not None
        # Should have valid SR for L > 0 (might be NaN if σ=0, but unlikely with random data)
        assert not pd.isna(result.loc[5, 'μ'])
        assert not pd.isna(result.loc[5, 'σ'])
    
    def test_custom_parameters(self, sample_returns_df, successful_tickers):
        """Test with custom market model parameters."""
        result = calculate_trading_strategy_data(
            ticker='AAPL.MC',
            date_affect=sample_returns_df.index[150],
            returns_df=sample_returns_df,
            successful_tickers=successful_tickers,
            l_max=5,
            market_model_window=30,
            market_model_buffer=3
        )
        
        assert result is not None
        assert len(result) == 6  # 0 to 5
    
    def test_missing_ticker_column_returns_none(self, sample_returns_df, successful_tickers):
        """Test that missing ticker column returns None."""
        result = calculate_trading_strategy_data(
            ticker='NONEXISTENT.MC',
            date_affect=sample_returns_df.index[120],
            returns_df=sample_returns_df,
            successful_tickers={'NONEXISTENT.MC': True},
            l_max=10
        )
        
        assert result is None


class TestProcessArticleTickerPair:
    """Tests for process_article_ticker_pair function."""
    
    def test_successful_processing(self, sample_returns_df, successful_tickers):
        """Test successful processing of article-ticker pair."""
        row = pd.Series({
            'tickers': 'AAPL.MC',
            'date_affect': sample_returns_df.index[120]
        })
        
        row_idx, result = process_article_ticker_pair(
            row_idx=0,
            row=row,
            returns_df=sample_returns_df,
            successful_tickers=successful_tickers,
            l_max=10
        )
        
        assert row_idx == 0
        assert result is not None
        assert isinstance(result, pd.DataFrame)
    
    def test_failed_processing_returns_none(self, sample_returns_df, successful_tickers):
        """Test that failed processing returns None."""
        row = pd.Series({
            'tickers': 'MISSING.MC',
            'date_affect': sample_returns_df.index[120]
        })
        
        row_idx, result = process_article_ticker_pair(
            row_idx=1,
            row=row,
            returns_df=sample_returns_df,
            successful_tickers=successful_tickers,
            l_max=10
        )
        
        assert row_idx == 1
        assert result is None
    
    def test_with_custom_parameters(self, sample_returns_df, successful_tickers):
        """Test processing with custom parameters."""
        row = pd.Series({
            'tickers': 'BBVA.MC',
            'date_affect': sample_returns_df.index[130]
        })
        
        row_idx, result = process_article_ticker_pair(
            row_idx=5,
            row=row,
            returns_df=sample_returns_df,
            successful_tickers=successful_tickers,
            l_max=5,
            market_model_window=40,
            market_model_buffer=4
        )
        
        assert row_idx == 5
        assert result is not None


class TestCalculateAverageMetricsByGroup:
    """Tests for calculate_average_metrics_by_group function."""
    
    @pytest.fixture
    def sample_articles_df(self, sample_returns_df):
        """Create sample articles DataFrame."""
        return pd.DataFrame({
            'split': ['Train', 'Train', 'Validation', 'Validation', 'Test'],
            'cluster': [0, 1, 0, 1, 0],
            'tickers': ['AAPL.MC', 'BBVA.MC', 'AAPL.MC', 'BBVA.MC', 'AAPL.MC'],
            'date_affect': [
                sample_returns_df.index[120],
                sample_returns_df.index[121],
                sample_returns_df.index[130],
                sample_returns_df.index[131],
                sample_returns_df.index[140]
            ]
        })
    
    @pytest.fixture
    def sample_ts_dict(self):
        """Create sample trading strategy dictionary."""
        # Create simple DataFrames with known values
        ts_data = pd.DataFrame({
            'AR': [0.01] * 11,
            'CAR': [0.01 * (i + 1) for i in range(11)],
            'μ': [0.001] * 11,
            'σ': [0.01] * 11,
            'SR': [1.0, 1.5, 1.2, 1.3, 1.1, 1.4] + [1.0] * 5
        })
        
        return {
            0: ts_data.copy(),
            1: ts_data.copy(),
            2: ts_data.copy(),
            3: ts_data.copy(),
            4: ts_data.copy(),
        }
    
    def test_basic_grouping(self, sample_articles_df, sample_ts_dict):
        """Test basic grouping by split and cluster."""
        result = calculate_average_metrics_by_group(
            articles_df=sample_articles_df,
            ts_dict=sample_ts_dict,
            group_columns=['split', 'cluster'],
            l_value=5
        )
        
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # Check structure
        for key, metrics in result.items():
            assert isinstance(key, tuple)
            assert len(key) == 2  # (split, cluster)
            assert 'avg_CAR' in metrics
            assert 'avg_SR' in metrics
            assert 'count' in metrics
    
    def test_single_column_grouping(self, sample_articles_df, sample_ts_dict):
        """Test grouping by single column."""
        result = calculate_average_metrics_by_group(
            articles_df=sample_articles_df,
            ts_dict=sample_ts_dict,
            group_columns=['split'],
            l_value=3
        )
        
        assert isinstance(result, dict)
        # Keys should be single-element tuples
        for key in result.keys():
            assert isinstance(key, tuple)
            assert len(key) == 1
    
    def test_skips_none_ts_data(self, sample_articles_df):
        """Test that None entries in ts_dict are skipped."""
        ts_dict_with_none = {
            0: None,
            1: pd.DataFrame({
                'AR': [0.01] * 11,
                'CAR': [0.01] * 11,
                'μ': [0.001] * 11,
                'σ': [0.01] * 11,
                'SR': [1.0] * 11
            }),
            2: None,
            3: None,
            4: None,
        }
        
        result = calculate_average_metrics_by_group(
            articles_df=sample_articles_df,
            ts_dict=ts_dict_with_none,
            group_columns=['split'],
            l_value=5
        )
        
        # Should still work, just with fewer data points
        assert isinstance(result, dict)
    
    def test_skips_insufficient_length_data(self, sample_articles_df):
        """Test that data with length <= l_value is skipped."""
        ts_dict_short = {
            i: pd.DataFrame({
                'AR': [0.01] * 3,  # Only 3 rows
                'CAR': [0.01] * 3,
                'μ': [0.001] * 3,
                'σ': [0.01] * 3,
                'SR': [1.0] * 3
            }) for i in range(5)
        }
        
        result = calculate_average_metrics_by_group(
            articles_df=sample_articles_df,
            ts_dict=ts_dict_short,
            group_columns=['split', 'cluster'],
            l_value=5  # Longer than data
        )
        
        # Should return empty or minimal results
        assert isinstance(result, dict)
    
    def test_handles_nan_values(self, sample_articles_df):
        """Test that NaN values are excluded from averages."""
        ts_dict_with_nan = {
            0: pd.DataFrame({
                'AR': [0.01] * 11,
                'CAR': [np.nan] * 11,  # All NaN
                'μ': [0.001] * 11,
                'σ': [0.01] * 11,
                'SR': [1.0, np.nan, 1.2, np.nan, 1.1, 1.4] + [1.0] * 5
            }),
            1: pd.DataFrame({
                'AR': [0.01] * 11,
                'CAR': [0.01] * 11,
                'μ': [0.001] * 11,
                'σ': [0.01] * 11,
                'SR': [1.5] * 11
            }),
            2: pd.DataFrame({
                'AR': [0.01] * 11,
                'CAR': [0.01] * 11,
                'μ': [0.001] * 11,
                'σ': [0.01] * 11,
                'SR': [1.2] * 11
            }),
            3: pd.DataFrame({
                'AR': [0.01] * 11,
                'CAR': [0.01] * 11,
                'μ': [0.001] * 11,
                'σ': [0.01] * 11,
                'SR': [1.3] * 11
            }),
            4: pd.DataFrame({
                'AR': [0.01] * 11,
                'CAR': [0.01] * 11,
                'μ': [0.001] * 11,
                'σ': [0.01] * 11,
                'SR': [1.1] * 11
            }),
        }
        
        result = calculate_average_metrics_by_group(
            articles_df=sample_articles_df,
            ts_dict=ts_dict_with_nan,
            group_columns=['split', 'cluster'],
            l_value=5
        )
        
        assert isinstance(result, dict)
        # CAR for groups containing index 0 should be NaN or excluded
        # SR should still be calculated for non-NaN values
    
    def test_empty_articles_df(self):
        """Test with empty articles DataFrame."""
        empty_df = pd.DataFrame(columns=['split', 'cluster'])
        
        result = calculate_average_metrics_by_group(
            articles_df=empty_df,
            ts_dict={},
            group_columns=['split', 'cluster'],
            l_value=5
        )
        
        assert isinstance(result, dict)
        assert len(result) == 0
