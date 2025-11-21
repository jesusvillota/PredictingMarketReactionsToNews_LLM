"""Unit tests for batch statistics computation functions."""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from PMRTN.analysis.statistics import (
    compute_statistics_for_l_values,
    compute_statistics_for_theta_values,
    StatisticsError
)


@pytest.fixture
def sample_articles_df():
    """Create a sample articles DataFrame for testing."""
    # Generate dates only in the first 30 days to ensure they're within trading_days range
    dates = pd.date_range('2020-01-01', periods=30, freq='D')
    
    df = pd.DataFrame({
        'tickers': [f'TICK{i % 10}' for i in range(50)],  # Add tickers column
        'date_affect': np.random.choice(dates, size=50),  # Only use first 30 days
        'cluster': np.random.randint(0, 5, size=50),
        'split': np.random.choice(['Train', 'Validation', 'Test'], size=50, p=[0.6, 0.2, 0.2]),
        'TR': np.random.choice([1, -1], size=50)
    })
    
    return df


@pytest.fixture
def sample_ts_dict():
    """Create a sample trading strategy dictionary for testing."""
    ts_dict = {}
    
    for i in range(50):
        # Create a trading strategy DataFrame with returns, SR, and AR columns
        # Index starts from 0 to match days_since_publication calculation
        ts_data = pd.DataFrame({
            'returns': np.random.randn(30) * 0.01,
            'SR': np.random.randn(30),
            'AR': np.random.randn(30) * 0.01,  # Add AR (Abnormal Returns) column
        }, index=range(0, 30))  # Changed from range(1, 26) to range(0, 30)
        
        ts_dict[i] = ts_data
    
    return ts_dict


@pytest.fixture
def sample_trading_days():
    """Create a sample list of trading days."""
    return list(pd.date_range('2020-01-01', periods=100, freq='D'))


class TestComputeStatisticsForLValues:
    """Test cases for compute_statistics_for_l_values function."""
    
    def test_basic_computation(self, sample_articles_df, sample_ts_dict, sample_trading_days):
        """Test basic statistics computation for multiple L values."""
        l_values = [5, 10, 15]
        
        stats = compute_statistics_for_l_values(
            articles_df=sample_articles_df,
            ts_dict=sample_ts_dict,
            trading_days=sample_trading_days,
            l_values=l_values,
            algorithms=['Greedy'],
            splits=['All'],
            verbose=False
        )
        
        assert len(stats) == 3  # 3 L values
        assert 5 in stats
        assert 10 in stats
        assert 15 in stats
        assert 'Greedy' in stats[5]
        assert 'All' in stats[5]['Greedy']
    
    def test_multiple_algorithms(self, sample_articles_df, sample_ts_dict, sample_trading_days):
        """Test computation with multiple algorithms."""
        l_values = [10]
        
        stats = compute_statistics_for_l_values(
            articles_df=sample_articles_df,
            ts_dict=sample_ts_dict,
            trading_days=sample_trading_days,
            l_values=l_values,
            algorithms=['Greedy', 'Stable'],
            splits=['All'],
            verbose=False
        )
        
        assert 'Greedy' in stats[10]
        assert 'Stable' in stats[10]
    
    def test_multiple_splits(self, sample_articles_df, sample_ts_dict, sample_trading_days):
        """Test computation with multiple splits."""
        l_values = [10]
        
        stats = compute_statistics_for_l_values(
            articles_df=sample_articles_df,
            ts_dict=sample_ts_dict,
            trading_days=sample_trading_days,
            l_values=l_values,
            algorithms=['Greedy'],
            splits=['Train', 'Test'],
            verbose=False
        )
        
        assert 'Train' in stats[10]['Greedy']
        assert 'Test' in stats[10]['Greedy']
    
    def test_statistics_keys(self, sample_articles_df, sample_ts_dict, sample_trading_days):
        """Test that all expected statistics keys are present."""
        l_values = [10]
        
        stats = compute_statistics_for_l_values(
            articles_df=sample_articles_df,
            ts_dict=sample_ts_dict,
            trading_days=sample_trading_days,
            l_values=l_values,
            algorithms=['Greedy'],
            splits=['All'],
            verbose=False
        )
        
        stat_keys = stats[10]['Greedy']['All'].keys()
        
        expected_keys = [
            'cumulative_return_gross', 'cumulative_return_net',
            'annualized_return_gross', 'annualized_return_net',
            'volatility_gross', 'volatility_net',
            'sharpe_ratio_gross', 'sharpe_ratio_net',
            'max_drawdown_gross', 'max_drawdown_net',
            'calmar_ratio_gross', 'calmar_ratio_net'
        ]
        
        for key in expected_keys:
            assert key in stat_keys
    
    def test_empty_l_values_raises_error(self, sample_articles_df, sample_ts_dict, sample_trading_days):
        """Test that empty l_values list raises error."""
        with pytest.raises(ValueError, match="l_values cannot be empty"):
            compute_statistics_for_l_values(
                articles_df=sample_articles_df,
                ts_dict=sample_ts_dict,
                trading_days=sample_trading_days,
                l_values=[],
                verbose=False
            )
    
    def test_invalid_algorithm_raises_error(self, sample_articles_df, sample_ts_dict, sample_trading_days):
        """Test that invalid algorithm name raises error."""
        with pytest.raises(ValueError, match="Invalid algorithm"):
            compute_statistics_for_l_values(
                articles_df=sample_articles_df,
                ts_dict=sample_ts_dict,
                trading_days=sample_trading_days,
                l_values=[10],
                algorithms=['InvalidAlgo'],
                verbose=False
            )
    
    def test_invalid_split_raises_error(self, sample_articles_df, sample_ts_dict, sample_trading_days):
        """Test that invalid split name raises error."""
        with pytest.raises(ValueError, match="Invalid split"):
            compute_statistics_for_l_values(
                articles_df=sample_articles_df,
                ts_dict=sample_ts_dict,
                trading_days=sample_trading_days,
                l_values=[10],
                splits=['InvalidSplit'],
                verbose=False
            )
    
    def test_missing_trading_rule_column_raises_error(self, sample_articles_df, sample_ts_dict, sample_trading_days):
        """Test that missing trading rule column raises error."""
        df_no_tr = sample_articles_df.drop(columns=['TR'])
        
        with pytest.raises(StatisticsError, match="Trading rule column"):
            compute_statistics_for_l_values(
                articles_df=df_no_tr,
                ts_dict=sample_ts_dict,
                trading_days=sample_trading_days,
                l_values=[10],
                verbose=False
            )
    
    def test_default_algorithms_and_splits(self, sample_articles_df, sample_ts_dict, sample_trading_days):
        """Test that default algorithms and splits are used when None."""
        l_values = [10]
        
        stats = compute_statistics_for_l_values(
            articles_df=sample_articles_df,
            ts_dict=sample_ts_dict,
            trading_days=sample_trading_days,
            l_values=l_values,
            algorithms=None,  # Should default to ['Greedy', 'Stable']
            splits=None,  # Should default to ['All', 'Train', 'Validation', 'Test']
            verbose=False
        )
        
        # Check default algorithms
        assert 'Greedy' in stats[10]
        assert 'Stable' in stats[10]
        
        # Check default splits
        assert 'All' in stats[10]['Greedy']
        assert 'Train' in stats[10]['Greedy']
        assert 'Validation' in stats[10]['Greedy']
        assert 'Test' in stats[10]['Greedy']


class TestComputeStatisticsForThetaValues:
    """Test cases for compute_statistics_for_theta_values function."""
    
    def test_basic_computation(self, sample_articles_df, sample_ts_dict, sample_trading_days):
        """Test basic statistics computation for multiple theta values."""
        theta_values = [0.2, 0.4]
        
        stats = compute_statistics_for_theta_values(
            articles_df=sample_articles_df,
            ts_dict=sample_ts_dict,
            trading_days=sample_trading_days,
            l_value=10,
            theta_values=theta_values,
            algorithms=['Greedy'],
            splits=['All'],
            verbose=False
        )
        
        assert len(stats) == 2  # 2 theta values
        assert 0.2 in stats
        assert 0.4 in stats
        assert 'Greedy' in stats[0.2]
        assert 'All' in stats[0.2]['Greedy']
    
    def test_multiple_algorithms(self, sample_articles_df, sample_ts_dict, sample_trading_days):
        """Test computation with multiple algorithms."""
        theta_values = [0.3]
        
        stats = compute_statistics_for_theta_values(
            articles_df=sample_articles_df,
            ts_dict=sample_ts_dict,
            trading_days=sample_trading_days,
            l_value=10,
            theta_values=theta_values,
            algorithms=['Greedy', 'Stable'],
            splits=['All'],
            verbose=False
        )
        
        assert 'Greedy' in stats[0.3]
        assert 'Stable' in stats[0.3]
    
    def test_statistics_keys(self, sample_articles_df, sample_ts_dict, sample_trading_days):
        """Test that all expected statistics keys are present."""
        theta_values = [0.2]
        
        stats = compute_statistics_for_theta_values(
            articles_df=sample_articles_df,
            ts_dict=sample_ts_dict,
            trading_days=sample_trading_days,
            l_value=10,
            theta_values=theta_values,
            algorithms=['Greedy'],
            splits=['All'],
            verbose=False
        )
        
        stat_keys = stats[0.2]['Greedy']['All'].keys()
        
        expected_keys = [
            'cumulative_return_gross', 'cumulative_return_net',
            'annualized_return_gross', 'annualized_return_net',
            'volatility_gross', 'volatility_net',
            'sharpe_ratio_gross', 'sharpe_ratio_net',
            'max_drawdown_gross', 'max_drawdown_net',
            'calmar_ratio_gross', 'calmar_ratio_net'
        ]
        
        for key in expected_keys:
            assert key in stat_keys
    
    def test_empty_theta_values_raises_error(self, sample_articles_df, sample_ts_dict, sample_trading_days):
        """Test that empty theta_values list raises error."""
        with pytest.raises(ValueError, match="theta_values cannot be empty"):
            compute_statistics_for_theta_values(
                articles_df=sample_articles_df,
                ts_dict=sample_ts_dict,
                trading_days=sample_trading_days,
                l_value=10,
                theta_values=[],
                verbose=False
            )
    
    def test_invalid_theta_range_raises_error(self, sample_articles_df, sample_ts_dict, sample_trading_days):
        """Test that theta value outside (0,1] raises error."""
        with pytest.raises(ValueError, match="theta value.*out of range"):
            compute_statistics_for_theta_values(
                articles_df=sample_articles_df,
                ts_dict=sample_ts_dict,
                trading_days=sample_trading_days,
                l_value=10,
                theta_values=[0.0],  # Invalid: must be > 0
                verbose=False
            )
        
        with pytest.raises(ValueError, match="theta value.*out of range"):
            compute_statistics_for_theta_values(
                articles_df=sample_articles_df,
                ts_dict=sample_ts_dict,
                trading_days=sample_trading_days,
                l_value=10,
                theta_values=[1.5],  # Invalid: must be <= 1
                verbose=False
            )
    
    def test_edge_case_theta_one(self, sample_articles_df, sample_ts_dict, sample_trading_days):
        """Test that theta=1.0 (select all clusters) works."""
        theta_values = [1.0]
        
        stats = compute_statistics_for_theta_values(
            articles_df=sample_articles_df,
            ts_dict=sample_ts_dict,
            trading_days=sample_trading_days,
            l_value=10,
            theta_values=theta_values,
            algorithms=['Greedy'],
            splits=['All'],
            verbose=False
        )
        
        assert 1.0 in stats
        assert 'Greedy' in stats[1.0]
    
    def test_invalid_algorithm_raises_error(self, sample_articles_df, sample_ts_dict, sample_trading_days):
        """Test that invalid algorithm name raises error."""
        with pytest.raises(ValueError, match="Invalid algorithm"):
            compute_statistics_for_theta_values(
                articles_df=sample_articles_df,
                ts_dict=sample_ts_dict,
                trading_days=sample_trading_days,
                l_value=10,
                theta_values=[0.2],
                algorithms=['BadAlgo'],
                verbose=False
            )
    
    def test_default_algorithms_and_splits(self, sample_articles_df, sample_ts_dict, sample_trading_days):
        """Test that default algorithms and splits are used when None."""
        theta_values = [0.2]
        
        stats = compute_statistics_for_theta_values(
            articles_df=sample_articles_df,
            ts_dict=sample_ts_dict,
            trading_days=sample_trading_days,
            l_value=10,
            theta_values=theta_values,
            algorithms=None,  # Should default to ['Greedy', 'Stable']
            splits=None,  # Should default to ['All', 'Train', 'Validation', 'Test']
            verbose=False
        )
        
        # Check default algorithms
        assert 'Greedy' in stats[0.2]
        assert 'Stable' in stats[0.2]
        
        # Check default splits
        assert 'All' in stats[0.2]['Greedy']
        assert 'Train' in stats[0.2]['Greedy']
        assert 'Validation' in stats[0.2]['Greedy']
        assert 'Test' in stats[0.2]['Greedy']
