"""Tests for portfolio analysis module."""

import numpy as np
import pandas as pd
import pytest

from PMRTN.analysis.portfolio import (
    PortfolioError,
    calculate_portfolio_returns,
    calculate_portfolio_statistics,
    calculate_trading_intensity_statistics,
    initialize_portfolio,
)


class TestPortfolioError:
    """Tests for PortfolioError exception."""

    def test_portfolio_error_can_be_raised(self) -> None:
        """Test exception can be raised and caught."""
        with pytest.raises(PortfolioError):
            raise PortfolioError("Test error")

    def test_portfolio_error_message_preserved(self) -> None:
        """Test exception message is preserved."""
        error_msg = "Portfolio operation failed"
        try:
            raise PortfolioError(error_msg)
        except PortfolioError as e:
            assert str(e) == error_msg


class TestInitializePortfolio:
    """Tests for initialize_portfolio function."""

    @pytest.fixture
    def sample_articles_df(self) -> pd.DataFrame:
        """Create sample articles DataFrame."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        return pd.DataFrame({
            'split': ['Train'] * 3 + ['Validation'] * 3 + ['Test'] * 4,
            'date_affect': dates[:10],
            'tickers': ['TEF.MC'] * 10,
            'cluster': [0, 1, 2] * 3 + [0]
        })

    @pytest.fixture
    def sample_trading_days(self) -> list:
        """Create sample trading days list."""
        return pd.date_range('2023-01-01', periods=20, freq='D').tolist()

    def test_initialize_portfolio_happy_path(
        self, sample_articles_df: pd.DataFrame, sample_trading_days: list
    ) -> None:
        """Test portfolio initialization with valid inputs."""
        r_p_dict, trading_days_dict = initialize_portfolio(
            sample_articles_df, sample_trading_days, l_value=5
        )

        assert isinstance(r_p_dict, dict)
        assert isinstance(trading_days_dict, dict)
        assert 'All' in r_p_dict
        assert 'Train' in r_p_dict
        assert 'Validation' in r_p_dict
        assert 'Test' in r_p_dict

        # Check that DataFrames are created
        for split in ['All', 'Train', 'Validation', 'Test']:
            assert isinstance(r_p_dict[split], pd.DataFrame)
            assert 'returns' in r_p_dict[split].columns

    def test_initialize_portfolio_extends_timeline(
        self, sample_articles_df: pd.DataFrame, sample_trading_days: list
    ) -> None:
        """Test that timeline is extended by L days."""
        r_p_dict, trading_days_dict = initialize_portfolio(
            sample_articles_df, sample_trading_days, l_value=5
        )

        # Test split should extend beyond last article
        test_days = trading_days_dict['Test']
        last_article_date = sample_articles_df['date_affect'].max()
        assert test_days[-1] >= last_article_date

    def test_initialize_portfolio_empty_articles_df(
        self, sample_trading_days: list
    ) -> None:
        """Test initialization with empty articles DataFrame."""
        empty_df = pd.DataFrame(columns=['split', 'date_affect'])
        r_p_dict, trading_days_dict = initialize_portfolio(
            empty_df, sample_trading_days, l_value=5
        )

        # Should still create DataFrames but they may be empty
        assert 'All' in r_p_dict
        assert isinstance(r_p_dict['All'], pd.DataFrame)

    def test_initialize_portfolio_no_articles_in_split(
        self, sample_trading_days: list
    ) -> None:
        """Test initialization when no articles in a split."""
        df = pd.DataFrame({
            'split': ['Train'] * 5,
            'date_affect': sample_trading_days[:5],
            'tickers': ['TEF.MC'] * 5
        })

        r_p_dict, trading_days_dict = initialize_portfolio(df, sample_trading_days, l_value=5)

        # Validation and Test splits should still exist but may be empty
        assert 'Validation' in r_p_dict
        assert 'Test' in r_p_dict

    def test_initialize_portfolio_l_value_zero(
        self, sample_articles_df: pd.DataFrame, sample_trading_days: list
    ) -> None:
        """Test initialization with l_value = 0."""
        r_p_dict, trading_days_dict = initialize_portfolio(
            sample_articles_df, sample_trading_days, l_value=0
        )

        assert 'All' in r_p_dict
        # Timeline should not extend beyond last article
        test_days = trading_days_dict['Test']
        last_article_date = sample_articles_df['date_affect'].max()
        assert test_days[-1] <= last_article_date

    def test_initialize_portfolio_large_l_value(
        self, sample_articles_df: pd.DataFrame, sample_trading_days: list
    ) -> None:
        """Test initialization with large l_value."""
        r_p_dict, trading_days_dict = initialize_portfolio(
            sample_articles_df, sample_trading_days, l_value=100
        )

        # Should not extend beyond available trading days
        test_days = trading_days_dict['Test']
        assert test_days[-1] <= sample_trading_days[-1]

    def test_initialize_portfolio_trading_days_extend_beyond_data(
        self, sample_articles_df: pd.DataFrame
    ) -> None:
        """Test when trading days extend beyond available data."""
        limited_trading_days = pd.date_range('2023-01-01', periods=5, freq='D').tolist()
        
        r_p_dict, trading_days_dict = initialize_portfolio(
            sample_articles_df, limited_trading_days, l_value=10
        )

        # Should handle gracefully
        assert 'All' in r_p_dict


class TestCalculatePortfolioReturns:
    """Tests for calculate_portfolio_returns function."""

    @pytest.fixture
    def sample_articles_df(self) -> pd.DataFrame:
        """Create sample articles DataFrame with trading rules."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        return pd.DataFrame({
            'split': ['Train'] * 2 + ['Validation'] * 2 + ['Test'],
            'date_affect': dates,
            'tickers': ['TEF.MC', 'SAN.MC', 'TEF.MC', 'SAN.MC', 'TEF.MC'],
            'cluster': [0, 1, 0, 1, 0],
            'TR': [1, -1, 1, -1, 1]
        })

    @pytest.fixture
    def sample_trading_days(self) -> list:
        """Create sample trading days."""
        return pd.date_range('2023-01-01', periods=10, freq='D').tolist()

    @pytest.fixture
    def sample_ts_dict(self) -> dict:
        """Create sample trading strategy dictionary."""
        ts_dict = {}
        for i in range(5):
            df = pd.DataFrame({
                'AR': np.random.randn(5) * 0.01,
                'SR': np.random.randn(5) * 0.5
            })
            ts_dict[i] = df
        return ts_dict

    def test_calculate_portfolio_returns_happy_path(
        self,
        sample_articles_df: pd.DataFrame,
        sample_trading_days: list,
        sample_ts_dict: dict
    ) -> None:
        """Test portfolio returns calculation with valid inputs."""
        result = calculate_portfolio_returns(
            sample_articles_df,
            sample_trading_days,
            l_value=3,
            ts_dict=sample_ts_dict,
            trading_rule_col='TR',
            trading_cost_bps=10.0,
            verbose=False
        )

        assert isinstance(result, dict)
        assert 'All' in result
        assert 'Train' in result
        assert 'Validation' in result
        assert 'Test' in result
        assert 'trading_signal_evolution' in result
        assert 'turnover' in result
        assert 'turnover_stats' in result

        # Check return columns
        for split in ['All', 'Train', 'Validation', 'Test']:
            assert 'gross_returns' in result[split].columns
            assert 'net_returns' in result[split].columns

    def test_calculate_portfolio_returns_trading_cost_zero(
        self,
        sample_articles_df: pd.DataFrame,
        sample_trading_days: list,
        sample_ts_dict: dict
    ) -> None:
        """Test with trading_cost_bps = 0 (no costs)."""
        result = calculate_portfolio_returns(
            sample_articles_df,
            sample_trading_days,
            l_value=3,
            ts_dict=sample_ts_dict,
            trading_cost_bps=0.0
        )

        # Net returns should equal gross returns when costs are zero
        all_returns = result['All']
        np.testing.assert_array_almost_equal(
            all_returns['gross_returns'].values,
            all_returns['net_returns'].values
        )

    def test_calculate_portfolio_returns_trading_cost_high(
        self,
        sample_articles_df: pd.DataFrame,
        sample_trading_days: list,
        sample_ts_dict: dict
    ) -> None:
        """Test with trading_cost_bps = 100 (1% costs)."""
        result = calculate_portfolio_returns(
            sample_articles_df,
            sample_trading_days,
            l_value=3,
            ts_dict=sample_ts_dict,
            trading_cost_bps=100.0
        )

        # Net returns should be less than gross returns
        all_returns = result['All']
        assert (all_returns['net_returns'] <= all_returns['gross_returns']).all()

    def test_calculate_portfolio_returns_l_value_one(
        self,
        sample_articles_df: pd.DataFrame,
        sample_trading_days: list,
        sample_ts_dict: dict
    ) -> None:
        """Test with l_value = 1 (minimum holding period)."""
        result = calculate_portfolio_returns(
            sample_articles_df,
            sample_trading_days,
            l_value=1,
            ts_dict=sample_ts_dict
        )

        assert 'All' in result

    def test_calculate_portfolio_returns_no_trading_signals(
        self,
        sample_trading_days: list,
        sample_ts_dict: dict
    ) -> None:
        """Test when there are no trading signals."""
        df = pd.DataFrame({
            'split': ['Train'] * 3,
            'date_affect': sample_trading_days[:3],
            'tickers': ['TEF.MC'] * 3,
            'cluster': [0, 1, 2],
            'TR': [0, 0, 0]  # No trading signals
        })

        result = calculate_portfolio_returns(
            df, sample_trading_days, l_value=3, ts_dict=sample_ts_dict
        )

        # Returns should be zero
        all_returns = result['All']
        assert (all_returns['gross_returns'] == 0.0).all()

    def test_calculate_portfolio_returns_all_long_positions(
        self,
        sample_trading_days: list,
        sample_ts_dict: dict
    ) -> None:
        """Test with all positions long."""
        df = pd.DataFrame({
            'split': ['Train'] * 3,
            'date_affect': sample_trading_days[:3],
            'tickers': ['TEF.MC'] * 3,
            'cluster': [0, 1, 2],
            'TR': [1, 1, 1]  # All long
        })

        result = calculate_portfolio_returns(
            df, sample_trading_days, l_value=3, ts_dict=sample_ts_dict
        )

        assert 'All' in result

    def test_calculate_portfolio_returns_all_short_positions(
        self,
        sample_trading_days: list,
        sample_ts_dict: dict
    ) -> None:
        """Test with all positions short."""
        df = pd.DataFrame({
            'split': ['Train'] * 3,
            'date_affect': sample_trading_days[:3],
            'tickers': ['TEF.MC'] * 3,
            'cluster': [0, 1, 2],
            'TR': [-1, -1, -1]  # All short
        })

        result = calculate_portfolio_returns(
            df, sample_trading_days, l_value=3, ts_dict=sample_ts_dict
        )

        assert 'All' in result

    def test_calculate_portfolio_returns_mixed_positions(
        self,
        sample_trading_days: list,
        sample_ts_dict: dict
    ) -> None:
        """Test with mixed long/short positions."""
        df = pd.DataFrame({
            'split': ['Train'] * 4,
            'date_affect': sample_trading_days[:4],
            'tickers': ['TEF.MC'] * 4,
            'cluster': [0, 1, 2, 3],
            'TR': [1, -1, 1, -1]  # Mixed
        })

        result = calculate_portfolio_returns(
            df, sample_trading_days, l_value=3, ts_dict=sample_ts_dict
        )

        assert 'All' in result

    def test_calculate_portfolio_returns_verbose(
        self,
        sample_articles_df: pd.DataFrame,
        sample_trading_days: list,
        sample_ts_dict: dict,
        capsys
    ) -> None:
        """Test with verbose=True prints information."""
        calculate_portfolio_returns(
            sample_articles_df,
            sample_trading_days,
            l_value=3,
            ts_dict=sample_ts_dict,
            verbose=True
        )

        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestCalculateTradingIntensityStatistics:
    """Tests for calculate_trading_intensity_statistics function."""

    @pytest.fixture
    def sample_trading_signal_evolution(self) -> dict:
        """Create sample trading signal evolution dictionary."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        return {
            'All': pd.DataFrame({
                'total_trading_signal': np.random.randint(0, 10, 10)
            }, index=dates),
            'Train': pd.DataFrame({
                'total_trading_signal': np.random.randint(0, 5, 5)
            }, index=dates[:5]),
            'Validation': pd.DataFrame({
                'total_trading_signal': np.random.randint(0, 3, 3)
            }, index=dates[5:8]),
            'Test': pd.DataFrame({
                'total_trading_signal': np.random.randint(0, 2, 2)
            }, index=dates[8:10])
        }

    @pytest.fixture
    def sample_turnover_stats(self) -> dict:
        """Create sample turnover statistics."""
        return {
            'All': 0.15,
            'Train': 0.12,
            'Validation': 0.18,
            'Test': 0.20
        }

    def test_calculate_trading_intensity_statistics_happy_path(
        self,
        sample_trading_signal_evolution: dict,
        sample_turnover_stats: dict
    ) -> None:
        """Test trading intensity statistics calculation."""
        result = calculate_trading_intensity_statistics(
            sample_trading_signal_evolution,
            sample_turnover_stats
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4  # All, Train, Validation, Test
        assert 'Split' in result.columns
        assert 'Avg_Positions' in result.columns
        assert 'Std_Positions' in result.columns
        assert 'Max_Positions' in result.columns
        assert 'Min_Positions' in result.columns
        assert 'Turnover_Pct' in result.columns
        assert 'Active_Days_Pct' in result.columns

    def test_calculate_trading_intensity_statistics_empty_evolution(
        self, sample_turnover_stats: dict
    ) -> None:
        """Test with empty trading signal evolution."""
        empty_evolution = {
            'All': pd.DataFrame({'total_trading_signal': []}, index=pd.DatetimeIndex([])),
            'Train': pd.DataFrame({'total_trading_signal': []}, index=pd.DatetimeIndex([])),
            'Validation': pd.DataFrame({'total_trading_signal': []}, index=pd.DatetimeIndex([])),
            'Test': pd.DataFrame({'total_trading_signal': []}, index=pd.DatetimeIndex([]))
        }

        result = calculate_trading_intensity_statistics(empty_evolution, sample_turnover_stats)
        assert isinstance(result, pd.DataFrame)

    def test_calculate_trading_intensity_statistics_no_turnover(
        self, sample_trading_signal_evolution: dict
    ) -> None:
        """Test with zero turnover."""
        zero_turnover = {split: 0.0 for split in ['All', 'Train', 'Validation', 'Test']}
        
        result = calculate_trading_intensity_statistics(
            sample_trading_signal_evolution, zero_turnover
        )

        assert (result['Turnover_Pct'] == 0.0).all()

    def test_calculate_trading_intensity_statistics_high_turnover(
        self, sample_trading_signal_evolution: dict
    ) -> None:
        """Test with high turnover."""
        high_turnover = {split: 1.0 for split in ['All', 'Train', 'Validation', 'Test']}
        
        result = calculate_trading_intensity_statistics(
            sample_trading_signal_evolution, high_turnover
        )

        assert (result['Turnover_Pct'] == 100.0).all()


class TestCalculatePortfolioStatistics:
    """Tests for calculate_portfolio_statistics function."""

    @pytest.fixture
    def sample_returns_df(self) -> pd.DataFrame:
        """Create sample returns DataFrame."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        returns = np.random.randn(100) * 0.01
        return pd.DataFrame({
            'gross_returns': returns,
            'net_returns': returns - 0.0001  # Slightly lower due to costs
        }, index=dates)

    def test_calculate_portfolio_statistics_happy_path(
        self, sample_returns_df: pd.DataFrame
    ) -> None:
        """Test portfolio statistics calculation."""
        stats = calculate_portfolio_statistics(sample_returns_df)

        assert isinstance(stats, dict)
        assert 'cumulative_return_gross' in stats
        assert 'cumulative_return_net' in stats
        assert 'annualized_return_gross' in stats
        assert 'annualized_return_net' in stats
        assert 'volatility_gross' in stats
        assert 'volatility_net' in stats
        assert 'sharpe_ratio_gross' in stats
        assert 'sharpe_ratio_net' in stats
        assert 'max_drawdown_gross' in stats
        assert 'max_drawdown_net' in stats
        assert 'calmar_ratio_gross' in stats
        assert 'calmar_ratio_net' in stats

    def test_calculate_portfolio_statistics_empty_returns(self) -> None:
        """Test with empty returns series."""
        empty_df = pd.DataFrame({
            'gross_returns': [],
            'net_returns': []
        }, index=pd.DatetimeIndex([]))

        stats = calculate_portfolio_statistics(empty_df)
        assert stats['annualized_return_gross'] == 0.0
        assert stats['annualized_return_net'] == 0.0

    def test_calculate_portfolio_statistics_all_positive_returns(self) -> None:
        """Test with all positive returns."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        positive_returns = pd.DataFrame({
            'gross_returns': [0.01] * 10,
            'net_returns': [0.01] * 10
        }, index=dates)

        stats = calculate_portfolio_statistics(positive_returns)
        assert stats['cumulative_return_gross'] > 0
        assert stats['max_drawdown_gross'] == 0.0  # No drawdowns

    def test_calculate_portfolio_statistics_all_negative_returns(self) -> None:
        """Test with all negative returns."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        negative_returns = pd.DataFrame({
            'gross_returns': [-0.01] * 10,
            'net_returns': [-0.01] * 10
        }, index=dates)

        stats = calculate_portfolio_statistics(negative_returns)
        assert stats['cumulative_return_gross'] < 0

    def test_calculate_portfolio_statistics_zero_returns(self) -> None:
        """Test with zero returns."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        zero_returns = pd.DataFrame({
            'gross_returns': [0.0] * 10,
            'net_returns': [0.0] * 10
        }, index=dates)

        stats = calculate_portfolio_statistics(zero_returns)
        assert stats['cumulative_return_gross'] == 0.0
        assert stats['volatility_gross'] == 0.0

    def test_calculate_portfolio_statistics_extreme_values(self) -> None:
        """Test with extreme return values."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        extreme_returns = pd.DataFrame({
            'gross_returns': [0.5, -0.3, 0.8, -0.2, 0.1] * 2,
            'net_returns': [0.5, -0.3, 0.8, -0.2, 0.1] * 2
        }, index=dates)

        stats = calculate_portfolio_statistics(extreme_returns)
        # Should handle extreme values without error
        assert isinstance(stats, dict)

    def test_calculate_portfolio_statistics_custom_risk_free_rate(
        self, sample_returns_df: pd.DataFrame
    ) -> None:
        """Test with custom risk-free rate."""
        stats = calculate_portfolio_statistics(
            sample_returns_df, risk_free_rate=0.02
        )

        # Sharpe ratio should account for risk-free rate
        assert isinstance(stats['sharpe_ratio_gross'], (float, type(np.nan)))

    def test_calculate_portfolio_statistics_custom_trading_days(
        self, sample_returns_df: pd.DataFrame
    ) -> None:
        """Test with custom trading days per year."""
        stats = calculate_portfolio_statistics(
            sample_returns_df, trading_days_per_year=365
        )

        # Annualized metrics should use custom period
        assert isinstance(stats['annualized_return_gross'], float)


