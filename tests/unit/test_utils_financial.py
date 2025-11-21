"""Tests for financial utilities module."""

import numpy as np
import pandas as pd
import pytest

from PMRTN.utils.financial import (
    FinancialUtilsError,
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_beta,
    calculate_calmar_ratio,
    calculate_cumulative_return,
    calculate_cvar,
    calculate_excess_returns,
    calculate_information_ratio,
    calculate_max_drawdown,
    calculate_portfolio_statistics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_trading_costs,
    calculate_turnover,
    calculate_var,
)


@pytest.fixture
def sample_returns():
    """Create sample returns series for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
    return returns


@pytest.fixture
def positive_returns():
    """Create positive returns series."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    returns = pd.Series(np.random.uniform(0.001, 0.01, 100), index=dates)
    return returns


@pytest.fixture
def negative_returns():
    """Create negative returns series."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    returns = pd.Series(np.random.uniform(-0.01, -0.001, 100), index=dates)
    return returns


class TestSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    def test_sharpe_ratio_calculation(self, sample_returns):
        """Test basic Sharpe ratio calculation."""
        sharpe = calculate_sharpe_ratio(sample_returns)
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        assert not np.isinf(sharpe)

    def test_sharpe_ratio_with_risk_free_rate(self, sample_returns):
        """Test Sharpe ratio with non-zero risk-free rate."""
        sharpe = calculate_sharpe_ratio(sample_returns, risk_free_rate=0.0001)
        assert isinstance(sharpe, float)

    def test_sharpe_ratio_empty_returns(self):
        """Test error on empty returns."""
        empty_returns = pd.Series([], dtype=float)
        with pytest.raises(FinancialUtilsError, match="empty"):
            calculate_sharpe_ratio(empty_returns)

    def test_sharpe_ratio_zero_volatility(self):
        """Test Sharpe ratio returns zero when volatility is zero but mean is also zero."""
        constant_returns = pd.Series([0.0] * 100)
        sharpe = calculate_sharpe_ratio(constant_returns)
        assert sharpe == 0.0

    def test_sharpe_ratio_zero_mean_and_volatility(self):
        """Test zero Sharpe when mean and volatility are zero."""
        zero_returns = pd.Series([0.0] * 100)
        sharpe = calculate_sharpe_ratio(zero_returns)
        assert sharpe == 0.0


class TestSortinoRatio:
    """Tests for Sortino ratio calculation."""

    def test_sortino_ratio_calculation(self, sample_returns):
        """Test basic Sortino ratio calculation."""
        sortino = calculate_sortino_ratio(sample_returns)
        assert isinstance(sortino, float)
        assert not np.isnan(sortino)

    def test_sortino_ratio_no_negative_returns(self, positive_returns):
        """Test Sortino ratio with no negative returns."""
        sortino = calculate_sortino_ratio(positive_returns)
        # Should return infinity for perfect upside performance
        assert sortino == np.inf or sortino > 10

    def test_sortino_ratio_empty_returns(self):
        """Test error on empty returns."""
        empty_returns = pd.Series([], dtype=float)
        with pytest.raises(FinancialUtilsError, match="empty"):
            calculate_sortino_ratio(empty_returns)


class TestCalmarRatio:
    """Tests for Calmar ratio calculation."""

    def test_calmar_ratio_calculation(self, sample_returns):
        """Test basic Calmar ratio calculation."""
        calmar = calculate_calmar_ratio(sample_returns)
        assert isinstance(calmar, float)
        assert not np.isnan(calmar)

    def test_calmar_ratio_no_drawdown(self, positive_returns):
        """Test Calmar ratio with no drawdown."""
        calmar = calculate_calmar_ratio(positive_returns)
        # Should return infinity for no drawdown
        assert calmar == np.inf or calmar > 100

    def test_calmar_ratio_empty_returns(self):
        """Test error on empty returns."""
        empty_returns = pd.Series([], dtype=float)
        with pytest.raises(FinancialUtilsError, match="empty"):
            calculate_calmar_ratio(empty_returns)


class TestMaxDrawdown:
    """Tests for maximum drawdown calculation."""

    def test_max_drawdown_calculation(self, sample_returns):
        """Test basic max drawdown calculation."""
        max_dd = calculate_max_drawdown(sample_returns)
        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Drawdown should be negative or zero

    def test_max_drawdown_positive_returns(self, positive_returns):
        """Test max drawdown with all positive returns."""
        max_dd = calculate_max_drawdown(positive_returns)
        assert max_dd == 0.0

    def test_max_drawdown_negative_returns(self, negative_returns):
        """Test max drawdown with all negative returns."""
        max_dd = calculate_max_drawdown(negative_returns)
        assert max_dd < 0

    def test_max_drawdown_empty_returns(self):
        """Test error on empty returns."""
        empty_returns = pd.Series([], dtype=float)
        with pytest.raises(FinancialUtilsError, match="empty"):
            calculate_max_drawdown(empty_returns)


class TestCumulativeReturn:
    """Tests for cumulative return calculation."""

    def test_cumulative_return_calculation(self, sample_returns):
        """Test basic cumulative return calculation."""
        cum_returns = calculate_cumulative_return(sample_returns)
        assert isinstance(cum_returns, pd.Series)
        assert len(cum_returns) == len(sample_returns)
        # First cumulative return should be approximately equal to first return
        assert abs(cum_returns.iloc[0] - sample_returns.iloc[0]) < 1e-10

    def test_cumulative_return_positive(self, positive_returns):
        """Test cumulative return with positive returns."""
        cum_returns = calculate_cumulative_return(positive_returns)
        # Cumulative should be monotonically increasing
        assert cum_returns.iloc[-1] > cum_returns.iloc[0]

    def test_cumulative_return_empty_returns(self):
        """Test error on empty returns."""
        empty_returns = pd.Series([], dtype=float)
        with pytest.raises(FinancialUtilsError, match="empty"):
            calculate_cumulative_return(empty_returns)


class TestAnnualizedMetrics:
    """Tests for annualized return and volatility."""

    def test_annualized_return(self, sample_returns):
        """Test annualized return calculation."""
        ann_return = calculate_annualized_return(sample_returns)
        assert isinstance(ann_return, float)

    def test_annualized_volatility(self, sample_returns):
        """Test annualized volatility calculation."""
        ann_vol = calculate_annualized_volatility(sample_returns)
        assert isinstance(ann_vol, float)
        assert ann_vol >= 0

    def test_annualized_return_empty(self):
        """Test error on empty returns."""
        empty_returns = pd.Series([], dtype=float)
        with pytest.raises(FinancialUtilsError, match="empty"):
            calculate_annualized_return(empty_returns)

    def test_annualized_volatility_empty(self):
        """Test error on empty returns."""
        empty_returns = pd.Series([], dtype=float)
        with pytest.raises(FinancialUtilsError, match="empty"):
            calculate_annualized_volatility(empty_returns)


class TestPortfolioStatistics:
    """Tests for comprehensive portfolio statistics."""

    def test_portfolio_statistics_calculation(self, sample_returns):
        """Test calculation of all portfolio statistics."""
        stats = calculate_portfolio_statistics(sample_returns)

        assert isinstance(stats, dict)
        assert 'cumulative_return' in stats
        assert 'annualized_return' in stats
        assert 'annualized_volatility' in stats
        assert 'sharpe_ratio' in stats
        assert 'sortino_ratio' in stats
        assert 'calmar_ratio' in stats
        assert 'max_drawdown' in stats

        # Check types
        for key, value in stats.items():
            assert isinstance(value, (float, np.floating))

    def test_portfolio_statistics_empty_returns(self):
        """Test error on empty returns."""
        empty_returns = pd.Series([], dtype=float)
        with pytest.raises(FinancialUtilsError, match="empty"):
            calculate_portfolio_statistics(empty_returns)


class TestTurnover:
    """Tests for turnover calculation."""

    def test_turnover_basic(self):
        """Test basic turnover calculation."""
        current = {'AAPL': 0.5, 'GOOGL': 0.5}
        previous = {'AAPL': 0.3, 'GOOGL': 0.7}
        turnover = calculate_turnover(current, previous)

        assert isinstance(turnover, float)
        assert 0 <= turnover <= 2.0

    def test_turnover_no_change(self):
        """Test turnover when positions don't change."""
        positions = {'AAPL': 0.5, 'GOOGL': 0.5}
        turnover = calculate_turnover(positions, positions)
        assert turnover == 0.0

    def test_turnover_complete_replacement(self):
        """Test turnover with complete position replacement."""
        current = {'AAPL': 1.0}
        previous = {'GOOGL': 1.0}
        turnover = calculate_turnover(current, previous)
        assert turnover == 2.0

    def test_turnover_first_period(self):
        """Test turnover calculation for first period."""
        current = {'AAPL': 0.5, 'GOOGL': 0.5}
        previous = {}
        turnover = calculate_turnover(current, previous)
        assert turnover == 0.0

    def test_turnover_empty_current(self):
        """Test turnover with empty current positions."""
        current = {}
        previous = {'AAPL': 0.5}
        turnover = calculate_turnover(current, previous)
        assert turnover == 0.0

    def test_turnover_invalid_input(self):
        """Test error with invalid input types."""
        with pytest.raises(FinancialUtilsError, match="must be dictionaries"):
            calculate_turnover([1, 2], {'AAPL': 0.5})


class TestTradingCosts:
    """Tests for trading costs calculation."""

    def test_trading_costs_basic(self):
        """Test basic trading costs calculation."""
        costs = calculate_trading_costs(turnover=0.5, cost_bps=10)
        assert isinstance(costs, float)
        assert costs == 0.5 * 10 / 10000

    def test_trading_costs_zero_turnover(self):
        """Test trading costs with zero turnover."""
        costs = calculate_trading_costs(turnover=0.0, cost_bps=10)
        assert costs == 0.0

    def test_trading_costs_negative_turnover(self):
        """Test error with negative turnover."""
        with pytest.raises(FinancialUtilsError, match="non-negative"):
            calculate_trading_costs(turnover=-0.5, cost_bps=10)

    def test_trading_costs_negative_bps(self):
        """Test error with negative cost in bps."""
        with pytest.raises(FinancialUtilsError, match="non-negative"):
            calculate_trading_costs(turnover=0.5, cost_bps=-10)


class TestExcessReturns:
    """Tests for excess returns calculation."""

    def test_excess_returns_calculation(self, sample_returns):
        """Test basic excess returns calculation."""
        benchmark = sample_returns * 0.8  # Correlated benchmark
        excess = calculate_excess_returns(sample_returns, benchmark)

        assert isinstance(excess, pd.Series)
        assert len(excess) == len(sample_returns)

    def test_excess_returns_different_indices(self):
        """Test excess returns with partially overlapping indices."""
        dates1 = pd.date_range('2020-01-01', periods=100, freq='D')
        dates2 = pd.date_range('2020-02-15', periods=100, freq='D')
        returns1 = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates1)
        returns2 = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates2)

        excess = calculate_excess_returns(returns1, returns2)
        assert len(excess) > 0
        assert len(excess) < min(len(returns1), len(returns2))

    def test_excess_returns_no_overlap(self):
        """Test error when no common dates."""
        dates1 = pd.date_range('2020-01-01', periods=50, freq='D')
        dates2 = pd.date_range('2021-01-01', periods=50, freq='D')
        returns1 = pd.Series(np.random.normal(0.001, 0.02, 50), index=dates1)
        returns2 = pd.Series(np.random.normal(0.001, 0.02, 50), index=dates2)

        with pytest.raises(FinancialUtilsError, match="No common dates"):
            calculate_excess_returns(returns1, returns2)


class TestInformationRatio:
    """Tests for information ratio calculation."""

    def test_information_ratio_calculation(self, sample_returns):
        """Test basic information ratio calculation."""
        benchmark = sample_returns * 0.9 + np.random.normal(0, 0.001, len(sample_returns))
        ir = calculate_information_ratio(sample_returns, benchmark)

        assert isinstance(ir, float)
        assert not np.isnan(ir)


class TestBeta:
    """Tests for beta calculation."""

    def test_beta_calculation(self, sample_returns):
        """Test basic beta calculation."""
        market_returns = sample_returns * 1.2 + np.random.normal(0, 0.01, len(sample_returns))
        beta, alpha = calculate_beta(sample_returns, market_returns)

        assert isinstance(beta, float)
        assert isinstance(alpha, float)
        assert not np.isnan(beta)
        assert not np.isnan(alpha)

    def test_beta_insufficient_data(self):
        """Test error with insufficient data."""
        dates = pd.date_range('2020-01-01', periods=1, freq='D')
        returns1 = pd.Series([0.01], index=dates)
        returns2 = pd.Series([0.01], index=dates)

        with pytest.raises(FinancialUtilsError, match="Insufficient common dates"):
            calculate_beta(returns1, returns2)

    def test_beta_zero_market_variance(self):
        """Test that beta calculation handles near-zero market variance."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
        constant_market = pd.Series([0.0] * 100, index=dates)

        with pytest.raises(FinancialUtilsError, match="Market variance is zero"):
            calculate_beta(returns, constant_market)


class TestVaR:
    """Tests for Value at Risk calculation."""

    def test_var_calculation(self, sample_returns):
        """Test basic VaR calculation."""
        var = calculate_var(sample_returns, confidence_level=0.95)

        assert isinstance(var, float)
        assert not np.isnan(var)

    def test_var_different_confidence_levels(self, sample_returns):
        """Test VaR at different confidence levels."""
        var_90 = calculate_var(sample_returns, confidence_level=0.90)
        var_95 = calculate_var(sample_returns, confidence_level=0.95)
        var_99 = calculate_var(sample_returns, confidence_level=0.99)

        # Higher confidence level should give worse (more negative) VaR
        assert var_99 <= var_95 <= var_90

    def test_var_invalid_confidence_level(self, sample_returns):
        """Test error with invalid confidence level."""
        with pytest.raises(FinancialUtilsError, match="between 0 and 1"):
            calculate_var(sample_returns, confidence_level=1.5)

    def test_var_empty_returns(self):
        """Test error on empty returns."""
        empty_returns = pd.Series([], dtype=float)
        with pytest.raises(FinancialUtilsError, match="empty"):
            calculate_var(empty_returns)


class TestCVaR:
    """Tests for Conditional Value at Risk calculation."""

    def test_cvar_calculation(self, sample_returns):
        """Test basic CVaR calculation."""
        cvar = calculate_cvar(sample_returns, confidence_level=0.95)

        assert isinstance(cvar, float)
        assert not np.isnan(cvar)

    def test_cvar_worse_than_var(self, sample_returns):
        """Test that CVaR is worse (more negative) than VaR."""
        var = calculate_var(sample_returns, confidence_level=0.95)
        cvar = calculate_cvar(sample_returns, confidence_level=0.95)

        # CVaR should be worse than or equal to VaR
        assert cvar <= var

    def test_cvar_empty_returns(self):
        """Test error on empty returns."""
        empty_returns = pd.Series([], dtype=float)
        with pytest.raises(FinancialUtilsError, match="empty"):
            calculate_cvar(empty_returns)


class TestIntegration:
    """Integration tests for financial utilities."""

    def test_complete_portfolio_analysis(self, sample_returns):
        """Test complete portfolio analysis workflow."""
        # Calculate all statistics
        stats = calculate_portfolio_statistics(sample_returns)

        # Verify relationships
        assert stats['annualized_volatility'] >= 0
        if stats['annualized_volatility'] > 0:
            # Sharpe ratio should be finite when volatility is positive
            assert not np.isinf(stats['sharpe_ratio'])

        # Cumulative return should be related to final cumulative value
        cum_return = calculate_cumulative_return(sample_returns)
        assert stats['cumulative_return'] == cum_return.iloc[-1]

        # Max drawdown should be negative or zero
        assert stats['max_drawdown'] <= 0

    def test_turnover_and_costs_workflow(self):
        """Test turnover and trading costs workflow."""
        positions_t0 = {'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.3}
        positions_t1 = {'AAPL': 0.5, 'GOOGL': 0.2, 'AMZN': 0.3}

        # Calculate turnover
        turnover = calculate_turnover(positions_t1, positions_t0)

        # Calculate costs
        costs = calculate_trading_costs(turnover, cost_bps=10)

        assert turnover > 0
        assert costs > 0
        assert costs == turnover * 10 / 10000
