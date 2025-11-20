"""Financial utilities for portfolio analysis and risk metrics.

This module provides utility functions for calculating portfolio statistics,
risk metrics, returns, and other financial indicators used in the analysis.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


class FinancialUtilsError(Exception):
    """Raised when financial calculations encounter errors."""

    pass


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sharpe ratio.

    Args:
        returns: Series of portfolio returns.
        risk_free_rate: Risk-free rate (default 0.0).
        periods_per_year: Number of trading periods per year (default 252 for daily).

    Returns:
        Annualized Sharpe ratio.

    Raises:
        FinancialUtilsError: If returns are empty or standard deviation is zero.
    """
    if len(returns) == 0:
        raise FinancialUtilsError("Returns series is empty")

    excess_returns = returns - risk_free_rate
    mean_excess_return = excess_returns.mean()
    std_excess_return = excess_returns.std()

    if std_excess_return == 0:
        if mean_excess_return == 0:
            return 0.0
        raise FinancialUtilsError("Standard deviation is zero with non-zero returns")

    sharpe_ratio = (mean_excess_return / std_excess_return) * np.sqrt(periods_per_year)
    return sharpe_ratio


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sortino ratio (uses downside deviation).

    Args:
        returns: Series of portfolio returns.
        risk_free_rate: Risk-free rate (default 0.0).
        periods_per_year: Number of trading periods per year (default 252 for daily).

    Returns:
        Annualized Sortino ratio.

    Raises:
        FinancialUtilsError: If returns are empty or downside deviation is zero.
    """
    if len(returns) == 0:
        raise FinancialUtilsError("Returns series is empty")

    excess_returns = returns - risk_free_rate
    mean_excess_return = excess_returns.mean()

    # Calculate downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        # No negative returns - excellent performance
        return np.inf if mean_excess_return > 0 else 0.0

    downside_deviation = downside_returns.std()
    if downside_deviation == 0:
        if mean_excess_return == 0:
            return 0.0
        raise FinancialUtilsError("Downside deviation is zero with non-zero returns")

    sortino_ratio = (mean_excess_return / downside_deviation) * np.sqrt(periods_per_year)
    return sortino_ratio


def calculate_calmar_ratio(
    returns: pd.Series, periods_per_year: int = 252
) -> float:
    """Calculate Calmar ratio (annualized return / maximum drawdown).

    Args:
        returns: Series of portfolio returns.
        periods_per_year: Number of trading periods per year (default 252 for daily).

    Returns:
        Calmar ratio.

    Raises:
        FinancialUtilsError: If returns are empty or max drawdown is zero.
    """
    if len(returns) == 0:
        raise FinancialUtilsError("Returns series is empty")

    # Calculate annualized return
    mean_return = returns.mean()
    annualized_return = mean_return * periods_per_year

    # Calculate maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdowns.min()

    if max_drawdown == 0:
        # No drawdown - excellent performance
        return np.inf if annualized_return > 0 else 0.0

    calmar_ratio = annualized_return / abs(max_drawdown)
    return calmar_ratio


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from returns series.

    Args:
        returns: Series of portfolio returns.

    Returns:
        Maximum drawdown (negative value).

    Raises:
        FinancialUtilsError: If returns are empty.
    """
    if len(returns) == 0:
        raise FinancialUtilsError("Returns series is empty")

    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdowns.min()

    return max_drawdown


def calculate_cumulative_return(returns: pd.Series) -> pd.Series:
    """Calculate cumulative returns from simple returns.

    Args:
        returns: Series of simple returns.

    Returns:
        Series of cumulative returns.

    Raises:
        FinancialUtilsError: If returns are empty.
    """
    if len(returns) == 0:
        raise FinancialUtilsError("Returns series is empty")

    cumulative_returns = (1 + returns).cumprod() - 1
    return cumulative_returns


def calculate_annualized_return(
    returns: pd.Series, periods_per_year: int = 252
) -> float:
    """Calculate annualized return from simple returns.

    Args:
        returns: Series of simple returns.
        periods_per_year: Number of trading periods per year (default 252 for daily).

    Returns:
        Annualized return.

    Raises:
        FinancialUtilsError: If returns are empty.
    """
    if len(returns) == 0:
        raise FinancialUtilsError("Returns series is empty")

    mean_return = returns.mean()
    annualized_return = mean_return * periods_per_year
    return annualized_return


def calculate_annualized_volatility(
    returns: pd.Series, periods_per_year: int = 252
) -> float:
    """Calculate annualized volatility from simple returns.

    Args:
        returns: Series of simple returns.
        periods_per_year: Number of trading periods per year (default 252 for daily).

    Returns:
        Annualized volatility.

    Raises:
        FinancialUtilsError: If returns are empty.
    """
    if len(returns) == 0:
        raise FinancialUtilsError("Returns series is empty")

    volatility = returns.std() * np.sqrt(periods_per_year)
    return volatility


def calculate_portfolio_statistics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """Calculate comprehensive portfolio statistics.

    Args:
        returns: Series of portfolio returns.
        risk_free_rate: Risk-free rate (default 0.0).
        periods_per_year: Number of trading periods per year (default 252 for daily).

    Returns:
        Dictionary containing:
            - cumulative_return: Final cumulative return
            - annualized_return: Annualized mean return
            - annualized_volatility: Annualized standard deviation
            - sharpe_ratio: Annualized Sharpe ratio
            - sortino_ratio: Annualized Sortino ratio
            - calmar_ratio: Calmar ratio
            - max_drawdown: Maximum drawdown (negative value)

    Raises:
        FinancialUtilsError: If returns are empty or calculations fail.
    """
    if len(returns) == 0:
        raise FinancialUtilsError("Returns series is empty")

    cumulative_return = calculate_cumulative_return(returns).iloc[-1]
    annualized_return = calculate_annualized_return(returns, periods_per_year)
    annualized_volatility = calculate_annualized_volatility(returns, periods_per_year)
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    calmar = calculate_calmar_ratio(returns, periods_per_year)
    max_dd = calculate_max_drawdown(returns)

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_dd,
    }


def calculate_turnover(
    current_positions: Dict[str, float],
    previous_positions: Dict[str, float],
) -> float:
    """Calculate portfolio turnover between two periods.

    Turnover is calculated as the sum of absolute changes in positions
    divided by the total position size.

    Args:
        current_positions: Dictionary mapping ticker to position size (current period).
        previous_positions: Dictionary mapping ticker to position size (previous period).

    Returns:
        Portfolio turnover (0.0 to 2.0, where 2.0 is complete replacement).

    Raises:
        FinancialUtilsError: If inputs are invalid.
    """
    if not isinstance(current_positions, dict) or not isinstance(
        previous_positions, dict
    ):
        raise FinancialUtilsError("Positions must be dictionaries")

    if not previous_positions:
        # First period - no turnover calculation
        return 0.0

    # Calculate total current position size
    total_position_size = sum(abs(pos) for pos in current_positions.values())

    if total_position_size == 0:
        return 0.0

    # Calculate position changes
    all_tickers = set(current_positions.keys()) | set(previous_positions.keys())
    position_changes = 0.0

    for ticker in all_tickers:
        curr_pos = current_positions.get(ticker, 0.0)
        prev_pos = previous_positions.get(ticker, 0.0)
        position_changes += abs(curr_pos - prev_pos)

    turnover = position_changes / total_position_size
    return turnover


def calculate_trading_costs(
    turnover: float, cost_bps: float = 10.0
) -> float:
    """Calculate trading costs from turnover.

    Args:
        turnover: Portfolio turnover (from calculate_turnover).
        cost_bps: Trading cost in basis points (default 10 bps).

    Returns:
        Trading cost as a fraction of portfolio value.

    Raises:
        FinancialUtilsError: If inputs are invalid.
    """
    if turnover < 0:
        raise FinancialUtilsError("Turnover must be non-negative")
    if cost_bps < 0:
        raise FinancialUtilsError("Cost in bps must be non-negative")

    trading_cost = turnover * (cost_bps / 10000.0)
    return trading_cost


def calculate_excess_returns(
    returns: pd.Series, benchmark_returns: pd.Series
) -> pd.Series:
    """Calculate excess returns over a benchmark.

    Args:
        returns: Series of portfolio returns.
        benchmark_returns: Series of benchmark returns (must have same index).

    Returns:
        Series of excess returns.

    Raises:
        FinancialUtilsError: If series don't have matching indices.
    """
    if len(returns) == 0 or len(benchmark_returns) == 0:
        raise FinancialUtilsError("Returns series are empty")

    # Align indices
    common_index = returns.index.intersection(benchmark_returns.index)
    if len(common_index) == 0:
        raise FinancialUtilsError("No common dates between returns series")

    excess_returns = returns.loc[common_index] - benchmark_returns.loc[common_index]
    return excess_returns


def calculate_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate information ratio (excess return / tracking error).

    Args:
        returns: Series of portfolio returns.
        benchmark_returns: Series of benchmark returns.
        periods_per_year: Number of trading periods per year (default 252 for daily).

    Returns:
        Annualized information ratio.

    Raises:
        FinancialUtilsError: If calculation fails.
    """
    excess_returns = calculate_excess_returns(returns, benchmark_returns)

    if len(excess_returns) == 0:
        raise FinancialUtilsError("No excess returns to calculate")

    mean_excess = excess_returns.mean()
    tracking_error = excess_returns.std()

    if tracking_error == 0:
        if mean_excess == 0:
            return 0.0
        raise FinancialUtilsError("Tracking error is zero with non-zero excess return")

    information_ratio = (mean_excess / tracking_error) * np.sqrt(periods_per_year)
    return information_ratio


def calculate_beta(
    returns: pd.Series, market_returns: pd.Series
) -> Tuple[float, float]:
    """Calculate portfolio beta and alpha using linear regression.

    Args:
        returns: Series of portfolio returns.
        market_returns: Series of market returns.

    Returns:
        Tuple of (beta, alpha).

    Raises:
        FinancialUtilsError: If calculation fails.
    """
    # Align indices
    common_index = returns.index.intersection(market_returns.index)
    if len(common_index) < 2:
        raise FinancialUtilsError("Insufficient common dates for beta calculation")

    portfolio_returns = returns.loc[common_index].values
    market_rets = market_returns.loc[common_index].values

    # Calculate covariance and variance
    covariance = np.cov(portfolio_returns, market_rets)[0, 1]
    market_variance = np.var(market_rets)

    if market_variance == 0:
        raise FinancialUtilsError("Market variance is zero")

    beta = covariance / market_variance
    alpha = np.mean(portfolio_returns) - beta * np.mean(market_rets)

    return beta, alpha


def calculate_var(
    returns: pd.Series, confidence_level: float = 0.95
) -> float:
    """Calculate Value at Risk (VaR) using historical method.

    Args:
        returns: Series of portfolio returns.
        confidence_level: Confidence level (default 0.95 for 95% VaR).

    Returns:
        Value at Risk (negative value representing potential loss).

    Raises:
        FinancialUtilsError: If returns are empty or confidence level is invalid.
    """
    if len(returns) == 0:
        raise FinancialUtilsError("Returns series is empty")

    if not 0 < confidence_level < 1:
        raise FinancialUtilsError("Confidence level must be between 0 and 1")

    var = returns.quantile(1 - confidence_level)
    return var


def calculate_cvar(
    returns: pd.Series, confidence_level: float = 0.95
) -> float:
    """Calculate Conditional Value at Risk (CVaR/Expected Shortfall).

    CVaR is the expected return given that the return is below VaR.

    Args:
        returns: Series of portfolio returns.
        confidence_level: Confidence level (default 0.95 for 95% CVaR).

    Returns:
        Conditional Value at Risk (negative value).

    Raises:
        FinancialUtilsError: If returns are empty or confidence level is invalid.
    """
    if len(returns) == 0:
        raise FinancialUtilsError("Returns series is empty")

    var = calculate_var(returns, confidence_level)
    cvar = returns[returns <= var].mean()
    return cvar
