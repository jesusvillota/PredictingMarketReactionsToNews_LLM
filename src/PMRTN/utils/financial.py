"""Financial utilities for portfolio analysis and risk metrics.

This module provides utility functions for calculating portfolio statistics,
risk metrics, returns, and other financial indicators used in the analysis.
"""

import warnings
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from joblib import Parallel, delayed


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


# =============================================================================
# Stock Data Download Functions
# =============================================================================


def load_risk_free_rate(data_path: Path) -> pd.DataFrame:
    """Load risk-free rate data (ESTR) from CSV file.
    
    Loads the Euro Short-Term Rate (€STR) data and converts it to daily returns.
    The rate is converted from annual to daily using the formula:
    r_daily = (1 + r_annual)^(1/252) - 1
    
    Args:
        data_path: Path to the ESTR.csv file containing risk-free rate data.
                   Expected columns: ['datetime', 'TIME PERIOD', 'rf']
    
    Returns:
        DataFrame with datetime index and 'rf' column containing daily risk-free rates.
    
    Raises:
        FinancialUtilsError: If file not found or data format is invalid.
    
    Examples:
        >>> rf_data = load_risk_free_rate(Path('data/raw/ESTR.csv'))
        >>> print(rf_data.head())
                    rf
        datetime
        2020-01-01  0.0001
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FinancialUtilsError(f"Risk-free rate file not found: {data_path}")
    
    try:
        # Load ESTR data
        estr = pd.read_csv(data_path, index_col=0, parse_dates=True)
        estr.index.names = ['datetime']
        
        # Drop TIME PERIOD column if it exists
        if 'TIME PERIOD' in estr.columns:
            estr.drop(columns='TIME PERIOD', inplace=True)
        
        # Rename column to 'rf' if needed
        if estr.columns[0] != 'rf':
            estr.columns = ['rf']
        
        # Convert from percentage to decimal
        estr['rf'] = estr['rf'] / 100
        
        # Convert annual rate to daily rate
        estr['rf'] = (1 + estr['rf']) ** (1 / 252) - 1
        
        return estr
        
    except Exception as e:
        raise FinancialUtilsError(f"Error loading risk-free rate data: {e}")


def download_market_index(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp
) -> pd.DataFrame:
    """Download market index data and calculate returns.
    
    Downloads market index data (e.g., IBEX 35) using yfinance and calculates
    daily returns from adjusted close prices.
    
    Args:
        ticker: Market index ticker symbol (e.g., '^IBEX' for IBEX 35).
        start_date: Start date for downloading data.
        end_date: End date for downloading data.
    
    Returns:
        DataFrame with datetime index and 'r_market' column containing daily returns.
    
    Raises:
        FinancialUtilsError: If download fails or no data is retrieved.
    
    Examples:
        >>> from datetime import datetime
        >>> market_data = download_market_index(
        ...     '^IBEX',
        ...     pd.Timestamp('2020-01-01'),
        ...     pd.Timestamp('2023-12-31')
        ... )
        >>> print(market_data.head())
                    r_market
        datetime
        2020-01-02  0.0123
    """
    # Suppress yfinance warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance.utils")
    
    try:
        print(f"Downloading market index data for {ticker}...")
        
        # Download data with buffer
        ibex = yf.download(
            ticker,
            start=start_date - timedelta(days=1),
            end=end_date + timedelta(days=1),
            progress=False
        )
        
        if ibex.empty:
            raise FinancialUtilsError(f"No data retrieved for market index {ticker}")
        
        # Calculate returns
        ibex['r_market'] = ibex['Adj Close'].pct_change()
        ibex.index.names = ['datetime']
        
        # Keep only returns column and drop NaN
        ibex = ibex[['r_market']].dropna()
        
        print(f"Successfully downloaded {len(ibex)} trading days for {ticker}")
        
        return ibex
        
    except Exception as e:
        raise FinancialUtilsError(f"Error downloading market index {ticker}: {e}")


def _fetch_single_ticker_data(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    rf_series: pd.Series
) -> Tuple[str, Optional[pd.Series], Optional[pd.Series], Optional[str]]:
    """Helper function to fetch and process data for a single ticker.
    
    This function is designed to be called in parallel for multiple tickers.
    
    Args:
        ticker: Stock ticker symbol.
        start_date: Start date for downloading data.
        end_date: End date for downloading data.
        rf_series: Series with risk-free rate data (used for excess returns).
    
    Returns:
        Tuple of (ticker, returns, excess_returns, error_message).
        If successful, error_message is None. If failed, returns and excess_returns are None.
    """
    # Suppress yfinance warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance.utils")
    
    try:
        print(f"Downloading data for ticker: {ticker}")
        
        # Download price data
        prices = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False
        )
        
        if prices.empty:
            print(f"No data found for ticker: {ticker}")
            return ticker, None, None, "No data"
        
        # Convert index to date (remove time component)
        prices.index = prices.index.date
        
        # Calculate returns
        r_ticker = prices['Adj Close'].pct_change().dropna()
        
        # Calculate excess returns (subtract lagged risk-free rate)
        r_ticker_excess = r_ticker - rf_series.shift(1)
        
        return ticker, r_ticker, r_ticker_excess, None
            
    except Exception as e:
        print(f"Failed to download data for ticker: {ticker}. Error: {e}")
        return ticker, None, None, str(e)


def download_stock_returns(
    tickers: List[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    rf_data: pd.DataFrame,
    n_jobs: int = -1
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Download stock returns data for multiple tickers in parallel.
    
    Downloads adjusted close prices for all tickers, calculates returns,
    and computes excess returns over the risk-free rate. Uses parallel
    processing for efficient downloading.
    
    Args:
        tickers: List of stock ticker symbols to download.
        start_date: Start date for downloading data.
        end_date: End date for downloading data.
        rf_data: DataFrame with datetime index and 'rf' column containing
                 daily risk-free rates.
        n_jobs: Number of parallel jobs to run (-1 uses all available cores).
    
    Returns:
        Tuple containing:
            - returns_df: DataFrame with datetime index and columns for each ticker's
                         returns (r_{ticker}) and excess returns (r_{ticker}_excess).
            - successful_tickers: List of tickers successfully downloaded.
            - failed_tickers: List of tickers that failed to download.
    
    Raises:
        FinancialUtilsError: If inputs are invalid or processing fails.
    
    Examples:
        >>> tickers = ['SAN.MC', 'TEF.MC', 'BBVA.MC']
        >>> start = pd.Timestamp('2020-01-01')
        >>> end = pd.Timestamp('2023-12-31')
        >>> rf = load_risk_free_rate(Path('data/raw/ESTR.csv'))
        >>> returns_df, successful, failed = download_stock_returns(
        ...     tickers, start, end, rf, n_jobs=-1
        ... )
        >>> print(f"Downloaded {len(successful)} tickers successfully")
        >>> print(f"Failed: {len(failed)} tickers")
    """
    if not tickers:
        raise FinancialUtilsError("Tickers list is empty")
    
    if rf_data.empty:
        raise FinancialUtilsError("Risk-free rate data is empty")
    
    print(f"\nDownloading stock returns for {len(tickers)} tickers...")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Using {n_jobs if n_jobs > 0 else 'all available'} parallel workers")
    
    # Add buffer to dates
    download_start = start_date - timedelta(days=1)
    download_end = end_date + timedelta(days=1)
    
    # Parallel processing using Joblib
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fetch_single_ticker_data)(
            ticker, download_start, download_end, rf_data['rf']
        )
        for ticker in tickers
    )
    
    # Initialize output DataFrame with risk-free rate
    returns_df = rf_data.copy()
    
    # Track successful and failed tickers
    successful_tickers = []
    failed_tickers = []
    
    # Process results
    for ticker, r_ticker, r_ticker_excess, error in results:
        if error is None:
            # Add returns to DataFrame
            returns_df[f'r_{ticker}'] = r_ticker
            returns_df[f'r_{ticker}_excess'] = r_ticker_excess
            successful_tickers.append(ticker)
        else:
            failed_tickers.append(ticker)
    
    print(f"\n✓ Successfully downloaded: {len(successful_tickers)} tickers")
    print(f"✗ Failed to download: {len(failed_tickers)} tickers")
    
    if failed_tickers:
        print(f"Failed tickers: {', '.join(failed_tickers[:10])}")
        if len(failed_tickers) > 10:
            print(f"... and {len(failed_tickers) - 10} more")
    
    return returns_df, successful_tickers, failed_tickers
