"""Backtesting utilities for event study and trading strategy evaluation.

This module provides functions for performing event studies on individual
(ticker, date) pairs, calculating abnormal returns using a market model,
and computing performance metrics like Sharpe ratios.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


class BacktestingError(Exception):
    """Raised when backtesting operations fail."""
    pass


def calculate_trading_strategy_data(
    ticker: str,
    date_affect: pd.Timestamp,
    returns_df: pd.DataFrame,
    successful_tickers: Dict[str, bool],
    l_max: int = 260,
    market_model_window: int = 100,
    market_model_buffer: int = 10
) -> Optional[pd.DataFrame]:
    """Calculate event study metrics for a single (ticker, date) pair.
    
    This function performs an event study by:
    1. Estimating a market model (CAPM) using pre-event data
    2. Calculating abnormal returns (AR) as residuals from the market model
    3. Computing cumulative abnormal returns (CAR) over L trading days
    4. Computing performance metrics (mean, volatility, Sharpe ratio)
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'SAN.MC' for Santander).
    date_affect : pd.Timestamp
        Event date affecting the stock.
    returns_df : pd.DataFrame
        DataFrame with returns data, must have columns:
        - Index: trading dates
        - f'r_{ticker}_excess': excess returns for the ticker
        - 'r_market_excess': excess market returns
    successful_tickers : Dict[str, bool]
        Dictionary indicating which tickers have complete data.
    l_max : int, default=260
        Maximum holding period in trading days (approximately 1 year).
    market_model_window : int, default=100
        Number of trading days for market model estimation.
    market_model_buffer : int, default=10
        Buffer days before event date (avoids event contamination).
    
    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame with columns ['AR', 'CAR', 'μ', 'σ', 'SR'] for each L in [0, l_max],
        or None if calculation fails.
        - AR: Abnormal return at day L
        - CAR: Cumulative abnormal return from day 0 to day L
        - μ: Average daily log return over L days
        - σ: Standard deviation of daily log returns
        - SR: Annualized Sharpe ratio (μ/σ * sqrt(252))
    
    Notes
    -----
    The market model is estimated as:
        r_i,t^excess = α + β * r_M,t^excess + ε_t
    
    Abnormal returns are:
        AR_t = α + ε_t = r_i,t^excess - β * r_M,t^excess
    
    Cumulative abnormal returns use compounding:
        CAR_L = ∏(1 + AR_t) - 1 for t in [0, L]
    
    Examples
    --------
    >>> returns = pd.DataFrame({
    ...     'r_SAN.MC_excess': [0.01, -0.005, 0.002],
    ...     'r_market_excess': [0.008, -0.003, 0.001]
    ... }, index=pd.date_range('2020-01-01', periods=3, freq='B'))
    >>> result = calculate_trading_strategy_data(
    ...     'SAN.MC', 
    ...     pd.Timestamp('2020-01-02'),
    ...     returns,
    ...     {'SAN.MC': True},
    ...     l_max=2
    ... )
    """
    # Check if ticker is valid
    if ticker not in successful_tickers or not successful_tickers[ticker]:
        return None
    
    try:
        # Ensure date_affect is in the index
        if date_affect not in returns_df.index:
            raise ValueError(f"date_affect {date_affect} not in DataFrame index")
        
        # Get indices for working with trading days
        idx_date_affect = returns_df.index.get_loc(date_affect)
        idx_start_market_model = idx_date_affect - market_model_window - market_model_buffer
        idx_end_market_model = idx_date_affect - market_model_buffer
        
        # Ensure indices are within bounds
        if idx_start_market_model < 0:
            raise IndexError(
                f"Market model window starts before data: "
                f"idx_start={idx_start_market_model}, need >= 0"
            )
        if idx_end_market_model > len(returns_df):
            raise IndexError(
                f"Market model window extends beyond data: "
                f"idx_end={idx_end_market_model}, max={len(returns_df)}"
            )
        if idx_date_affect + l_max >= len(returns_df):
            raise IndexError(
                f"Holding period extends beyond data: "
                f"idx={idx_date_affect + l_max}, max={len(returns_df)}"
            )
        
        # Check if required columns exist
        ticker_col = f'r_{ticker}_excess'
        if ticker_col not in returns_df.columns:
            raise ValueError(f"Column {ticker_col} not found in returns_df")
        if 'r_market_excess' not in returns_df.columns:
            raise ValueError("Column 'r_market_excess' not found in returns_df")
        
        # 1) Fit the Market Model using pre-event data
        y = returns_df.iloc[
            idx_start_market_model:idx_end_market_model + 1
        ][ticker_col]
        X = sm.add_constant(
            returns_df.iloc[idx_start_market_model:idx_end_market_model + 1]['r_market_excess']
        )
        
        # Fit OLS regression
        model = sm.OLS(y, X, missing='drop').fit()
        
        # 2) Compute abnormal returns (AR) for all dates
        # AR = r_ticker_excess - β * r_market_excess = α + ε
        alpha = model.params['const']
        predicted_returns = model.predict(
            sm.add_constant(returns_df['r_market_excess'])
        )
        epsilon_t = returns_df[ticker_col] - predicted_returns
        AR_vector = alpha + epsilon_t
        
        # 3) Calculate metrics for each holding period L
        results = []
        
        for L in range(0, l_max + 1):
            # Get abnormal return at day L
            AR = AR_vector.iloc[idx_date_affect + L]
            
            # Calculate cumulative abnormal return over [0, L]
            AR_window = AR_vector.iloc[idx_date_affect:idx_date_affect + L + 1]
            CAR = (1 + AR_window).prod() - 1
            
            # Calculate performance metrics
            μ, σ, SR = np.nan, np.nan, np.nan
            
            if L > 0:
                # Log returns for better statistical properties
                log_returns = np.log(1 + AR_window)
                μ = log_returns.mean()
                σ = log_returns.std(ddof=1)
                
                # Annualized Sharpe ratio
                SR = (μ / σ) * np.sqrt(252) if σ != 0 and not np.isnan(σ) else np.nan
            
            results.append({
                'AR': AR,
                'CAR': CAR,
                'μ': μ,
                'σ': σ,
                'SR': SR
            })
        
        # Convert to DataFrame
        ts_data = pd.DataFrame(results)
        
        return ts_data
    
    except Exception as e:
        print(f'Error for {ticker} on {date_affect}: {e}')
        return None


def process_article_ticker_pair(
    row_idx: int,
    row: pd.Series,
    returns_df: pd.DataFrame,
    successful_tickers: Dict[str, bool],
    l_max: int = 260,
    market_model_window: int = 100,
    market_model_buffer: int = 10
) -> Tuple[int, Optional[pd.DataFrame]]:
    """Process a single article-ticker pair for parallel computation.
    
    Parameters
    ----------
    row_idx : int
        Index of the row in the articles DataFrame.
    row : pd.Series
        Row containing 'tickers' and 'date_affect' columns.
    returns_df : pd.DataFrame
        DataFrame with returns data.
    successful_tickers : Dict[str, bool]
        Dictionary indicating which tickers have complete data.
    l_max : int, default=260
        Maximum holding period in trading days.
    market_model_window : int, default=100
        Number of trading days for market model estimation.
    market_model_buffer : int, default=10
        Buffer days before event date.
    
    Returns
    -------
    Tuple[int, Optional[pd.DataFrame]]
        Tuple of (row_idx, trading_strategy_data).
    """
    ticker = row['tickers']
    date_affect = row['date_affect']
    
    try:
        ts_data = calculate_trading_strategy_data(
            ticker=ticker,
            date_affect=date_affect,
            returns_df=returns_df,
            successful_tickers=successful_tickers,
            l_max=l_max,
            market_model_window=market_model_window,
            market_model_buffer=market_model_buffer
        )
        return row_idx, ts_data
    except Exception as e:
        print(f'Error processing {ticker} on {date_affect}: {e}')
        return row_idx, None


def calculate_average_metrics_by_group(
    articles_df: pd.DataFrame,
    ts_dict: Dict[int, pd.DataFrame],
    group_columns: list,
    l_value: int
) -> Dict[Tuple, Dict[str, float]]:
    """Calculate average metrics (CAR, SR) by group.
    
    Parameters
    ----------
    articles_df : pd.DataFrame
        DataFrame containing articles with group columns.
    ts_dict : Dict[int, pd.DataFrame]
        Dictionary mapping article indices to trading strategy data.
    group_columns : list
        List of column names to group by (e.g., ['split', 'cluster']).
    l_value : int
        Holding period L to extract metrics for.
    
    Returns
    -------
    Dict[Tuple, Dict[str, float]]
        Dictionary mapping group keys to average metrics.
        Example: {('Train', 0): {'CAR': 0.05, 'SR': 1.2, 'count': 10}}
    """
    # Initialize accumulator
    group_metrics: Dict[Tuple, Dict[str, list]] = {}
    
    for idx, row in articles_df.iterrows():
        ts_data = ts_dict.get(idx)
        
        if ts_data is None or not isinstance(ts_data, pd.DataFrame):
            continue
        
        if len(ts_data) <= l_value:
            continue
        
        # Get group key
        group_key = tuple(row[col] for col in group_columns)
        
        # Initialize group if not exists
        if group_key not in group_metrics:
            group_metrics[group_key] = {
                'CAR': [],
                'SR': [],
            }
        
        # Append metrics
        car_value = ts_data.loc[l_value, 'CAR']
        sr_value = ts_data.loc[l_value, 'SR']
        
        if not np.isnan(car_value):
            group_metrics[group_key]['CAR'].append(car_value)
        if not np.isnan(sr_value):
            group_metrics[group_key]['SR'].append(sr_value)
    
    # Calculate averages
    avg_metrics = {}
    for group_key, metrics in group_metrics.items():
        avg_metrics[group_key] = {
            'avg_CAR': np.mean(metrics['CAR']) if metrics['CAR'] else np.nan,
            'avg_SR': np.mean(metrics['SR']) if metrics['SR'] else np.nan,
            'count': len(metrics['SR'])
        }
    
    return avg_metrics
