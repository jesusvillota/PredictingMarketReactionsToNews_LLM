"""Process ticker data and fetch stock returns."""

import pandas as pd
import numpy as np
import ast
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Tuple, List, Optional

import yfinance as yf
from joblib import Parallel, delayed

from src.config import get_paths, get_logger, config_settings

logger = get_logger("data.process_tickers")

# Suppress FutureWarning from yfinance
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance.utils")


def load_risk_free_data(raw_data_path: Path, filename: str = "ESTR.csv") -> pd.DataFrame:
    """
    Load risk-free rate data (€STR).
    
    Args:
        raw_data_path: Path to raw data directory
        filename: Name of the ESTR file
    
    Returns:
        DataFrame with risk-free rate data
    """
    logger.info(f"Loading risk-free rate data from {filename}")
    
    filepath = raw_data_path / filename
    ESTR = pd.read_csv(filepath, index_col=0, parse_dates=True)
    ESTR.index.names = ['datetime']
    
    if 'TIME PERIOD' in ESTR.columns:
        ESTR.drop(columns='TIME PERIOD', inplace=True)
    
    ESTR.columns = ['rf']
    ESTR['rf'] = ESTR['rf'] / 100
    ESTR['rf'] = (1 + ESTR['rf'])**(1/252) - 1  # Convert to daily rate
    
    logger.info(f"Loaded risk-free data from {ESTR.index[0].date()} to {ESTR.index[-1].date()}")
    return ESTR


def load_market_index_data(start_date, end_date, index_symbol: str = "^IBEX") -> pd.DataFrame:
    """
    Load market index data (IBEX 35).
    
    Args:
        start_date: Start date for data
        end_date: End date for data
        index_symbol: Yahoo Finance symbol for the index
    
    Returns:
        DataFrame with market returns
    """
    logger.info(f"Loading market index data ({index_symbol})")
    
    IBEX = yf.download(
        index_symbol,
        start=start_date - timedelta(days=1),
        end=end_date + timedelta(days=1)
    )
    
    IBEX['r_market'] = IBEX['Adj Close'].pct_change()
    IBEX.index.names = ['datetime']
    IBEX = IBEX['r_market'].dropna().to_frame()
    
    logger.info(f"Loaded market index data")
    return IBEX


def prepare_return_data(raw_data_path: Path, index_symbol: str = "^IBEX") -> pd.DataFrame:
    """
    Prepare return data with risk-free rate and market returns.
    
    Args:
        raw_data_path: Path to raw data directory
        index_symbol: Yahoo Finance symbol for market index
    
    Returns:
        DataFrame with risk-free rate, market returns, and excess market returns
    """
    logger.info("Preparing return data")
    
    # Load risk-free data
    ESTR = load_risk_free_data(raw_data_path)
    first_date = ESTR.index[0].date()
    last_date = ESTR.index[-1].date()
    
    # Load market index
    IBEX = load_market_index_data(first_date, last_date, index_symbol)
    
    # Join both series
    r_data = ESTR.join(IBEX, how='inner')
    r_data['r_market_excess'] = r_data['r_market'] - r_data['rf']
    
    # Extract trading days
    trading_days = [dt.date() for dt in r_data.index]
    r_data.index = trading_days
    
    logger.info(f"Prepared return data for {len(r_data)} trading days")
    return r_data


def fetch_ticker_data(ticker: str, start_date, end_date, rf_data: pd.Series) -> Tuple:
    """
    Fetch and process data for a single ticker.
    
    Args:
        ticker: Ticker symbol
        start_date: Start date for data
        end_date: End date for data
        rf_data: Risk-free rate series
    
    Returns:
        Tuple of (ticker, returns, excess_returns, error_message)
    """
    warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance.utils")
    
    try:
        logger.debug(f"Downloading data for ticker: {ticker}")
        prices = yf.download(ticker, start=start_date, end=end_date, progress=False)
        prices.index = prices.index.date
        
        if not prices.empty:
            r_ticker = prices['Adj Close'].pct_change().dropna()
            r_ticker_excess = r_ticker - rf_data.shift(1)
            return ticker, r_ticker, r_ticker_excess, None
        else:
            logger.warning(f"No data found for ticker: {ticker}")
            return ticker, None, None, "No data"
    except Exception as e:
        logger.error(f"Failed to download data for ticker: {ticker}. Error: {e}")
        return ticker, None, None, str(e)


def get_stock_data(
    df: pd.DataFrame, 
    r_data: pd.DataFrame,
    n_jobs: int = -1
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Fetch stock data and calculate returns for all tickers.
    
    Args:
        df: DataFrame with a 'tickers' column containing lists of tickers
        r_data: DataFrame with risk-free rate and market returns
        n_jobs: Number of parallel jobs (-1 for all cores)
    
    Returns:
        Tuple of (updated r_data with stock returns, successful tickers, failed tickers)
    """
    logger.info("Fetching stock data for all tickers")
    
    r_data = r_data.copy()
    
    # Get unique tickers
    # Handle both list and string representations of lists
    all_tickers = []
    for ticker_list in df['tickers']:
        if isinstance(ticker_list, str):
            try:
                ticker_list = ast.literal_eval(ticker_list)
            except:
                continue
        if isinstance(ticker_list, list):
            all_tickers.extend(ticker_list)
        else:
            all_tickers.append(ticker_list)
    
    tickers = list(set(all_tickers))
    
    logger.info(f"Found {len(tickers)} unique tickers")
    
    start_date = r_data.index[0] - timedelta(days=1)
    end_date = r_data.index[-1] + timedelta(days=1)
    
    # Parallel processing
    results = Parallel(n_jobs=n_jobs)(
        delayed(fetch_ticker_data)(ticker, start_date, end_date, r_data['rf']) 
        for ticker in tickers
    )
    
    successful_tickers = []
    failed_tickers = []
    
    for ticker, r_ticker, r_ticker_excess, error in results:
        if error is None:
            r_data[f'r_{ticker}'] = r_ticker
            r_data[f'r_{ticker}_excess'] = r_ticker_excess
            successful_tickers.append(ticker)
        else:
            failed_tickers.append(ticker)
    
    logger.info(f"Successfully fetched data for {len(successful_tickers)} tickers")
    logger.info(f"Failed to fetch data for {len(failed_tickers)} tickers")
    
    return r_data, successful_tickers, failed_tickers


def get_df_for_model(model: str, processed_data_path: Path, raw_data_path: Path) -> pd.DataFrame:
    """
    Get exploded dataframe for a specific model (KMeans or LLAMA).
    
    Args:
        model: Model name ('KMeans' or 'LLAMA')
        processed_data_path: Path to processed data directory
        raw_data_path: Path to raw data directory
    
    Returns:
        Exploded DataFrame with one row per article-ticker pair
    """
    if model == 'KMeans':
        # Load the data
        D = pd.read_csv(processed_data_path / 'D.csv')
        
        # Convert columns to correct data types
        D['publ_datetime'] = pd.to_datetime(D['publ_datetime'])
        D['tickers'] = D['tickers'].apply(lambda x: ast.literal_eval(x))
        
        # Explode rows with multiple tickers
        df = D.explode('tickers').reset_index(drop=True)
        
    elif model == 'LLAMA':
        df = pd.read_csv(raw_data_path / 'LLAMA_parsed_news.csv')
    else:
        raise ValueError(f"Unknown model: {model}. Must be 'KMeans' or 'LLAMA'")
    
    return df


def process_ticker_data(
    model: str,
    raw_data_path: Optional[Path] = None,
    processed_data_path: Optional[Path] = None,
    save_output: bool = True
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Main function to process ticker data and fetch stock returns.
    
    Args:
        model: Model name ('KMeans' or 'LLAMA')
        raw_data_path: Path to raw data directory. If None, uses config default.
        processed_data_path: Path to processed data directory. If None, uses config default.
        save_output: Whether to save outputs to files
    
    Returns:
        Tuple of (return data, successful tickers, failed tickers)
    """
    if raw_data_path is None or processed_data_path is None:
        path_manager = get_paths()
        if raw_data_path is None:
            raw_data_path = path_manager.get_raw_data_path()
        if processed_data_path is None:
            processed_data_path = path_manager.get_processed_data_path()
    
    # Prepare return data
    index_symbol = config_settings.data_config.get("market_index", "^IBEX")
    r_data = prepare_return_data(raw_data_path, index_symbol)
    
    # Get dataframe for model
    df = get_df_for_model(model, processed_data_path, raw_data_path)
    
    # Fetch stock data
    R, successful_tickers, failed_tickers = get_stock_data(df, r_data)
    
    # Save outputs
    if save_output:
        # Export R
        R.to_csv(processed_data_path / f'R_{model}.csv')
        logger.info(f"Saved return data to {processed_data_path / f'R_{model}.csv'}")
        
        # Export successful tickers
        with open(processed_data_path / f'successful_tickers_{model}.txt', 'w') as f:
            for ticker in successful_tickers:
                f.write(f'{ticker}\n')
        
        # Export failed tickers
        with open(processed_data_path / f'failed_tickers_{model}.txt', 'w') as f:
            for ticker in failed_tickers:
                f.write(f'{ticker}\n')
        
        logger.info(f"Saved ticker lists to {processed_data_path}")
    
    return R, successful_tickers, failed_tickers

