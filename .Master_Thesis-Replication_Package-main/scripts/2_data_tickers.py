"""
Script converted from notebook: 2_data_tickers.ipynb
Original notebook: 2_data_tickers
"""

# <!-- Container with white background -->
# <div style="background-color: white; padding: 20px; border-radius: 10px;">

#   <!-- Name in bold and approximate LaTeX font style with the logo blue color -->
#   <h1 style="font-family: 'Times New Roman', Times, serif; font-weight: bold; color: #38549c; text-align: center;">
#     Jesus Villota Miranda
#   </h1>

#   <!-- Project name in similar style with the logo blue color in italics -->
#   <h2 style="font-family: 'Times New Roman', Times, serif; color: #38549c; text-align: center; font-style: italic;">
#     Predicting Market Reactions to News: An LLM-Based Approach Using Spanish Business Articles
#   </h2>

#   <!-- CEMFI logo centered -->
#   <div style="text-align: center; margin-bottom: 40px;">
#     <img src="https://www.cemfi.es/images/Logo-Azul.png" alt="CEMFI Logo" style="width:200px;">
#   </div>

#   <!-- Catchy message about authorship -->
#   <p style="font-family: 'Times New Roman', Times, serif; color: #38549c; text-align: center; font-size: 1.2em;">
#     All code and work associated with this project are solely created and authored by Jesus Villota Miranda. © 2024
#   </p>

#   <!-- Contact information with logos -->
#   <p style="font-family: 'Times New Roman', Times, serif; color: #38549c; text-align: center; font-size: 1em;">
#     Contact:
#     <a href="mailto:jesus.villota@cemfi.edu.es" style="color: #38549c;">
#       <img src="https://www.logolynx.com/images/logolynx/64/64319177556c729f1806922bcd3adef5.png" alt="Email Logo" style="width: 20px; vertical-align: middle;">
#       jesus.villota@cemfi.edu.es
#     </a> |
#     <a href="https://www.linkedin.com/in/jesusvillotamiranda/" target="_blank" style="color: #38549c;">
#       <img src="https://1.bp.blogspot.com/-onvhHUdW1Us/YI52e9j4eKI/AAAAAAAAE4c/6s9wzOpIDYcAo4YmTX1Qg51OlwMFmilFACLcBGAsYHQ/s1600/Logo%2BLinkedin.png" alt="LinkedIn Logo" style="width: 20px; vertical-align: middle;">
#       LinkedIn
#     </a>
#   </p>

# </div>


# Preamble

# Packages

# %reset -f  # Jupyter magic command removed

import csv
import os
import pandas as pd
import numpy as np

from datetime import timedelta

import ast
import yfinance as yf
# import statsmodels.api as sm

from joblib import Parallel, delayed
import multiprocessing

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance.utils") # Suppress specific FutureWarning from yfinance

# Directories

import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_project_structure(base_path, directories):
    for _, directory in directories.items():
        dir_path = os.path.join(base_path, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

if __name__ == "__main__":
    base_path = os.path.dirname(os.getcwd())
    config_path = os.path.join(base_path, 'config.yaml')
    config = load_config(config_path)
    create_project_structure(base_path, config['directories'])

path_raw_data       = os.path.join(base_path, config['directories']['raw_data'])
path_processed_data = os.path.join(base_path, config['directories']['processed_data'])

# Functions

# **Parallelized** functions to obtain ticker data

def fetch_ticker_data(ticker, start_date, end_date, rf_data):
    """Helper function to fetch and process data for a single ticker."""
    warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance.utils") # Suppress specific FutureWarning from yfinance
    try:
        print(f"Downloading data for ticker: {ticker}")
        prices = yf.download(ticker, start=start_date, end=end_date)
        prices.index = prices.index.date
        if not prices.empty:
            r_ticker = prices['Adj Close'].pct_change().dropna()
            r_ticker_excess = r_ticker - rf_data.shift(1)
            return ticker, r_ticker, r_ticker_excess, None
        else:
            print(f"No data found for ticker: {ticker}")
            return ticker, None, None, "No data"
    except Exception as e:
        print(f"Failed to download data for ticker: {ticker}. Error: {e}")
        return ticker, None, None, str(e)

#==============================================================================================================================

def get_stock_data(df: pd.DataFrame, r_data: pd.DataFrame) -> tuple:
    """Fetches stock data and calculates returns.
    Args:
        df (pd.DataFrame): DataFrame with a 'tickers' column.
        r_data (pd.DataFrame): Contains columns ['rf', 'r_market', 'r_market_excess'] and index with dates.
    Returns:
        tuple: Updated r_data DataFrame, list of successful tickers, list of failed tickers.
    """
    r_data      = r_data.copy()
    tickers     = df['tickers'].unique()
    start_date  = r_data.index[0] - timedelta(days=1)
    end_date    = r_data.index[-1] + timedelta(days=1)

    # Parallel processing using Joblib
    results = Parallel(n_jobs=-1)(delayed(fetch_ticker_data)(ticker, start_date, end_date, r_data['rf']) for ticker in tickers)

    successful_tickers = []
    failed_tickers = []

    for ticker, r_ticker, r_ticker_excess, error in results:
        if error is None:
            r_data[f'r_{ticker}'] = r_ticker
            r_data[f'r_{ticker}_excess'] = r_ticker_excess
            successful_tickers.append(ticker)
        else:
            failed_tickers.append(ticker)

    return r_data, successful_tickers, failed_tickers

# 
# ---
# ---

# Return data: Risk free (`rf`) & IBEX 35 (`r_market`)

#### Risk-free: €STR #### 
ESTR = pd.read_csv(os.path.join(path_raw_data, 'ESTR.csv'),index_col=0, parse_dates=True)
ESTR.index.names = ['datetime']
ESTR.drop(columns='TIME PERIOD', inplace=True)
ESTR.columns = ['rf']
ESTR['rf'] = ESTR['rf'] / 100
ESTR['rf'] = (1 + ESTR['rf'])**(1/252) - 1  # convert to daily rate 
first_date_ESTR_data = ESTR.index[0].date()
last_date_ESTR_data = ESTR.index[-1].date()

#### Market index: IBEX 35 ####
IBEX = yf.download('^IBEX', 
                start=first_date_ESTR_data - timedelta(days=1), 
                end=last_date_ESTR_data + timedelta(days=1))
IBEX['r_market'] = IBEX['Adj Close'].pct_change()
IBEX.index.names = ['datetime']
IBEX = IBEX['r_market'].dropna().to_frame()

#### Joining both series into `r_data` to ensure timeline conformability ####
r_data = ESTR.join(IBEX, how='inner')
r_data['r_market_excess'] = r_data['r_market'] - r_data['rf']
del first_date_ESTR_data, last_date_ESTR_data, ESTR, IBEX

#### Extract the trading days #####
trading_days = [dt.date() for dt in r_data.index]
r_data.index = trading_days

r_data

# 
# ---
# ---

# **"Exploded" data**: $~\mathcal B:=\{(i,j) \mid i\in\mathcal D ~\wedge~j\in\mathcal F^i\}$

def get_df(model):
#==============================================================================================================================
    if model == 'KMeans':

        # Load the data
        D = pd.read_csv(os.path.join(path_processed_data, 'D.csv'))

        # Convert the columns to the correct data types
        D['publ_datetime']  = pd.to_datetime(D['publ_datetime'])        
        D['tickers'] = D['tickers'].apply(lambda x: ast.literal_eval(x))

        # Explode rows with multiple tickers and reset index
        df = D.explode('tickers').reset_index(drop=True) # Expand rows with multiple tickers and reset index

    elif model == 'LLAMA':
        df = pd.read_csv(os.path.join(path_raw_data, 'LLAMA_parsed_news.csv'))
#==============================================================================================================================
    return df

# 
# ---
# ---

# **Complete return data**: $~\mathcal R:= \left\{\{r^f_d\}_{d\in\tilde{\mathfrak{d}}}, \{r^M_d\}_{d\in\tilde{\mathfrak{d}}}, \{\{\{r^j_d\}_{d\in\tilde{\mathfrak{d}}}\}_{j\in \mathcal F^i}\}_{i\in\mathcal D}\right\}$

# - Completes the data from `r_data` = [rf | r_market | r_market_excess]
# - Output: `r_data_comlete` = [rf | r_market | r_market_excess | r_{ticker.MC} | r_{ticker.MC}_excess ] for ticker.MC in tickers.MC


# Running this codes takes $\approx 1$ minute

for model in ['KMeans', 'LLAMA']:

    # Load the data
    df = get_df(model)

    # Fetch stock data
    R, successful_tickers, failed_tickers = get_stock_data(df, r_data)

    # Export R
    R.to_csv(os.path.join(path_processed_data, f'R_{model}.csv')) 

    # Export successful_tickers
    with open(os.path.join(path_processed_data, f'successful_tickers_{model}.txt'), 'w') as f:
        for ticker in successful_tickers:
            f.write(f'{ticker}\n')

    # Export failed_tickers
    with open(os.path.join(path_processed_data, f'failed_tickers_{model}.txt'), 'w') as f:
        for ticker in failed_tickers:
            f.write(f'{ticker}\n')

# # download_R = True
# #==============================================================================================================================

# # if download_R:
#     #=============================== DOWNLOAD ======================================
#     if __name__ == "__main__":
#         R, successful_tickers, failed_tickers = get_stock_data(df, r_data)
#     #================================ EXPORT =======================================
#     # - Export R
#     R.to_csv(os.path.join(Export_Path, f'R_{model}.csv')) 
#     # - Export successful_tickers
#     with open(os.path.join(Export_Path, f'successful_tickers_{model}.txt'), 'w') as f:
#         for ticker in successful_tickers:
#             f.write(f'{ticker}\n')
#     # - Export failed_tickers
#     with open(os.path.join(Export_Path, f'failed_tickers_{model}.txt'), 'w') as f:
#         for ticker in failed_tickers:
#             f.write(f'{ticker}\n')
# # else:
# #     #================================ IMPORT =======================================
# #     # - Load R
# #     R = pd.read_csv(os.path.join(Export_Path, f'R_{model}.csv'), index_col=0, parse_dates=True)
# #     R.index = trading_days
# #     # - Read successful_tickers 
# #     with open(os.path.join(Export_Path, f'successful_tickers_{model}.txt'), 'r') as f:
# #         successful_tickers = f.read().splitlines()
# #     # - Read failed_tickers
# #     with open(os.path.join(Export_Path, f'failed_tickers_{model}.txt'), 'r') as f:
# #         failed_tickers = f.read().splitlines()

# # R.index = trading_days

# 
# ---
# ---
