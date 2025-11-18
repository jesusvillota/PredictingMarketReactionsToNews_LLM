# tickers/get_ticker_names.py
# run this file by running in the terminal the command below:
# poetry run python tickers/get_ticker_names.py
import pandas as pd
import json
from pathlib import Path

if __name__ == "__main__":    
    # Load the CSV file containing ticker data
    tickers_df = pd.read_csv("/Users/jesusvillotamiranda/Desktop/LOCAL_DATASETS/WHALES/stocks_sample.csv")

    # Create directory for storing ticker lists
    ticker_path = Path("tickers/")
    ticker_path.mkdir(parents=True, exist_ok=True)

    # List of equities: SHRCD != 73
    eqts_list = tickers_df[tickers_df["SHRCD"] != 73]["TICKER"].unique().tolist()
    with open(ticker_path / 'EQTS.json', 'w') as f:
        json.dump(eqts_list, f)

    # List of ETFs: SHRCD == 73
    etfs_list = tickers_df[tickers_df["SHRCD"] == 73]["TICKER"].unique().tolist()
    with open(ticker_path / 'ETFS.json', 'w') as f:
        json.dump(etfs_list, f)