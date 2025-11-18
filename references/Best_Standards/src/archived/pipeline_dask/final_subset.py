# src/trades_new/final_subset.py
import dask.dataframe as dd

def final_subset(ddf: dd.DataFrame) -> dd.DataFrame:
    """
    Select a final subset of columns from the trades DataFrame.
    """
    selected_columns: list[str] = [
        'okey_tk', 
        'okey_xx', 
        'okey_cp', 
        'tradingSession',
        'prtExch', 
        'prtPrice', 
        'prtType', 
        'prtVolume',
        'ebid', 
        'eask', 
        'prtSide', 
        'oBid', 
        'oAsk', 
        'oBidSz', 
        'oAskSz', 
        'uBid',
        'uAsk', 
        'uPrc', 
        'prtIv', 
        'prtDe', 
        'prtGa', 
        'surfVol',
        'prtProbability', 
        'oBidM1', 
        'oAskM1', 
        'oBidM10', 
        'oAskM10',
        'spread', 
        'midpointNBBO',
        'midpointExch', 
        'moneyness', 
        'timestamp_ny', 
        'expiration', 
        'midpriceM1',
        'midpriceM10', 
        # 'timestamp_ny_round3', 
        'time_to_maturity', 
        'dte',
        'impactPriceM1', 
        'impactPriceM10', 
        # Classifications
        'moneyness_class_delta',
        'moneyness_class_ratio', 
        'trading_hours_class', 
        'ticker_class', 
        'buy_sell_class',
        'bid_ask_proximity',
        'fragment_count', 
        # Created variables
        'prtSize_agg', 
        'unsigned_delta', 
        'signed_delta'
    ]
    return ddf[selected_columns]