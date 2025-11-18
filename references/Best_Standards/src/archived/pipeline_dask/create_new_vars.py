# src/trades_new/create_new_vars.py
import dask.dataframe as dd
import pandas as pd
from src import get_logger

def create_new_vars_1(ddf: dd.DataFrame) -> dd.DataFrame:
    """
    Create useful derived features on the dask DataFrame for trades.
    """
    logger = get_logger(__name__)
    logger.debug("Starting creation of new variables")
    
    ddf: dd.DataFrame = ddf.copy()
    
    # Define all new columns in a dict to add at once
    new_cols: dict[str, dd.Series] = {
        'spread': ddf['oAsk'] - ddf['oBid'],
        'BIDabove': ddf['prtPrice'] - ddf['oBid'] + (ddf['oAsk'] - ddf['oBid']),  # Reuse spread calculation
        'ASKbelow': ddf['prtPrice'] - ddf['oAsk'] - (ddf['oAsk'] - ddf['oBid']),
        'midpointNBBO': (ddf['oBid'] + ddf['oAsk']) / 2,
        'midpointExch': (ddf['ebid'] + ddf['eask']) / 2,
        'moneyness': ddf['okey_xx'] / ddf['uPrc'],
        'timestamp_ny': dd.to_datetime(ddf['timestamp_cst']) + pd.Timedelta(hours=1),
        'expiration': dd.to_datetime(
            ddf['okey_yr'].astype(str) + '-' +
            ddf['okey_mn'].astype(str).str.zfill(2) + '-' +
            ddf['okey_dy'].astype(str).str.zfill(2) + ' 23:59:59.999999'
        ),
        'midpriceM1': (ddf['oBidM1'] + ddf['oAskM1']) / 2,
        'midpriceM10': (ddf['oBidM10'] + ddf['oAskM10']) / 2,
    }
    
    # Derived from new_cols (to avoid intermediate assignments)
    new_cols['timestamp_ny_round3'] = new_cols['timestamp_ny'].dt.round('1ms')
    new_cols['time_to_maturity'] = new_cols['expiration'] - dd.to_datetime(ddf['timestamp_cst'])
    new_cols['dte'] = new_cols['time_to_maturity'].dt.days
    new_cols['impactPriceM1'] = (new_cols['midpriceM1'] - ddf['prtPrice']) / ddf['prtPrice']
    new_cols['impactPriceM10'] = (new_cols['midpriceM10'] - ddf['prtPrice']) / ddf['prtPrice']
    
    # ---- variables from create_new_vars_2 ------------------------------------------------------------------------
    # new_cols['intrinsic_value'] = (
    #     (ddf['prtPrice'] - ddf['okey_xx']).clip(lower=0) * (ddf['okey_cp'] == 'Call') +
    #     (ddf['okey_xx'] - ddf['prtPrice']).clip(lower=0) * (ddf['okey_cp'] == 'Put')
    # )
    # new_cols['trade_size_dollar'] = ddf['prtPrice'] * ddf['prtSize_agg'] * 100
    # new_cols['quoted_spread'] = (ddf["oAsk"] - ddf["oBid"]) / new_cols['midpointNBBO']
    # new_cols['relative_spread'] = abs(new_cols['midpointNBBO'] - ddf['prtPrice'])**2 / new_cols['midpointNBBO']
    # new_cols['leverage'] = (ddf['prtDe']*ddf['uPrc']) / new_cols['midpointNBBO']
    # new_cols['notional_value'] = new_cols['intrinsic_value'] * ddf['prtSize_agg'] * 100
    # -------------------------------------------------------------------------------------------------------------
    
    # Assign all at once
    ddf = ddf.assign(**new_cols)
    
    logger.debug("Successfully created new variables for options trades")
    return ddf


def create_new_vars_2(ddf: dd.DataFrame) -> dd.DataFrame:
    """
    Create additional derived features on the dask DataFrame for trades.
    """
    logger = get_logger(__name__)
    logger.debug("Starting creation of additional new variables")
    
    ddf: dd.DataFrame = ddf.copy()
    
    # Define all new columns in a dict to add at once
    new_cols: dict[str, dd.Series] = {
        'intrinsic_value': (ddf['prtPrice'] - ddf['okey_xx']).clip(lower=0) * (ddf['okey_cp'] == 'Call') +
                           (ddf['okey_xx'] - ddf['prtPrice']).clip(lower=0) * (ddf['okey_cp'] == 'Put'),
        'trade_size_dollar': ddf['prtPrice'] * ddf['prtSize_agg'] * 100,
        'quoted_spread': (ddf["oAsk"] - ddf["oBid"]) / ddf['midpointNBBO'],
        'relative_spread': abs(ddf['midpointNBBO'] - ddf['prtPrice'])**2 / ddf['midpointNBBO'],
        'leverage': (ddf['prtDe']*ddf['uPrc']) / ddf['midpointNBBO'],
        'unsigned_delta': 100 * ddf['prtDe'].abs() * ddf['prtSize_agg'],
        'signed_delta': 100 * ddf['prtDe'].abs() * ddf['prtSize_agg'] * ddf['buy_sell_class'].map({'Buy': 1, 'Sell': -1}, meta=('sign', 'float64')).fillna(0),
    }

    new_cols['notional_value'] = new_cols['intrinsic_value'] * ddf['prtSize_agg'] * 100

    ddf = ddf.assign(**new_cols)
    logger.debug("Successfully created additional new variables")
    return ddf