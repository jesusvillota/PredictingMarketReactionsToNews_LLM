# src/trades_new/classify.py
from pathlib import Path
import dask.dataframe as dd
import numpy as np
import json
from src.config.logger import get_logger
from datetime import date, datetime
from typing import Set

def _load_early_close_dates(path: Path) -> Set[date]:
    with path.open("r") as f:
        return {
            datetime.strptime(line.strip(), "%Y-%m-%d").date()
            for line in f
            if line.strip()
        }

EARLY_CLOSE_DATES = _load_early_close_dates(Path("src/utils/early_closing/dates.txt"))


def classify_trades(ddf: dd.DataFrame) -> dd.DataFrame:
    logger = get_logger(__name__)
    logger.debug("Starting trades classification")
    
    cols_to_drop = [
        col for col in [
            'moneyness_class', 'moneyness_class_delta', 'moneyness_class_ratio', 
            'trading_hours_class', 'ticker_class', 'buy_sell_class',
            'bid_ask_proximity', 'moment_of_the_day', 'trade_type', 'time_to_expiry'
        ] if col in ddf.columns
    ]
    if cols_to_drop:
        logger.debug(f"Dropping existing columns: {cols_to_drop}")
        ddf = ddf.drop(columns=cols_to_drop)
    
    # Load equity tickers list
    try:
        with open("src/utils/tickers/EQTS.json", "r") as f:
            eqts_list = json.load(f)
        logger.debug(f"Loaded {len(eqts_list)} equity tickers from EQTS.json")
        eqts_set = set(eqts_list)  # Convert to set for faster lookup
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load equity tickers list: {e}")
        eqts_set = set()  # Empty set as fallback
    
    def classify_partition(df):
        df = df.copy()
        
        #-------- Moneyness classification (Method 1: Delta-based) --------#
        # logger.debug("Classifying by moneyness using delta")
        df['moneyness_class_delta'] = 'Unknown'
        # Delta-based classification: ATM (0.35-0.65), ITM (>0.65), OTM (<0.35)
        abs_delta = df['prtDe'].abs() # Use absolute value of delta for classification
        
        delta_atm = (abs_delta >= 0.35) & (abs_delta <= 0.65)
        delta_itm = abs_delta > 0.65
        delta_otm = abs_delta < 0.35
        
        delta_conditions = [delta_atm, delta_itm, delta_otm]
        delta_choices = ['ATM', 'ITM', 'OTM']
        df['moneyness_class_delta'] = np.select(condlist=delta_conditions, 
                                                 choicelist=delta_choices, 
                                                 default='Unknown')

        #-------- Moneyness classification (Method 2: Strike/Underlying ratio) --------#
        # logger.debug("Classifying by moneyness using strike/underlying ratio")
        df['moneyness_class_ratio'] = 'Unknown'
        # ATM: 0.9 <= moneyness <= 1.2
        ratio_atm = (df['moneyness'] >= 0.9) & (df['moneyness'] <= 1.2)
        
        # Calls: ITM when moneyness < 0.9, OTM when moneyness > 1.2
        call_itm = (df['okey_cp'] == 'Call') & (df['moneyness'] < 0.9)
        call_otm = (df['okey_cp'] == 'Call') & (df['moneyness'] > 1.2)
        
        # Puts: ITM when moneyness > 1.2, OTM when moneyness < 0.9
        put_itm = (df['okey_cp'] == 'Put') & (df['moneyness'] > 1.2)
        put_otm = (df['okey_cp'] == 'Put') & (df['moneyness'] < 0.9)
        
        ratio_conditions = [ratio_atm, call_itm, call_otm, put_itm, put_otm]
        ratio_choices = ['ATM', 'ITM', 'OTM', 'ITM', 'OTM']
        df['moneyness_class_ratio'] = np.select(condlist=ratio_conditions, 
                                                 choicelist=ratio_choices, 
                                                 default='Unknown')
        
        #-------- Trading hours classification --------#
        # logger.debug("Classifying by trading hours")
        df['trading_hours_class'] = 'Unknown'
        
        ts = df["timestamp_ny"]
        date = ts.dt.date
        hour = ts.dt.hour
        minute = ts.dt.minute
        is_early = date.isin(EARLY_CLOSE_DATES)

        # Convert time to minutes from midnight for easier comparison
        time_in_minutes = hour * 60 + minute
        
        # Define time ranges in minutes
        market_open = 9 * 60 + 30  # 9:30 AM = 570 minutes
        market_close_normal = 16 * 60  # 4:00 PM = 960 minutes
        market_close_early = 13 * 60   # 1:00 PM = 780 minutes
        
        # Determine market close time based on early close dates
        market_close = np.where(is_early.to_numpy(), market_close_early, market_close_normal)
        
        # Classify using vectorized conditions
        before = time_in_minutes < market_open
        market = (time_in_minutes >= market_open) & (time_in_minutes < market_close)
        
        # Default to AfterMarket when not before or market
        df['trading_hours_class'] = np.select(
            [before, market],
            ['BeforeMarket', 'MarketHours'],
            default='AfterMarket'
        )

        #-------- Moment of the day classification --------#
        # logger.debug("Classifying by moment of the day")
        df['moment_of_the_day'] = "overnight"
        
        # ts = df["timestamp_ny"]
        # date = ts.dt.date
        # hour = ts.dt.hour
        # minute = ts.dt.minute
        # is_early = date.isin(EARLY_CLOSE_DATES)
        
        # Convert time to minutes from midnight for easier comparison
        # time_in_minutes = hour * 60 + minute
        
        # Define time ranges in minutes
        # 9:30 = 570 minutes, 11:00 = 660 minutes, 13:00 = 780 minutes, 16:00 = 960 minutes
        morning_start = 9 * 60 + 30  # 9:30 AM
        midday_start = 11 * 60       # 11:00 AM
        afternoon_start = 13 * 60    # 1:00 PM
        # market_close_normal = 16 * 60  # 4:00 PM
        # market_close_early = 13 * 60   # 1:00 PM (early close days)
        
        # Determine market close time based on early close dates
        # market_close = np.where(is_early.to_numpy(), market_close_early, market_close_normal)
        
        # Apply classification logic accounting for early close dates
        morning = (time_in_minutes >= morning_start) & (time_in_minutes < midday_start)
        midday = (time_in_minutes >= midday_start) & (time_in_minutes < afternoon_start)
        # On early close days, afternoon period doesn't exist (market closes at 13:00)
        # On normal days, afternoon is from 13:00 to 16:00
        afternoon = (~is_early.to_numpy()) & (time_in_minutes >= afternoon_start) & (time_in_minutes < market_close)
        
        moment_conditions = [morning, midday, afternoon]
        moment_choices = ['morning', 'midday', 'afternoon']
        df['moment_of_the_day'] = np.select(condlist=moment_conditions,
                                           choicelist=moment_choices,
                                           default='overnight')

        #-------- Ticker classification --------#
        # logger.debug("Classifying by ticker type")
        df['ticker_class'] = np.where(df['okey_tk'].isin(eqts_set), 'Equity', 'Other')

        #-------- Buy/Sell classification --------#
        # logger.debug("Classifying by buy/sell")
        df['buy_sell_class'] = "Unknown"
        buy_nbbo = (df['prtPrice'] != df['midpointNBBO']) & (df['prtPrice'] > df['midpointNBBO'])
        sell_nbbo = (df['prtPrice'] != df['midpointNBBO']) & (df['prtPrice'] < df['midpointNBBO'])
        buy_exch = (df['prtPrice'] == df['midpointNBBO']) & (df['prtPrice'] > df['midpointExch'])
        sell_exch = (df['prtPrice'] == df['midpointNBBO']) & (df['prtPrice'] < df['midpointExch'])
        buysell_conditions = [buy_nbbo, sell_nbbo, buy_exch, sell_exch]
        buysell_choices = ['Buy', 'Sell', 'Buy', 'Sell']
        df['buy_sell_class'] = np.select(condlist=buysell_conditions, 
                                         choicelist=buysell_choices, 
                                         default='Midpoint')

        #-------- Bid/Ask proximity classification --------#
        # logger.debug("Classifying by proximity to bid/ask")
        df['bid_ask_proximity'] = "Unknown"
        
        # Calculate absolute distances from price to bid and ask
        dist_to_bid = np.abs(df['prtPrice'] - df['oBid'])
        dist_to_ask = np.abs(df['prtPrice'] - df['oAsk'])
        
        # Apply classification logic
        closer_to_bid = dist_to_bid < dist_to_ask
        closer_to_ask = dist_to_bid > dist_to_ask  
        same_distance = dist_to_bid == dist_to_ask
        
        proximity_conditions = [closer_to_bid, same_distance, closer_to_ask]
        proximity_choices = ['closer_to_bid', 'same_distance', 'closer_to_ask']
        df['bid_ask_proximity'] = np.select(condlist=proximity_conditions,
                                           choicelist=proximity_choices,
                                           default='Unknown')

        #-------- Trade type classification --------#
        # logger.debug("Classifying by trade type")
        df['trade_type'] = np.where(df['prtType'] >= 102, 'complex', 'simple')

        #-------- Time to expiry classification --------#
        # logger.debug("Classifying by time to expiry")
        df['time_to_expiry'] = 'Unknown'
        days_to_expiry = df['dte']
        # Define conditions for time to expiry classification
        less_than_week = days_to_expiry < 7
        one_to_two_weeks = (days_to_expiry >= 7) & (days_to_expiry < 14)
        two_to_four_weeks = (days_to_expiry >= 14) & (days_to_expiry < 28)
        one_to_three_months = (days_to_expiry >= 28) & (days_to_expiry < 91)
        three_to_twelve_months = (days_to_expiry >= 91) & (days_to_expiry < 365)
        over_a_year = days_to_expiry >= 365
        
        expiry_conditions = [less_than_week, one_to_two_weeks, two_to_four_weeks, 
                            one_to_three_months, three_to_twelve_months, over_a_year]
        expiry_choices = ['less than a week', '1-2 weeks', '2-4 weeks', 
                         '1-3 months', '3-12 months', 'over a year']
        df['time_to_expiry'] = np.select(condlist=expiry_conditions,
                                        choicelist=expiry_choices,
                                        default='Unknown')

        return df
    
    # Log the classification steps at the main function level
    logger.debug("Applying delta-based moneyness classification")
    logger.debug("Applying ratio-based moneyness classification")
    logger.debug("Applying trading hours classification") 
    logger.debug("Applying ticker type classification")
    logger.debug("Applying buy/sell classification")
    logger.debug("Applying bid/ask proximity classification")
    logger.debug("Applying moment of the day classification")
    logger.debug("Applying trade type classification")
    logger.debug("Applying time to expiry classification")
    
    # Single map_partitions for all classifications
    meta = ddf._meta.copy()
    meta['moneyness_class_delta'] = 'Unknown'
    meta['moneyness_class_ratio'] = 'Unknown'
    meta['trading_hours_class'] = 'Unknown'
    meta['moment_of_the_day'] = 'overnight'
    meta['ticker_class'] = 'Other'
    meta['buy_sell_class'] = 'Unknown'
    meta['bid_ask_proximity'] = 'Unknown'
    meta['trade_type'] = 'simple'
    meta['time_to_expiry'] = 'Unknown'
    ddf = ddf.map_partitions(classify_partition, meta=meta)
    
    logger.debug("Trades classification completed successfully")
    return ddf