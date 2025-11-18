import dask.dataframe as dd

def analyze_volume_distribution(ddf: dd.DataFrame) -> dict:
    """Analyze basic volume distribution characteristics"""
    return {
        'count': len(ddf),
        'volume_stats': ddf['prtSize_agg'].describe().compute(),
        'price_stats': ddf['prtPrice'].describe().compute(),
        'avg_fragment_count': ddf['fragment_count'].mean().compute()
    }

def analyze_by_moneyness(ddf: dd.DataFrame) -> dict:
    """Analyze large trades by moneyness classification"""
    moneyness_analysis = ddf.groupby('moneyness_class').agg({
        'prtSize_agg': ['count', 'mean', 'sum'],
        'prtPrice': 'mean',
        'spread': 'mean',
        'prtIv': 'mean',  # Implied volatility
        'prtDe': 'mean'   # Delta
    }).compute()
    
    return moneyness_analysis


def analyze_market_impact(ddf: dd.DataFrame) -> dict:
    """Analyze market impact using existing variables"""
    # Your pipeline already creates impactPriceM1 and impactPriceM10
    impact_stats = ddf.agg({
        'impactPriceM1': ['mean', 'std', 'count'],
        'impactPriceM10': ['mean', 'std', 'count']
    }).compute()
    
    # Analyze by trade proximity to bid/ask
    bid_ask_analysis = analyze_bid_ask_proximity(ddf)
    
    return {
        'impact_statistics': impact_stats,
        'bid_ask_proximity': bid_ask_analysis
    }

def analyze_bid_ask_proximity(ddf: dd.DataFrame) -> dict:
    """Analyze if trades are closer to bid or ask"""
    # Using your existing midpointNBBO variable
    ddf_with_proximity = ddf.assign(
        closer_to_bid=ddf['prtPrice'] < ddf['midpointNBBO'],
        closer_to_ask=ddf['prtPrice'] > ddf['midpointNBBO'],
        at_midpoint=ddf['prtPrice'] == ddf['midpointNBBO']
    )
    
    proximity_stats = ddf_with_proximity.agg({
        'closer_to_bid': 'sum',
        'closer_to_ask': 'sum', 
        'at_midpoint': 'sum'
    }).compute()
    
    return proximity_stats

def analyze_by_trading_hours(ddf: dd.DataFrame) -> dict:
    """Compare market hours vs after hours trading"""
    # Using your existing trading_hours_class variable
    hours_analysis = ddf.groupby('trading_hours_class').agg({
        'prtSize_agg': ['count', 'mean', 'sum'],
        'prtPrice': 'mean',
        'spread': 'mean',
        'unsigned_delta': 'sum',  # From your delta calculations
        'signed_delta': 'sum'
    }).compute()
    
    return hours_analysis