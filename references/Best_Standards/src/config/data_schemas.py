DTYPES: dict[str, str] = {
    'okey_at': 'category',
    'okey_ts': 'str',
    'okey_tk': 'str',
    'okey_yr': 'uint16',
    'okey_mn': 'uint8',
    'okey_dy': 'uint8',
    'okey_xx': 'float64',
    'okey_cp': 'category',
    'timestamp': 'str',
    'prtNumber': 'int64',
    'tradingDate': 'str',
    'tradingSession': 'category',
    'ticker_at': 'category',
    'ticker_ts': 'category',
    'ticker_tk': 'str',
    'undSecKey_at': 'str',
    'undSecKey_ts': 'str',
    'undSecKey_tk': 'str',
    'undSecKey_yr': 'int16',
    'undSecKey_mn': 'int16',
    'undSecKey_dy': 'int16',
    'undSecType': 'str',
    'prtExch': 'category',
    'prtSize': 'int64',
    'prtPrice': 'float64',
    'prtType': 'int64',
    'prtOrders': 'int64',
    'prtClusterNum': 'int64',
    'prtClusterSize': 'int64',
    'prtVolume': 'int64',
    'cxlVolume': 'int64',
    'bidCount': 'int64',
    'askCount': 'int64',
    'bidVolume': 'int64',
    'askVolume': 'int64',
    'ebid': 'float64',
    'eask': 'float64',
    'ebsz': 'float64',
    'easz': 'float64',
    'eage': 'float64',
    'prtSide': 'category',
    'prtTimestamp': 'float64',
    'netTimestamp': 'float64',
    'oBid': 'float64',
    'oAsk': 'float64',
    'oBidSz': 'int64',
    'oAskSz': 'int64',
    'oBidEx': 'category',
    'oAskEx': 'category',
    'oBidExSz': 'int64',
    'oAskExSz': 'int64',
    'oBidCnt': 'int64',
    'oAskCnt': 'int64',
    'oBid2': 'float64',
    'oAsk2': 'float64',
    'oBidSz2': 'int64',
    'oAskSz2': 'int64',
    'uBid': 'float64',
    'uAsk': 'float64',
    'uPrc': 'float64',
    'yrs': 'float64',
    'rate': 'float64',
    'sdiv': 'float64',
    'ddiv': 'float64',
    'xDe': 'float64',
    'xAxis': 'float64',
    'prtIv': 'float64',
    'prtDe': 'float64',
    'prtGa': 'float64',
    'prtTh': 'float64',
    'prtVe': 'float64',
    'prtRo': 'float64',
    'calcErr': 'object',
    'surfVol': 'float64',
    'surfOpx': 'float64',
    'surfAtm': 'float64',
    'prtProbability': 'float64',
    'oBidM1': 'float64',
    'oAskM1': 'float64',
    'uBidM1': 'float64',
    'uAskM1': 'float64',
    'uPrcM1': 'float64',
    'sVolM1': 'float64',
    'sOpxM1': 'float64',
    'sDivM1': 'float64',
    'sErrM1': 'object',
    'pnlM1': 'float64',
    'pnlM1Err': 'object',
    'oBidM10': 'float64',
    'oAskM10': 'float64',
    'uBidM10': 'float64',
    'uAskM10': 'float64',
    'uPrcM10': 'float64',
    'sVolM10': 'float64',
    'sOpxM10': 'float64',
    'sDivM10': 'float64',
    'sErrM10': 'object',
    'pnlM10': 'float64',
    'pnlM10Err': 'object',
    'multihedge': 'object',
    'timestamp_cst': 'str',
    'securityID': 'int64'
}


DTYPES_OLD: dict[str, str] = {
    'okey_at': 'category',
    'okey_ts': 'str',
    'okey_tk': 'str',
    'okey_yr': 'int16',
    'okey_mn': 'int16',
    'okey_dy': 'int16',
    'okey_xx': 'float64',
    'okey_cp': 'category',
    'timestamp': 'str',
    'prtNumber': 'int64',
    'tradingDate': 'str',
    'tradingSession': 'str',
    'ticker_at': 'str',
    'ticker_ts': 'str',
    'ticker_tk': 'str',
    'undSecKey_at': 'str',
    'undSecKey_ts': 'str',
    'undSecKey_tk': 'str',
    'undSecKey_yr': 'int16',
    'undSecKey_mn': 'int16',
    'undSecKey_dy': 'int16',
    'undSecType': 'category',
    'prtExch': 'category',
    'prtSize': 'int64',
    'prtPrice': 'float64',
    'prtType': 'int64',
    'prtOrders': 'int64',
    'prtClusterNum': 'int64',
    'prtClusterSize': 'int64',
    'prtVolume': 'int64',
    'cxlVolume': 'int64',
    'bidCount': 'int64',
    'askCount': 'int64',
    'bidVolume': 'int64',
    'askVolume': 'int64',
    'ebid': 'float64',
    'eask': 'float64',
    'ebsz': 'float64',
    'easz': 'float64',
    'eage': 'float64',
    'prtSide': 'category',
    'prtTimestamp': 'float64',
    'netTimestamp': 'float64',
    'oBid': 'float64',
    'oAsk': 'float64',
    'oBidSz': 'int64',
    'oAskSz': 'int64',
    'oBidEx': 'category',
    'oAskEx': 'category',
    'oBidExSz': 'int64',
    'oAskExSz': 'int64',
    'oBidCnt': 'int64',
    'oAskCnt': 'int64',
    'oBid2': 'float64',
    'oAsk2': 'float64',
    'oBidSz2': 'int64',
    'oAskSz2': 'int64',
    'uBid': 'float64',
    'uAsk': 'float64',
    'uPrc': 'float64',
    'yrs': 'float64',
    'rate': 'float64',
    'sdiv': 'float64',
    'ddiv': 'float64',
    'xDe': 'float64',
    'xAxis': 'float64',
    'prtIv': 'float64',
    'prtDe': 'float64',
    'prtGa': 'float64',
    'prtTh': 'float64',
    'prtVe': 'float64',
    'prtRo': 'float64',
    'calcErr': 'object',
    'surfVol': 'float64',
    'surfOpx': 'float64',
    'surfAtm': 'float64',
    'prtProbability': 'float64',
    'oBidM1': 'float64',
    'oAskM1': 'float64',
    'uBidM1': 'float64',
    'uAskM1': 'float64',
    'uPrcM1': 'float64',
    'sVolM1': 'float64',
    'sOpxM1': 'float64',
    'sDivM1': 'float64',
    'sErrM1': 'object',
    'pnlM1': 'float64',
    'pnlM1Err': 'object',
    'oBidM10': 'float64',
    'oAskM10': 'float64',
    'uBidM10': 'float64',
    'uAskM10': 'float64',
    'uPrcM10': 'float64',
    'sVolM10': 'float64',
    'sOpxM10': 'object',
    'sDivM10': 'float64',
    'sErrM10': 'object',
    'pnlM10': 'float64',
    'pnlM10Err': 'object',
    'multihedge': 'object',
    'timestamp_cst': 'str',
    'securityID': 'int64'
}

ALL_COLUMNS: list[str] = [
    *DTYPES.keys()
]

SELECTED_COLUMNS: list[str] = [
        #---- Option identifiers ----#
        "okey_at", # underlying asset type
        "okey_ts", # ticker source
        "okey_tk", # underlying symbol
        "okey_yr", # expiration year
        "okey_mn", # expiration month
        "okey_dy", # expiration day
        "okey_xx", # strike
        "okey_cp", # call/put indicator

        #---- Underlying identifiers ----#
        "ticker_at", # underlying asset type
        "ticker_ts", # underlying ticker source
        "ticker_tk", # underlying ticker
        "uBid", # underlying bid
        "uAsk", # underlying ask
        "uPrc", # underlying price

        #---- Trade Print identifiers ----#
        "prtExch", # exchange where trade/print took place
        "prtSize", # number of contracts in the trade
        "prtPrice", # print price
        "prtType", # trade type (i.e., multi-leg, auction)
        "prtVolume", # volume of contracts traded
        "ebid", # NBBO bid at print time
        "eask", # NBBO ask at print time
        "prtSide", # trade side (buy/sell)

        #---- Market Data (NBBO) ----#
        "oBid", # Option NBBO bid at print time
        "oAsk", # Option NBBO ask at print time
        "oBidSz", # Option NBBO bid size at print time
        "oAskSz", # Option NBBO ask size at print time

        #---- Session and Date identifiers ----#
        "tradingSession", # trading session
        "timestamp_cst",
        
        #---- SpiderRock Analytics ----#
        "yrs", # years to expiration
        "prtIv", # implied volatility at print time
        "prtDe", # delta at print time
        "prtGa", # gamma at print time
        "surfVol", # surface volume at print time
        "prtProbability", # probability of expiring in the money at print time
        
        #---- Post-Print Market Data (1 Minute) ----#
        "oBidM1", # Option NBBO bid 1 minute after print
        "oAskM1", # Option NBBO ask 1 minute after print
        
        #---- Post-Print Market Data (10 Minutes) ----#
        "oBidM10", # Option NBBO bid 10 minutes after print
        "oAskM10", # Option NBBO ask 10 minutes after print
]