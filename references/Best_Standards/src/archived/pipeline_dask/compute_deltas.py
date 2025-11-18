# src/deltas/compute_deltas.py
import dask.dataframe as dd
from src import get_logger

def compute_deltas(ddf: dd.DataFrame) -> dd.DataFrame:
    """
    Compute the deltas
    """
    logger = get_logger(__name__)
    logger.debug("Starting computation of deltas")

    sign = ddf['buy_sell_class'].map({'Buy': 1, 'Sell': -1}, meta=('sign', 'float64')).fillna(0)

    new_cols = {
        'unsigned_delta': 100 * ddf['prtDe'].abs() * ddf['prtSize_agg'],
        'signed_delta': 100 * ddf['prtDe'].abs() * ddf['prtSize_agg'] * sign
    }

    ddf = ddf.assign(**new_cols)
    logger.debug("Successfully computed deltas for options trades")
    return ddf