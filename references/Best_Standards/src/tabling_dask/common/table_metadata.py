"""Metadata registry for table scripts.

Maps table names to their required columns for efficient data loading.
"""

from typing import Dict, List, Optional

# Registry mapping table names to their required columns
# None indicates the table uses a different data source (e.g., COMPLEX_TRADES_PATH)
TABLE_COLUMNS: Dict[str, Optional[List[str]]] = {
    "Table1": ['prtSize_agg', 'fragment_count'],
    "Table2_broad": [
        'prtSize_agg', 'okey_cp', 'trade_type', 'prtPrice', 'moneyness', 
        'leverage', 'quoted_spread', 'relative_spread', 'moment_of_the_day', 
        'moneyness_class_ratio', 'bid_ask_proximity', 'time_to_expiry', 
        'trade_size_dollar', 'notional_value'
    ],
    "Table2_granular": [
        'prtSize_agg', 'okey_cp', 'trade_type', 'prtPrice', 'moneyness', 
        'leverage', 'quoted_spread', 'relative_spread', 'moment_of_the_day', 
        'moneyness_class_ratio', 'bid_ask_proximity', 'time_to_expiry', 
        'trade_size_dollar', 'notional_value'
    ],
    "Table3": ['prtSize_agg', 'fragment_count'],
    "Table4": None,
    "Table5": None,  # Uses COMPLEX_TRADES_PATH, handled separately
    "Table6": ['prtType', 'prtSize_agg'],
    "Table7": ['prtType', 'prtSize_agg'],
    "Table8": ['prtExch', 'prtSize_agg'],
}


def get_required_columns(table_name: str) -> Optional[List[str]]:
    """Get required columns for a table.
    
    Args:
        table_name: Name of the table (e.g., "Table1")
        
    Returns:
        List of required column names, or None if table uses different data source
    """
    return TABLE_COLUMNS.get(table_name)


def get_column_union(table_names: List[str]) -> List[str]:
    """Get union of all required columns for a list of tables.
    
    Args:
        table_names: List of table names
        
    Returns:
        Sorted list of unique column names (excluding None entries)
    """
    columns_set = set()
    for name in table_names:
        cols = get_required_columns(name)
        if cols is not None:
            columns_set.update(cols)
    return sorted(list(columns_set))

