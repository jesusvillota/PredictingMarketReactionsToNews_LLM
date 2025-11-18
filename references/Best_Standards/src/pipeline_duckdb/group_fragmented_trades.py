# src/pipeline_duckdb/group_fragmented_trades.py

def get_group_fragmented_query(source_view: str, target_view: str = "grouped_data") -> str:
    """
    Generate SQL query to group fragmented trades using window functions.
    
    Groups by: okey_tk, okey_xx, okey_cp, uBid, uAsk, uPrc, prtExch, prtPrice,
               prtType, timestamp_round3, tradingSession
    
    Adds: fragment_count, prtSize_agg
    Deduplicates to one row per group
    """
    
    grouping_cols = [
        "okey_tk", 
        "okey_xx", 
        "okey_cp",
        "uBid",
        "uAsk", 
        "uPrc",
        "prtExch",
        "prtPrice", 
        "prtType", 
        "timestamp_round3",
        "tradingSession"
    ]
    
    partition_clause = ", ".join(grouping_cols)
    
    # Add deterministic tie-breakers to the ORDER BY clause
    # Using multiple columns ensures a stable, repeatable sort order
    # This prevents duplicates when rows in the same partition have identical prtSize values
    return f"""
        CREATE OR REPLACE VIEW {target_view} AS
        SELECT *,
               COUNT(*) OVER (PARTITION BY {partition_clause}) as fragment_count,
               SUM(prtSize) OVER (PARTITION BY {partition_clause}) as prtSize_agg
        FROM {source_view}
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY {partition_clause} 
            ORDER BY prtSize, okey_tk, okey_xx, timestamp_round3
        ) = 1
    """
