# src/pipeline_duckdb/preprocessing_filters.py

def get_filters_query(source_view: str, target_view: str = "filtered_data") -> str:
    """
    Generate SQL query to apply preprocessing filters.
    
    Filters:
    - prtPrice > 0
    - prtSize > 0
    - uBid > 0.1
    - spread >= 0
    - BIDabove >= 0
    - ASKbelow <= 0
    """
    return f"""
        CREATE OR REPLACE VIEW {target_view} AS
        SELECT * 
        FROM {source_view}
        WHERE prtPrice > 0
          AND prtSize > 0
          AND uBid > 0.1
          AND spread >= 0
          AND BIDabove >= 0
          AND ASKbelow <= 0
    """
