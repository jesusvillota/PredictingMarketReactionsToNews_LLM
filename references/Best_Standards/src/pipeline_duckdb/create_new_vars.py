# src/pipeline_duckdb/create_new_vars.py

from pathlib import Path

def get_dst_csv_path() -> str:
    """Get the absolute path to the timechange_iso.csv file."""
    # Assume the CSV is in the project root (2 levels up from src/pipeline_duckdb/)
    project_root = Path(__file__).resolve().parents[2]
    dst_csv_path = project_root / "timechange_iso.csv"
    return str(dst_csv_path)


def get_create_vars_1_query(source_view: str, target_view: str = "vars_1_data") -> str:
    """
    Generate SQL query to create derived variables (step 1).
    
    Creates: spread, BIDabove, ASKbelow, midpointNBBO, midpointExch, moneyness,
    timestamp_ny, expiration, timestamp_ny_round3, time_to_maturity, dte,
    midpriceM1, midpriceM10, impactPriceM1, impactPriceM10
    
    Note: timestamp_ny is created by converting timestamp (UTC) to New York time,
    accounting for Daylight Saving Time (EDT: UTC-4, EST: UTC-5).
    """
    dst_csv_path = get_dst_csv_path()
    
    return f"""
        CREATE OR REPLACE VIEW {target_view} AS
        WITH dst_transitions AS (
            SELECT 
                year,
                -- Convert local NY time to UTC for comparison
                -- Summer starts at 2 AM EST (which is UTC-5), so add 5 hours to get UTC
                CAST(summer AS TIMESTAMP) + INTERVAL '5 hours' AS summer_start_utc,
                -- Winter starts at 2 AM EDT (which is UTC-4), so add 4 hours to get UTC
                CAST(winter AS TIMESTAMP) + INTERVAL '4 hours' AS winter_start_utc
            FROM read_csv_auto('{dst_csv_path}')
        )
        SELECT 
            src.*,
            -- Basic spread calculations
            oAsk - oBid AS spread,
            prtPrice - oBid + (oAsk - oBid) AS BIDabove,
            prtPrice - oAsk - (oAsk - oBid) AS ASKbelow,
            
            -- Midpoint calculations
            (oBid + oAsk) / 2.0 AS midpointNBBO,
            (ebid + eask) / 2.0 AS midpointExch,
            
            -- Moneyness
            okey_xx / uPrc AS moneyness,
            
            -- Timestamp conversion from UTC to New York time with DST handling
            CASE 
                WHEN CAST(src.timestamp AS TIMESTAMP) >= dst.summer_start_utc 
                     AND CAST(src.timestamp AS TIMESTAMP) < dst.winter_start_utc 
                THEN CAST(src.timestamp AS TIMESTAMP) - INTERVAL '4 hours'  -- EDT (UTC-4)
                ELSE CAST(src.timestamp AS TIMESTAMP) - INTERVAL '5 hours'  -- EST (UTC-5)
            END AS timestamp_ny,
            
            -- New York time from Chicago time (simpler method for validation)
            CAST(src.timestamp_cst AS TIMESTAMP) + INTERVAL '1 hour' AS timestamp_ny_2,
            
            -- Expiration date from components
            -- MAKE_DATE(
            --     CAST(okey_yr AS INTEGER),
            --     CAST(okey_mn AS INTEGER),
            --     CAST(okey_dy AS INTEGER)
            -- ) AS expiration,
            
            MAKE_TIMESTAMP(
                CAST(okey_yr AS BIGINT),
                CAST(okey_mn AS BIGINT),
                CAST(okey_dy AS BIGINT),
                23, -- hour
                59, -- minute
                59  -- second
            ) AS expiration,
            
            -- Midprice calculations for impact
            (oBidM1 + oAskM1) / 2.0 AS midpriceM1,
            (oBidM10 + oAskM10) / 2.0 AS midpriceM10
        FROM {source_view} AS src
        LEFT JOIN dst_transitions AS dst 
            ON YEAR(CAST(src.timestamp AS TIMESTAMP)) = dst.year
    """


def get_create_vars_1_step2_query(source_view: str, target_view: str = "vars_1_complete") -> str:
    """
    Generate SQL query to create derived variables that depend on step 1 outputs.
    
    Creates: timestamp_ny_round3, timestamp_round3, time_to_maturity, dte, impactPriceM1, impactPriceM10
    """
    return f"""
        CREATE OR REPLACE VIEW {target_view} AS
        SELECT *,
            -- Round NY timestamp to milliseconds (mathematical rounding by adding 0.5ms then truncating)
            DATE_TRUNC('millisecond', timestamp_ny + INTERVAL '0.0005 seconds') AS timestamp_ny_round3,
            
            -- Round UTC timestamp to milliseconds (mathematical rounding by adding 0.5ms then truncating)
            DATE_TRUNC('millisecond', CAST(timestamp AS TIMESTAMP) + INTERVAL '0.0005 seconds') AS timestamp_round3,
            
            -- Time to maturity (using UTC timestamp for consistency)
            expiration - CAST(timestamp_ny AS TIMESTAMP) AS time_to_maturity,
            
            -- Days to expiration (using UTC timestamp for date calculation)
            DATE_DIFF('day', CAST(timestamp AS DATE), expiration) AS dte,
            
            -- Impact price calculations
            (midpriceM1 - prtPrice) / prtPrice AS impactPriceM1,
            (midpriceM10 - prtPrice) / prtPrice AS impactPriceM10
        FROM {source_view}
    """


def get_create_vars_2_query(source_view: str, target_view: str = "vars_2_data") -> str:
    """
    Generate SQL query to create additional derived variables (step 2).
    
    Creates: intrinsic_value, trade_size_dollar, quoted_spread, relative_spread,
    leverage, notional_value
    
    Note: This should be called after group_fragmented_trades since it uses prtSize_agg
    """
    return f"""
        CREATE OR REPLACE VIEW {target_view} AS
        SELECT *,
            -- Intrinsic value (different for Calls vs Puts)
            CASE 
                WHEN okey_cp = 'Call' THEN GREATEST(prtPrice - okey_xx, 0)
                WHEN okey_cp = 'Put' THEN GREATEST(okey_xx - prtPrice, 0)
                ELSE 0
            END AS intrinsic_value,
            
            -- Trade size in dollars (assuming prtSize_agg exists from grouping)
            prtPrice * prtSize_agg * 100 AS trade_size_dollar,
            
            -- Spread calculations
            (oAsk - oBid) / midpointNBBO AS quoted_spread,
            POW(ABS(midpointNBBO - prtPrice), 2) / midpointNBBO AS relative_spread,
            
            -- Leverage
            (prtDe * uPrc) / midpointNBBO AS leverage
        FROM {source_view}
    """


def get_create_vars_2_step2_query(source_view: str, target_view: str = "vars_2_complete") -> str:
    """
    Generate SQL query for variables that depend on vars_2 outputs.
    
    Creates: notional_value
    """
    return f"""
        CREATE OR REPLACE VIEW {target_view} AS
        SELECT *,
            -- Notional value (depends on intrinsic_value)
            intrinsic_value * prtSize_agg * 100 AS notional_value
        FROM {source_view}
    """
