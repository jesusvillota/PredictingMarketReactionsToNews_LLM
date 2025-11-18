# uv run src/mains_duckdb/simple_attention_sentiment_duckdb.py

"""
Compute attention and sentiment measures for whale trades using DuckDB.

Output matches src/mains/simple_attention_sentiment.py but uses DuckDB for speed.

Attention_{i,t} = # whale trades / # total trades, where whale = prtSize_agg >= 200
Sentiment_{i,t} = (# bull - # bear) / (# bull + # bear),
  bull = whale & ((buy & Call) | (sell & Put))
  bear = whale & ((sell & Call) | (buy & Put))
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import gc

from src.config import config_settings, initialize_main
from src.config.config_settings import PROCESSED_PATH, OUTPUT_PATH
from src.config.duckdb_manager import DuckDBManager

# --------------------------------------------------------------------------------------------
# Optional year filter to limit folders scanned under PROCESSED_PATH
TARGET_YEARS: list[int] | None = None # [2021]  # Set to None to process all years
# --------------------------------------------------------------------------------------------
OUTPUT_DIR = OUTPUT_PATH / "_ATTENTION_SENTIMENT_DIRECTION_"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# --------------------------------------------------------------------------------------------


def discover_daily_folders(base: Path, years: list[int] | None) -> list[Path]:
    folders = sorted([d for d in base.iterdir() if d.is_dir()])
    if years is not None:
        folders = [f for f in folders if f.name[:4].isdigit() and int(f.name[:4]) in years]
    return folders

def build_core_select(from_source: str) -> str:
    """Return SQL that aggregates required measures from the given source.
    from_source is a SQL FROM clause content like "trades" or a subselect.
    """
    return f"""
        SELECT
            okey_tk,
            strftime(timestamp_ny, '%Y-%m') AS year_month,
            SUM((prtSize_agg >= 200)::INT) AS whale_count,
            COUNT(*) AS total_count,
            SUM(((prtSize_agg >= 200) AND ((buy_sell_class='buy' AND okey_cp='Call') OR (buy_sell_class='sell' AND okey_cp='Put')))::INT) AS whale_bull_count,
            SUM(((prtSize_agg >= 200) AND ((buy_sell_class='sell' AND okey_cp='Call') OR (buy_sell_class='buy' AND okey_cp='Put')))::INT) AS whale_bear_count
        FROM {from_source}
        GROUP BY 1, 2
    """


def write_parquet_from_query(con: duckdb.DuckDBPyConnection, query: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"""
        COPY (
            {query}
        ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION 'zstd', ROW_GROUP_SIZE 122880)
    """)


def process_non_batched(con: duckdb.DuckDBPyConnection, output_file: Path, logger) -> None:
    logger.info("Creating DuckDB view over processed parquet files...")

    # Apply optional TARGET_YEARS by restricting the read_parquet file list
    folders = discover_daily_folders(PROCESSED_PATH, TARGET_YEARS)
    if TARGET_YEARS is not None:
        if not folders:
            logger.error(f"No daily folders found for years: {TARGET_YEARS}")
            return
        file_patterns = ", ".join([f"'{str(f / '**/*.parquet')}'" for f in folders])
        read_source = f"read_parquet([{file_patterns}])"
    else:
        read_source = f"read_parquet('{PROCESSED_PATH}/**/*.parquet')"

    con.execute(f"""
        CREATE OR REPLACE VIEW trades AS
        SELECT okey_tk, okey_cp, timestamp_ny, prtSize_agg, buy_sell_class, prtType, ticker_class
        FROM {read_source}
        WHERE ticker_class = 'equity' AND prtType >= 73 AND prtType < 102
    """)

    core = build_core_select("trades")
    final_query = f"""
        SELECT
            okey_tk AS ticker,
            year_month,
            whale_count,
            total_count,
            CAST(whale_count AS DOUBLE) / NULLIF(total_count, 0) AS attention,
            whale_bull_count,
            whale_bear_count,
            COALESCE((CAST(whale_bull_count AS DOUBLE) - CAST(whale_bear_count AS DOUBLE)) /
                     NULLIF(CAST(whale_bull_count AS DOUBLE) + CAST(whale_bear_count AS DOUBLE), 0), 0) AS sentiment
        FROM (
            {core}
        )
    """

    write_parquet_from_query(con, final_query, output_file)

    # Drop the view to free catalog memory
    con.execute("DROP VIEW IF EXISTS trades")


if __name__ == '__main__':
    logger = initialize_main()
    logger.info("Starting simple_attention_sentiment_duckdb.py")

    output_file = OUTPUT_DIR / "attention_sentiment_direction.parquet"

    logger.info(f"Reading from {PROCESSED_PATH}")
    logger.info(f"Output file: {output_file}")
    if TARGET_YEARS is not None:
        logger.info(f"Target years filter: {TARGET_YEARS}")

    manager = DuckDBManager()
    con = manager.connect()
    try:
        # Already configured via manager.connect()

        logger.info("Mode: non-batched (single query over all files)")
        process_non_batched(con, output_file, logger)

        # Basic sanity logs
        try:
            cnt = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output_file}')").fetchone()[0]
            logger.info(f"Saved {cnt} ticker-month rows to {output_file}")
            rng = con.execute(f"""
                SELECT MIN(attention), MAX(attention), MIN(sentiment), MAX(sentiment)
                FROM read_parquet('{output_file}')
            """).fetchone()
            logger.info(f"Attention range: {rng[0]:.4f} - {rng[1]:.4f}; Sentiment range: {rng[2]:.4f} - {rng[3]:.4f}")
        except Exception:
            pass

        logger.info("Attention/sentiment computation completed successfully")

    finally:
        con.close()
        logger.info("DuckDB connection closed")