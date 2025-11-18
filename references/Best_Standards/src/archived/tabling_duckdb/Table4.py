# uv run src/tabling_duckdb/Table4.py
#---------------------------------------------------------------
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(PROJECT_ROOT))
#---------------------------------------------------------------

import duckdb
import pandas as pd
from src.config import initialize_main
from src.config.config_settings import PROCESSED_PATH, tables

OUTPUT_DIR = PROJECT_ROOT / tables["duckdb_path"]
OUTPUT_PATH = OUTPUT_DIR / "Table_4.tex"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':

    logger = initialize_main()
    logger.info("Starting Table4.py script with DuckDB.")
    
    try:
        # Connect to DuckDB
        con = duckdb.connect()
        logger.info("DuckDB connection established.")
        
        # Get the parquet path pattern
        parquet_path = str(PROCESSED_PATH / "**/*.parquet")
        logger.info(f"Reading parquet files from: {parquet_path}")
        
        # Build SQL query using CTEs for leg counting
        logger.info("Building DuckDB query for leg counting...")
        
        query = f"""
        WITH grouped_strategies AS (
            -- Group complex trades by strategy identifier and count legs
            SELECT 
                okey_tk,
                prtExch,
                prtType,
                timestamp_ny_round3,
                COUNT(*) as n_legs,
                FIRST(prtSize_agg) as size
            FROM read_parquet('{parquet_path}', hive_partitioning=0)
            WHERE prtType >= 102
                AND ticker_class = 'equity'
            GROUP BY okey_tk, prtExch, prtType, timestamp_ny_round3
        ),
        categorized AS (
            -- Categorize by size and leg count
            SELECT 
                CASE 
                    WHEN size BETWEEN 1 AND 10 THEN '1_10'
                    WHEN size BETWEEN 11 AND 200 THEN '11_200'
                    WHEN size > 200 THEN 'over_200'
                END AS size_category,
                CASE 
                    WHEN n_legs > 4 THEN 5
                    ELSE n_legs
                END AS leg_count
            FROM grouped_strategies
        ),
        all_category AS (
            -- Count for 'all' category
            SELECT 
                'all' as category,
                leg_count,
                COUNT(*) as count
            FROM categorized
            GROUP BY leg_count
        ),
        size_categories AS (
            -- Count for each size category
            SELECT 
                size_category as category,
                leg_count,
                COUNT(*) as count
            FROM categorized
            WHERE size_category IS NOT NULL
            GROUP BY size_category, leg_count
        )
        -- Combine all results
        SELECT * FROM (
            SELECT * FROM all_category
            UNION ALL
            SELECT * FROM size_categories
        )
        ORDER BY 
            CASE category
                WHEN 'all' THEN 1
                WHEN '1_10' THEN 2
                WHEN '11_200' THEN 3
                WHEN 'over_200' THEN 4
            END,
            leg_count
        """
        
        logger.info("Executing DuckDB query...")
        result_pdf = con.execute(query).fetchdf()
        
        # Pivot to get the desired structure
        result_pdf = result_pdf.set_index(['category', 'leg_count'])
        
        logger.info("Aggregation complete!")
        
    except Exception as e:
        logger.exception(f"Error during DuckDB query execution: {e}")
        raise
    finally:
        con.close()
        logger.info("DuckDB connection closed.")
    
    logger.info("Building LaTeX table...")
    
    # Prepare data structure for table
    categories: list[str] = ["all", "1_10", "11_200", "over_200"]
    leg_counts = [1, 2, 3, 4, 5]  # 5 represents >4
    
    # Create a pivot table structure
    table_data = {}
    category_totals = {}
    
    for cat in categories:
        table_data[cat] = {}
        category_total = 0
        
        for leg in leg_counts:
            if (cat, leg) in result_pdf.index:
                count_value = result_pdf.loc[(cat, leg), 'count']
                # Ensure we have a scalar value (in case of any edge cases)
                count = count_value.item() if hasattr(count_value, 'item') else int(count_value)
                table_data[cat][leg] = count
                category_total += count
            else:
                table_data[cat][leg] = 0
        
        category_totals[cat] = category_total
    
    # Format numbers with thousands separators and percentages
    def format_number(value):
        if value == 0 or pd.isna(value):
            return "0"
        return f"{int(value):,}"
    
    def format_count(count):
        """Format count without percentage"""
        if count == 0 or pd.isna(count):
            return "0"
        return f"{int(count):,}"
    
    def format_percentage(count, total):
        """Format percentage only"""
        if count == 0 or total == 0 or pd.isna(count):
            return "(0.0\\%)"
        percentage = (count / total) * 100
        return f"({percentage:.1f}\\%)"
    
    # Get total sample size
    total_N = category_totals['all']
    
    # Build LaTeX table
    latex_content = r"""\begin{table}[htbp]
    \centering
    \caption{Number of Legs in Complex Strategies by Trade Size Category}
    \subcaption*{
    {\scriptsize
    Distribution of complex option strategies by the number of legs (individual option contracts) across different trade size categories. Each cell shows the count of strategies with the percentage within that category shown below in parentheses. Rows show the count of strategies with 1, 2, 3, 4, or more than 4 legs. Columns represent trade size categories based on the number of contracts: All (all trades), 1--10 (trades with $1 \leq$ size $\leq$ 10), 11--200 (trades with $10 <$ size $\leq$ 200), and >200 (trades with size $>$ 200). Data filtered to include only equity options (ticker\_class == ``Equity'') with prtType $\geq$ 73. Total observations: $N = """ + f'{int(total_N):,}' + r"""$ complex strategies.
    \par}
    \vspace{1em}
    }
    \label{tab:legs_by_category}
    \scriptsize
    \begin{tabular}{lcccc}
    \toprule
        \textbf{Number of Legs} 
        & \textbf{All} 
        & \textbf{1--10} 
        & \textbf{11--200} 
        & \textbf{>200} \\
    \midrule
"""
    
    # Add rows - each leg count gets two rows (count row and percentage row below)
    leg_labels = {1: "1 Leg", 2: "2 Legs", 3: "3 Legs", 4: "4 Legs", 5: ">4 Legs"}
    for leg in leg_counts:
        # Count row
        count_values = [format_count(table_data[cat][leg]) for cat in categories]
        latex_content += f"    {leg_labels[leg]} & {' & '.join(count_values)} \\\\\n"
        
        # Percentage row (below the count row, with smaller font)
        pct_values = [format_percentage(table_data[cat][leg], category_totals[cat]) for cat in categories]
        # Wrap each percentage value in scriptsize
        pct_values_formatted = [f"{{\\scriptsize {val}}}" for val in pct_values]
        latex_content += f"     & {' & '.join(pct_values_formatted)} \\\\\n"
        
        # Add some spacing between leg categories (except after the last one)
        if leg != leg_counts[-1]:
            latex_content += "    \\addlinespace[0.5em]\n"
    
    # Add total row
    latex_content += "    \\midrule\n"
    total_values = [format_count(category_totals[cat]) for cat in categories]
    latex_content += f"    \\textbf{{Total}} & {' & '.join(total_values)} \\\\\n"
    
    latex_content += r"""    \bottomrule
    \end{tabular}
\end{table}
"""
    
    # Write to file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        f.write(latex_content)
    
    logger.info(f"LaTeX table successfully written to: {OUTPUT_PATH}")
    logger.info("Script completed successfully.")

