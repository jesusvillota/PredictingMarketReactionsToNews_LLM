# uv run src/tabling_dask/Table5_good.py
#---------------------------------------------------------------
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(PROJECT_ROOT))
#---------------------------------------------------------------

import dask.dataframe as dd
import pandas as pd
from src.config import initialize_main, DaskManager
from src.config.config_settings import tables
from THIS_IS import COMPLEX_TRADES_PATH

OUTPUT_DIR = PROJECT_ROOT / tables["dask_path"]
OUTPUT_PATH = OUTPUT_DIR / "Table_complex_strategies.tex"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fmt_cell(count: int, total: int) -> str:
    """Format count with percentage for LaTeX table cell."""
    if total == 0:
        return "0 (0.0\\%)"
    pct = (count / total * 100.0)
    return f"{int(count):,} ({pct:.1f}\\%)"

def main():
    logger = initialize_main()
    logger.info("Starting Table5_good.py script with Dask.")
    
    with DaskManager() as _dm:
        logger.info("Loading complex_trades parquet with Dask...")
        
        # Read data
        ddf = dd.read_parquet(
            path=str(COMPLEX_TRADES_PATH / "**" / "*.parquet"),
            engine="pyarrow",
            columns=[
                "prtSize_agg",
                "n_legs",
                "sign",
                "flag",
                "strategy_name",
            ]
        )
        logger.info(f"Loaded Dask DataFrame with {ddf.npartitions} partitions.")
        
        # Compute to pandas for easier manipulation
        pdf = ddf.compute()
        
        # Drop rows with missing essential data
        pdf = pdf.dropna(subset=["prtSize_agg", "n_legs", "strategy_name"])
        
        # Map sign values for Single strategies: buy->Long, sell->Short, midpoint->Midpoint
        mask_single = (pdf["n_legs"] == 1) & (pdf["strategy_name"] == "Single")
        pdf.loc[mask_single & (pdf["sign"] == "buy"), "sign"] = "Long"
        pdf.loc[mask_single & (pdf["sign"] == "sell"), "sign"] = "Short"
        pdf.loc[mask_single & (pdf["sign"] == "midpoint"), "sign"] = "Midpoint"
        
        # Create size categories
        pdf.loc[(pdf["prtSize_agg"] >= 1) & (pdf["prtSize_agg"] <= 10), "size_category"] = "1_10"
        pdf.loc[(pdf["prtSize_agg"] >= 11) & (pdf["prtSize_agg"] <= 200), "size_category"] = "11_200"
        pdf.loc[pdf["prtSize_agg"] > 200, "size_category"] = "over_200"
        
        # Get total counts for each size category
        total_all = len(pdf)
        total_1_10 = len(pdf[pdf["size_category"] == "1_10"])
        total_11_200 = len(pdf[pdf["size_category"] == "11_200"])
        total_over_200 = len(pdf[pdf["size_category"] == "over_200"])
        
        totals_dict = {
            'all': total_all,
            '1_10': total_1_10,
            '11_200': total_11_200,
            'over_200': total_over_200
        }
        
        logger.info(f"Total complex strategies: {total_all:,}")
        logger.info(f"Size category totals: {totals_dict}")
        
        # Create groupby for "all" column (aggregate across all sizes)
        groups_all = pdf.groupby(["sign", "flag", "strategy_name"]).size().reset_index(name="count")
        
        # Create groupbys for each size category
        pdf_1_10 = pdf[pdf["size_category"] == "1_10"]
        pdf_11_200 = pdf[pdf["size_category"] == "11_200"]
        pdf_over_200 = pdf[pdf["size_category"] == "over_200"]
        
        groups_1_10 = pdf_1_10.groupby(["sign", "flag", "strategy_name"]).size().reset_index(name="count")
        groups_11_200 = pdf_11_200.groupby(["sign", "flag", "strategy_name"]).size().reset_index(name="count")
        groups_over_200 = pdf_over_200.groupby(["sign", "flag", "strategy_name"]).size().reset_index(name="count")
        
        # Create pivot tables for easier lookup
        all_pivot = groups_all.pivot_table(
            index=["sign", "flag", "strategy_name"],
            values="count",
            aggfunc='sum',
            fill_value=0
        ).reset_index()
        
        pivot_1_10 = groups_1_10.pivot_table(
            index=["sign", "flag", "strategy_name"],
            values="count",
            aggfunc='sum',
            fill_value=0
        ).reset_index()
        
        pivot_11_200 = groups_11_200.pivot_table(
            index=["sign", "flag", "strategy_name"],
            values="count",
            aggfunc='sum',
            fill_value=0
        ).reset_index()
        
        pivot_over_200 = groups_over_200.pivot_table(
            index=["sign", "flag", "strategy_name"],
            values="count",
            aggfunc='sum',
            fill_value=0
        ).reset_index()
        
        # Create lookup dictionaries
        def create_lookup(pivot_df):
            lookup = {}
            for _, row in pivot_df.iterrows():
                key = (row["sign"], row["flag"], row["strategy_name"])
                lookup[key] = row["count"]
            return lookup
        
        lookup_all = create_lookup(all_pivot)
        lookup_1_10 = create_lookup(pivot_1_10)
        lookup_11_200 = create_lookup(pivot_11_200)
        lookup_over_200 = create_lookup(pivot_over_200)
        
        def get_counts(sign, flag, strategy):
            """Helper to get counts for a specific strategy row."""
            key = (sign, flag, strategy)
            return {
                'all': lookup_all.get(key, 0),
                '1_10': lookup_1_10.get(key, 0),
                '11_200': lookup_11_200.get(key, 0),
                'over_200': lookup_over_200.get(key, 0)
            }
        
        logger.info("Writing LaTeX table...")
        
        # Build LaTeX table content
        latex_rows = []
        
        # Add header
        latex_rows.append(r"\begin{table}[htbp]")
        latex_rows.append(r"    \centering")
        latex_rows.append(r"    \caption{Distribution of Complex Option Strategies by Trade Size Category}")
        latex_rows.append(r"    \subcaption*{")
        latex_rows.append(r"    {\scriptsize")
        latex_rows.append(
            f"    Distribution of complex option strategies across different trade size categories. Each cell shows the count of trades for that strategy type with the percentage within that size category shown in parentheses. Rows show different strategy types as classified by the algorithm. Columns represent trade size categories based on the number of contracts: All (all trades), 1--10 (size between 1 and 10), 11--200 (size between 11 and 200), and >200 (size greater than 200). Total observations: $N = {int(total_all):,}$ complex strategies."
        )
        latex_rows.append(r"    \par}")
        latex_rows.append(r"    \vspace{1em}")
        latex_rows.append(r"    }")
        latex_rows.append(r"    \label{tab:complex_strategies_by_size}")
        latex_rows.append(r"    \scriptsize")
        latex_rows.append(r"    \begin{tabular}{llllcccc}")
        latex_rows.append(r"    \toprule")
        latex_rows.append(r"        \textbf{Legs} & \textbf{Strategy} & \textbf{Sign} & \textbf{Flag} & \textbf{All} & \textbf{1--10} & \textbf{11--200} & \textbf{>200} \\")
        latex_rows.append(r"    \midrule")
        
        # 1-Leg Strategies (6 rows: Long/Short/Midpoint × Call/Put)
        latex_rows.append(r"    \multirow{6}{*}{\textbf{1 Leg}}")
        for sign in ['Long', 'Short', 'Midpoint']:
            for flag in ['Call', 'Put']:
                counts_dict = get_counts(sign, flag, 'Single')
                counts = [
                    fmt_cell(counts_dict['all'], totals_dict.get('all', 1)),
                    fmt_cell(counts_dict['1_10'], totals_dict.get('1_10', 1)),
                    fmt_cell(counts_dict['11_200'], totals_dict.get('11_200', 1)),
                    fmt_cell(counts_dict['over_200'], totals_dict.get('over_200', 1))
                ]
                latex_rows.append(f"    & Single & {sign} & {flag} & {' & '.join(counts)} \\\\")
        
        latex_rows.append(r"    \midrule")
        
        # 2-Leg Strategies (24 rows: Spread/Calendar/Diagonal × Long/Short/Midpoint × Call/Put, Straddle/Strangle × Long/Short/Midpoint)
        latex_rows.append(r"    \multirow{24}{*}{\textbf{2 Legs}}")
        # Spread
        for sign in ['Long', 'Short', 'Midpoint']:
            for flag in ['Call', 'Put']:
                counts_dict = get_counts(sign, flag, 'Spread')
                counts = [
                    fmt_cell(counts_dict['all'], totals_dict.get('all', 1)),
                    fmt_cell(counts_dict['1_10'], totals_dict.get('1_10', 1)),
                    fmt_cell(counts_dict['11_200'], totals_dict.get('11_200', 1)),
                    fmt_cell(counts_dict['over_200'], totals_dict.get('over_200', 1))
                ]
                latex_rows.append(f"    & Spread & {sign} & {flag} & {' & '.join(counts)} \\\\")
        # Calendar
        for sign in ['Long', 'Short', 'Midpoint']:
            for flag in ['Call', 'Put']:
                counts_dict = get_counts(sign, flag, 'Calendar')
                counts = [
                    fmt_cell(counts_dict['all'], totals_dict.get('all', 1)),
                    fmt_cell(counts_dict['1_10'], totals_dict.get('1_10', 1)),
                    fmt_cell(counts_dict['11_200'], totals_dict.get('11_200', 1)),
                    fmt_cell(counts_dict['over_200'], totals_dict.get('over_200', 1))
                ]
                latex_rows.append(f"    & Calendar & {sign} & {flag} & {' & '.join(counts)} \\\\")
        # Diagonal
        for sign in ['Long', 'Short', 'Midpoint']:
            for flag in ['Call', 'Put']:
                counts_dict = get_counts(sign, flag, 'Diagonal')
                counts = [
                    fmt_cell(counts_dict['all'], totals_dict.get('all', 1)),
                    fmt_cell(counts_dict['1_10'], totals_dict.get('1_10', 1)),
                    fmt_cell(counts_dict['11_200'], totals_dict.get('11_200', 1)),
                    fmt_cell(counts_dict['over_200'], totals_dict.get('over_200', 1))
                ]
                latex_rows.append(f"    & Diagonal & {sign} & {flag} & {' & '.join(counts)} \\\\")
        # Straddle (no flag - these use Mixed flag)
        for sign in ['Long', 'Short', 'Midpoint']:
            counts_dict = get_counts(sign, 'Mixed', 'Straddle')
            counts = [
                fmt_cell(counts_dict['all'], totals_dict.get('all', 1)),
                fmt_cell(counts_dict['1_10'], totals_dict.get('1_10', 1)),
                fmt_cell(counts_dict['11_200'], totals_dict.get('11_200', 1)),
                fmt_cell(counts_dict['over_200'], totals_dict.get('over_200', 1))
            ]
            latex_rows.append(f"    & Straddle & {sign} &  & {' & '.join(counts)} \\\\")
        # Strangle (no flag - these use Mixed flag)
        for sign in ['Long', 'Short', 'Midpoint']:
            counts_dict = get_counts(sign, 'Mixed', 'Strangle')
            counts = [
                fmt_cell(counts_dict['all'], totals_dict.get('all', 1)),
                fmt_cell(counts_dict['1_10'], totals_dict.get('1_10', 1)),
                fmt_cell(counts_dict['11_200'], totals_dict.get('11_200', 1)),
                fmt_cell(counts_dict['over_200'], totals_dict.get('over_200', 1))
            ]
            latex_rows.append(f"    & Strangle & {sign} &  & {' & '.join(counts)} \\\\")
        
        latex_rows.append(r"    \midrule")
        
        # 3-Leg Strategies - Butterfly (6 rows: Long/Short/Midpoint × Call/Put)
        latex_rows.append(r"    \multirow{6}{*}{\textbf{3 Legs}}")
        for sign in ['Long', 'Short', 'Midpoint']:
            for flag in ['Call', 'Put']:
                counts_dict = get_counts(sign, flag, 'Butterfly')
                counts = [
                    fmt_cell(counts_dict['all'], totals_dict.get('all', 1)),
                    fmt_cell(counts_dict['1_10'], totals_dict.get('1_10', 1)),
                    fmt_cell(counts_dict['11_200'], totals_dict.get('11_200', 1)),
                    fmt_cell(counts_dict['over_200'], totals_dict.get('over_200', 1))
                ]
                latex_rows.append(f"    & Butterfly & {sign} & {flag} & {' & '.join(counts)} \\\\")
        
        latex_rows.append(r"    \midrule")
        
        # 4-Leg Strategies (9 rows: Condor × Long/Short/Midpoint × Call/Put, IronCondor × Long/Short/Midpoint)
        latex_rows.append(r"    \multirow{9}{*}{\textbf{4 Legs}}")
        # Condor
        for sign in ['Long', 'Short', 'Midpoint']:
            for flag in ['Call', 'Put']:
                counts_dict = get_counts(sign, flag, 'Condor')
                counts = [
                    fmt_cell(counts_dict['all'], totals_dict.get('all', 1)),
                    fmt_cell(counts_dict['1_10'], totals_dict.get('1_10', 1)),
                    fmt_cell(counts_dict['11_200'], totals_dict.get('11_200', 1)),
                    fmt_cell(counts_dict['over_200'], totals_dict.get('over_200', 1))
                ]
                latex_rows.append(f"    & Condor & {sign} & {flag} & {' & '.join(counts)} \\\\")
        # IronCondor (no flag - these use Mixed flag)
        for sign in ['Long', 'Short', 'Midpoint']:
            counts_dict = get_counts(sign, 'Mixed', 'IronCondor')
            counts = [
                fmt_cell(counts_dict['all'], totals_dict.get('all', 1)),
                fmt_cell(counts_dict['1_10'], totals_dict.get('1_10', 1)),
                fmt_cell(counts_dict['11_200'], totals_dict.get('11_200', 1)),
                fmt_cell(counts_dict['over_200'], totals_dict.get('over_200', 1))
            ]
            latex_rows.append(f"    & IronCondor & {sign} &  & {' & '.join(counts)} \\\\")
        
        latex_rows.append(r"    \midrule")
        
        # Other (no sign/flag)
        counts_dict = get_counts('', '', 'Other')
        counts = [
            fmt_cell(counts_dict['all'], totals_dict.get('all', 1)),
            fmt_cell(counts_dict['1_10'], totals_dict.get('1_10', 1)),
            fmt_cell(counts_dict['11_200'], totals_dict.get('11_200', 1)),
            fmt_cell(counts_dict['over_200'], totals_dict.get('over_200', 1))
        ]
        latex_rows.append(r"    \multirow{1}{*}{\textbf{ }}")
        latex_rows.append(f"    & Other &  &  & {' & '.join(counts)} \\\\")
        
        latex_rows.append(r"    \midrule")
        
        # Total row
        total_counts = [
            fmt_cell(totals_dict.get('all', 0), totals_dict.get('all', 1)),
            fmt_cell(totals_dict.get('1_10', 0), totals_dict.get('1_10', 1)),
            fmt_cell(totals_dict.get('11_200', 0), totals_dict.get('11_200', 1)),
            fmt_cell(totals_dict.get('over_200', 0), totals_dict.get('over_200', 1))
        ]
        latex_rows.append(f"    \\textbf{{Total}} &  &  &  & {' & '.join(total_counts)} \\\\")
        latex_rows.append(r"    \bottomrule")
        latex_rows.append(r"    \end{tabular}")
        latex_rows.append(r"\end{table}")
        
        # Write to file
        latex_content = '\n'.join(latex_rows) + '\n'
        
        with open(OUTPUT_PATH, 'w') as f:
            f.write(latex_content)
        
        logger.info(f"LaTeX table successfully written to: {OUTPUT_PATH}")
    
    logger.info("Table5_good.py script completed successfully.")

if __name__ == '__main__':
    main()

