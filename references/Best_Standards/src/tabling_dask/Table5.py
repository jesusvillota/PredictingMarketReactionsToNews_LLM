# uv run src/tabling_dask/Table5.py
#---------------------------------------------------------------
from pathlib import Path
import sys
import shutil
PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(PROJECT_ROOT))
#---------------------------------------------------------------

import dask.dataframe as dd
from src.config import get_logger
from src.config.config_settings import tables, COMPLEX_TRADES_PATH
from src.tabling_dask.common import fmt_cell

OUTPUT_DIR = PROJECT_ROOT / tables["dask_path"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def build_table(parquet_filters: list, output_dir):
    logger = get_logger(__name__)
    logger.info("Starting Table5.build_table() with Dask.")

    ddf = dd.read_parquet(
        path=COMPLEX_TRADES_PATH,
        engine="pyarrow",
        columns=[
            "prtSize_agg",
            "n_legs",
            "sign",
            "flag",
            "strategy_name",
        ]
        )


    pdf = ddf.compute()

    mask_1_10 = (pdf["prtSize_agg"] >= 1) & (pdf["prtSize_agg"] <= 10)
    mask_11_200 = (pdf["prtSize_agg"] >= 11) & (pdf["prtSize_agg"] <= 200)
    mask_over_200 = (pdf["prtSize_agg"] > 200)

    pdf_all = pdf.copy()
    pdf_1_10 = pdf[mask_1_10]
    pdf_11_200 = pdf[mask_11_200]
    pdf_over_200 = pdf[mask_over_200]

    groups_all = pdf_all.groupby(["sign", "flag", "strategy_name"]).size().reset_index(name="count")
    groups_1_10 = pdf_1_10.groupby(["sign", "flag", "strategy_name"]).size().reset_index(name="count")
    groups_11_200 = pdf_11_200.groupby(["sign", "flag", "strategy_name"]).size().reset_index(name="count")
    groups_over_200 = pdf_over_200.groupby(["sign", "flag", "strategy_name"]).size().reset_index(name="count")

    len_all = len(pdf_all)
    len_1_10 = len(pdf_1_10)
    len_11_200 = len(pdf_11_200)
    len_over_200 = len(pdf_over_200)


    def get_count(df, sign, flag, strat):
        mask = (df['sign'] == sign) & (df['flag'] == flag) & (df['strategy_name'] == strat)
        if mask.any():
            return df.loc[mask, 'count'].values[0]
        else:
            return 0

    totals = [len_all, len_1_10, len_11_200, len_over_200]
    groups = [groups_all, groups_1_10, groups_11_200, groups_over_200]

    # Helper to create rows for call/put strategies
    def make_call_put_rows(strat):
        res = []
        for sign in ['Long', 'Short', 'Midpoint']:
            for f in ['Call', 'Put']:
                res.append({
                    'strategy': strat,
                    'sign': sign,
                    'flag': f,
                    'q_sign': sign,
                    'q_flag': f,
                    'q_strat': strat
                })
        return res

    # Helper to create rows for mixed strategies (straddle, strangle, iron condor)
    def make_mixed_rows(strat):
        res = []
        for sign in ['Long', 'Short', 'Midpoint']:
            res.append({
                'strategy': strat,
                'sign': sign,
                'flag': '',
                'q_sign': sign,
                'q_flag': 'Mixed',
                'q_strat': strat
            })
        return res

    # Define all rows per section (with mappings for singles)
    single_rows = [
        {'strategy': 'Single', 'sign': 'Long', 'flag': 'Call', 'q_sign': 'buy', 'q_flag': 'Call', 'q_strat': 'Single'},
        {'strategy': 'Single', 'sign': 'Short', 'flag': 'Call', 'q_sign': 'sell', 'q_flag': 'Call', 'q_strat': 'Single'},
        {'strategy': 'Single', 'sign': 'Midpoint', 'flag': 'Call', 'q_sign': 'midpoint', 'q_flag': 'Call', 'q_strat': 'Single'},
        {'strategy': 'Single', 'sign': 'Long', 'flag': 'Put', 'q_sign': 'buy', 'q_flag': 'Put', 'q_strat': 'Single'},
        {'strategy': 'Single', 'sign': 'Short', 'flag': 'Put', 'q_sign': 'sell', 'q_flag': 'Put', 'q_strat': 'Single'},
        {'strategy': 'Single', 'sign': 'Midpoint', 'flag': 'Put', 'q_sign': 'midpoint', 'q_flag': 'Put', 'q_strat': 'Single'},
    ]

    spread_rows = make_call_put_rows('Spread')
    calendar_rows = make_call_put_rows('Calendar')
    diagonal_rows = make_call_put_rows('Diagonal')
    straddle_rows = make_mixed_rows('Straddle')
    strangle_rows = make_mixed_rows('Strangle')
    butterfly_rows = make_call_put_rows('Butterfly')
    condor_rows = make_call_put_rows('Condor')
    iron_rows = make_mixed_rows('IronCondor')
    other_rows = [
        {'strategy': 'Other', 'sign': '', 'flag': '', 'q_sign': 'Undetermined', 'q_flag': 'None', 'q_strat': 'Other'}
    ]

    # Sections
    sections = [
        {'legs': '\\textbf{1 Leg}', 'multirow': 6, 'rows': single_rows},
        {'legs': '\\textbf{2 Legs}', 'multirow': 24, 'rows': spread_rows + calendar_rows + diagonal_rows + straddle_rows + strangle_rows},
        {'legs': '\\textbf{3 Legs}', 'multirow': 6, 'rows': butterfly_rows},
        {'legs': '\\textbf{4 Legs}', 'multirow': 9, 'rows': condor_rows + iron_rows},
        {'legs': '\\textbf{ }', 'multirow': 1, 'rows': other_rows},
    ]

    # Build the table content lines
    table_lines = [
        r'\begin{table}[htbp]',
        # r'% \begin{sidewaystable}[htbp]',  # Uncomment if needed
        r'\centering',
        r'\caption{Distribution of Complex Option Strategies by Trade Size Category}',
        r'\subcaption*{',
        r'    {\scriptsize',
        # r'% Distribution of complex option strategies across different trade size categories. Each cell shows the count of trades for that strategy type with the percentage within that size category shown in parentheses. Rows show different strategy types as classified by the algorithm. Columns represent trade size categories based on the number of contracts: All (all trades), 1--10 (size between 1 and 10), 11--200 (size between 11 and 200), and >200 (size greater than 200).',
        r'\par}',
        r'\vspace{1em}',
        r'    }',
        r'\label{tab:complex_strategies_by_size}',
        r'\scriptsize',
        r'\begin{tabular}{llllcccc}',
        r'\toprule',
        r'\textbf{Legs}',
        r'& \textbf{Strategy}',
        r'& \textbf{Sign}',
        r'& \textbf{Flag}',
        r'& \textbf{All}',
        r'& \textbf{1--10}',
        r'& \textbf{11--200}',
        r'& \textbf{>200} \\',
        r'\midrule',
    ]

    for section in sections:
        table_lines.append(f"\\multirow{{{section['multirow']}}}{{*}}{{{section['legs']}}}")
        for row in section['rows']:
            cells = []
            for i in range(4):
                count = get_count(groups[i], row['q_sign'], row['q_flag'], row['q_strat'])
                cells.append(fmt_cell(count, totals[i], percentage_only=True))
            table_lines.append(f"& {row['strategy']} & {row['sign']} & {row['flag']} & {' & '.join(cells)} \\\\")
        table_lines.append(r'\midrule')

    # Total row
    table_lines.append(f"\\textbf{{Total}} & & & & {len_all:,} & {len_1_10:,} & {len_11_200:,} & {len_over_200:,} \\\\")

    # Footer
    table_lines += [
        r'\bottomrule',
        r'\end{tabular}',
        # r'% \end{sidewaystable}',  # Uncomment if needed
        r'\end{table}',
    ]

    # Print the full filled table
    latex_content = '\n'.join(table_lines)

    # Resolve output path
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "Table5.tex"
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)

    # Write to file 
    with open(output_path, 'w') as f:
        f.write(latex_content)

    logger.info(f"LaTeX table successfully written to: {output_path}")
    logger.info("Table5.build_table() completed successfully.")


if __name__ == '__main__':
    build_table([], OUTPUT_DIR)