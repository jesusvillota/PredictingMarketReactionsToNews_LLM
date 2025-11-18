# uv run src/tabling_dask/Table4.py
#---------------------------------------------------------------
from pathlib import Path
import sys
import shutil
PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(PROJECT_ROOT))
#---------------------------------------------------------------

import dask.dataframe as dd
import pandas as pd
from src.config import config_settings, get_logger, DaskManager
from src.config.config_settings import COMPLEX_TRADES_PATH, tables
from src.tabling_dask.common import format_count, format_percentage

OUTPUT_DIR = PROJECT_ROOT / tables["dask_path"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def build_table(parquet_filters: list, output_dir):

    logger = get_logger(__name__)
    logger.info("Starting Table4.build_table() with Dask.")
    required_columns = ['n_legs', 'prtSize_agg']
    
    with DaskManager() as dask_manager:
        logger.info("Loading complex trades data with Dask...")
        ddf = dd.read_parquet(
            path=COMPLEX_TRADES_PATH,
            engine=config_settings.parquet["engine"],
            columns=required_columns,
        )
        logger.info(f"Loaded Dask DataFrame with {ddf.npartitions} partitions")
        
        # Categorize each trade by its leg count and size
        def categorize_legs(partition):
            """Categorize each trade by its leg count and size - VECTORIZED"""
            if partition.empty:
                return pd.DataFrame(columns=['category', 'leg_count', 'count'])
            
            # Categorize leg count (1, 2, 3, 4, or >4) - vectorized
            leg_cat = partition['n_legs'].clip(upper=5)
            leg_cat = leg_cat.astype('int8')
            
            # Create base dataframe for 'all' category
            all_cat = pd.DataFrame({
                'category': 'all',
                'leg_count': leg_cat,
                'count': 1
            })
            
            # Define size categories using same logic as create_standard_size_bins()
            # Bin boundaries: 1-10 (inclusive), 11-200 (inclusive), >200
            size_series = partition['prtSize_agg']
            size_categories = [
                ('1_10', (size_series >= 1) & (size_series <= 10)),
                ('11_200', (size_series >= 11) & (size_series <= 200)),
                ('over_200', size_series > 200),
            ]
            
            # Create dataframes for each size category using vectorized operations
            category_dfs = [all_cat]
            
            for cat_name, mask in size_categories:
                if mask.any():
                    cat_df = pd.DataFrame({
                        'category': cat_name,
                        'leg_count': leg_cat[mask].values,
                        'count': 1
                    })
                    category_dfs.append(cat_df)
            
            # Concatenate all category dataframes
            result = pd.concat(category_dfs, ignore_index=True)
            
            # Optimize dtypes for memory efficiency
            result['category'] = result['category'].astype('category')
            result['leg_count'] = result['leg_count'].astype('int8')
            result['count'] = result['count'].astype('int32')
            
            return result
        
        # Apply categorization to all partitions
        # Define metadata with optimized dtypes
        cat_meta = pd.DataFrame(columns=['category', 'leg_count', 'count'])
        cat_meta['category'] = cat_meta['category'].astype('category')
        cat_meta['leg_count'] = cat_meta['leg_count'].astype('int8')
        cat_meta['count'] = cat_meta['count'].astype('int32')
        
        logger.info("Categorizing trades by leg count and size (vectorized)...")
        result_ddf = ddf.map_partitions(categorize_legs, meta=cat_meta)
        
        # Aggregate results
        logger.info("Aggregating final results across all categories...")
        logger.info("This operation may take several minutes with large datasets...")
        result_pdf = result_ddf.groupby(['category', 'leg_count'], observed=True).sum().compute()
        result_pdf = result_pdf.sort_index()  # Sort multi-index to avoid performance warnings
        logger.info("Aggregation complete!")
        
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
        out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "Table4.tex"
        if output_path.exists() and output_path.is_dir():
            shutil.rmtree(output_path)
        with open(output_path, 'w') as f:
            f.write(latex_content)
        
        logger.info(f"LaTeX table successfully written to: {output_path}")
        logger.info("Table4.build_table() completed successfully.")


if __name__ == '__main__':
    build_table([], OUTPUT_DIR)

