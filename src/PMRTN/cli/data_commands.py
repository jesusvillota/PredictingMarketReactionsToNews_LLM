"""CLI commands for data loading and processing."""

import logging
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from tqdm import tqdm

from PMRTN.config import get_settings, get_path_manager
from PMRTN.data import (
    load_raw_articles,
    process_articles,
    save_processed_data,
    load_processed_articles,
)
from PMRTN.embeddings import add_embeddings_to_dataframe
from PMRTN.visualization import (
    configure_matplotlib_style,
    plot_cluster_distribution,
)

logger = logging.getLogger(__name__)


@click.command('load-articles')
@click.option(
    '--config',
    type=click.Path(exists=True, path_type=Path),
    help='Path to configuration file',
)
@click.option(
    '--output',
    type=click.Path(path_type=Path),
    help='Output path for processed articles CSV',
)
@click.option(
    '--filter-agenda/--no-filter-agenda',
    default=True,
    help='Filter out agenda/calendar articles',
)
@click.pass_context
def load_articles(
    ctx: click.Context,
    config: Optional[Path],
    output: Optional[Path],
    filter_agenda: bool,
) -> None:
    """Load and process raw articles from parquet file.
    
    This command loads raw articles, cleans the text, extracts stock tickers,
    and saves the processed data to CSV format.
    
    Equivalent to script: 0_data_articles.py
    """
    logger.info("Starting article loading and processing")
    
    # Get configuration
    config_path = config or ctx.obj.get('config_path')
    settings = get_settings(config_path)
    path_manager = get_path_manager()
    
    # Determine input/output paths
    input_path = path_manager.raw_data / settings.get('files.raw_articles', 'articles.parquet')
    output_path = output or (path_manager.processed_data / 'articles_with_tickers.csv')
    
    logger.info(f"Loading raw articles from: {input_path}")
    
    # Load raw articles
    df = load_raw_articles(
        input_path,
        min_word_count=settings.get('preprocessing.min_word_count', 20),
        filter_agenda=filter_agenda,
    )
    
    logger.info(f"Loaded {len(df)} articles")
    
    # Process articles (merge components, clean text, extract tickers)
    logger.info("Processing articles (merging, cleaning, extracting tickers)...")
    df_processed = process_articles(df)
    
    # Filter articles with valid tickers
    df_with_tickers = df_processed[df_processed['tickers'].apply(len) > 0].copy()
    logger.info(f"Articles with valid tickers: {len(df_with_tickers)}")
    
    # Save processed data
    logger.info(f"Saving processed articles to: {output_path}")
    save_processed_data(df_with_tickers, output_path)
    
    click.echo(f"✓ Successfully processed {len(df_with_tickers)} articles")
    click.echo(f"  Output saved to: {output_path}")


@click.command('describe-data')
@click.option(
    '--config',
    type=click.Path(exists=True, path_type=Path),
    help='Path to configuration file',
)
@click.option(
    '--input',
    type=click.Path(exists=True, path_type=Path),
    help='Input path for processed articles CSV',
)
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    help='Output directory for descriptive statistics',
)
@click.pass_context
def describe_data(
    ctx: click.Context,
    config: Optional[Path],
    input: Optional[Path],
    output_dir: Optional[Path],
) -> None:
    """Generate descriptive statistics and visualizations.
    
    This command loads processed articles and generates various descriptive
    statistics including word counts, article distributions, and summary tables.
    
    Equivalent to script: 1_data_description.py
    """
    logger.info("Starting descriptive analysis")
    
    # Get configuration
    config_path = config or ctx.obj.get('config_path')
    settings = get_settings(config_path)
    path_manager = get_path_manager()
    
    # Determine paths
    input_path = input or (path_manager.processed_data / 'articles_with_tickers.csv')
    output_path = output_dir or path_manager.output_descriptives
    
    logger.info(f"Loading processed articles from: {input_path}")
    df = load_processed_articles(input_path)
    
    logger.info(f"Loaded {len(df)} articles")
    
    # Configure matplotlib for publication-quality plots
    configure_matplotlib_style()
    
    # Generate basic statistics
    click.echo("\n=== Article Statistics ===")
    click.echo(f"Total articles: {len(df)}")
    click.echo(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    click.echo(f"Unique tickers: {df['tickers'].explode().nunique()}")
    
    # Calculate word counts
    df['word_count'] = df['articles'].str.split().str.len()
    click.echo(f"\nWord count statistics:")
    click.echo(f"  Mean: {df['word_count'].mean():.1f}")
    click.echo(f"  Median: {df['word_count'].median():.1f}")
    click.echo(f"  Min: {df['word_count'].min()}")
    click.echo(f"  Max: {df['word_count'].max()}")
    
    # Save summary statistics
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / 'summary_statistics.csv'
    df.describe().to_csv(summary_path)
    
    click.echo(f"\n✓ Descriptive analysis complete")
    click.echo(f"  Output saved to: {output_path}")


@click.command('download-returns')
@click.option(
    '--config',
    type=click.Path(exists=True, path_type=Path),
    help='Path to configuration file',
)
@click.option(
    '--input',
    type=click.Path(exists=True, path_type=Path),
    help='Input path for processed articles CSV',
)
@click.option(
    '--rf-data',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to risk-free rate CSV file (ESTR.csv)',
)
@click.option(
    '--market-ticker',
    type=str,
    default='^IBEX',
    help='Market index ticker symbol (default: ^IBEX for IBEX 35)',
)
@click.option(
    '--model',
    type=click.Choice(['KMeans', 'LLAMA'], case_sensitive=False),
    default='KMeans',
    help='Model type to download data for (determines which tickers to use)',
)
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    help='Output directory for returns data (default: processed_data/)',
)
@click.option(
    '--n-jobs',
    type=int,
    default=-1,
    help='Number of parallel jobs (-1 uses all CPU cores)',
)
@click.pass_context
def download_returns(
    ctx: click.Context,
    config: Optional[Path],
    input: Optional[Path],
    rf_data: Path,
    market_ticker: str,
    model: str,
    output_dir: Optional[Path],
    n_jobs: int,
) -> None:
    """Download stock returns data from Yahoo Finance.
    
    This command downloads historical stock price data for all tickers mentioned
    in the processed articles. It:
    
    \b
    1. Loads risk-free rate (ESTR) data
    2. Downloads market index (IBEX 35) data
    3. Downloads individual stock data in parallel
    4. Calculates returns and excess returns
    5. Saves returns data and ticker lists (successful/failed)
    
    Equivalent to script: 2_data_tickers.py
    
    Example:
        pmrtn download-returns --rf-data data/raw/ESTR.csv --model KMeans
    """
    from PMRTN.utils.financial import (
        load_risk_free_rate,
        download_market_index,
        download_stock_returns,
    )
    
    logger.info("Starting stock returns download")
    
    # Get configuration
    config_path = config or ctx.obj.get('config_path')
    settings = get_settings(config_path)
    path_manager = get_path_manager()
    
    # Determine paths
    input_path = input or (path_manager.processed_data / 'articles_with_tickers.csv')
    output_path = output_dir or path_manager.processed_data
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load risk-free rate data
    click.echo(f"\n=== Step 1: Loading Risk-Free Rate ===")
    click.echo(f"Loading ESTR data from: {rf_data}")
    
    try:
        rf_df = load_risk_free_rate(rf_data)
        click.echo(f"✓ Loaded {len(rf_df)} days of risk-free rate data")
        click.echo(f"  Date range: {rf_df.index[0]} to {rf_df.index[-1]}")
    except Exception as e:
        click.echo(f"✗ Error loading risk-free rate data: {e}", err=True)
        raise click.Abort()
    
    # Step 2: Download market index
    click.echo(f"\n=== Step 2: Downloading Market Index ===")
    click.echo(f"Downloading {market_ticker} data...")
    
    try:
        market_df = download_market_index(
            market_ticker,
            rf_df.index[0],
            rf_df.index[-1]
        )
        click.echo(f"✓ Downloaded {len(market_df)} days of market data")
    except Exception as e:
        click.echo(f"✗ Error downloading market index: {e}", err=True)
        raise click.Abort()
    
    # Combine risk-free rate and market data
    returns_df = rf_df.join(market_df, how='inner')
    returns_df['r_market_excess'] = returns_df['r_market'] - returns_df['rf']
    
    click.echo(f"✓ Combined data: {len(returns_df)} trading days")
    
    # Step 3: Load articles and extract tickers
    click.echo(f"\n=== Step 3: Loading Articles and Tickers ===")
    click.echo(f"Loading articles from: {input_path}")
    
    try:
        df_articles = load_processed_articles(input_path)
        # Get unique tickers
        tickers = sorted(df_articles['tickers'].explode().unique().tolist())
        click.echo(f"✓ Found {len(tickers)} unique tickers")
        
        if len(tickers) == 0:
            click.echo("✗ No tickers found in articles!", err=True)
            raise click.Abort()
            
    except Exception as e:
        click.echo(f"✗ Error loading articles: {e}", err=True)
        raise click.Abort()
    
    # Step 4: Download stock returns
    click.echo(f"\n=== Step 4: Downloading Stock Returns ===")
    click.echo(f"Downloading returns for {len(tickers)} tickers using {n_jobs} workers...")
    click.echo("This may take several minutes...\n")
    
    try:
        returns_df, successful_tickers, failed_tickers = download_stock_returns(
            tickers,
            returns_df.index[0],
            returns_df.index[-1],
            returns_df[['rf']],
            n_jobs=n_jobs
        )
    except Exception as e:
        click.echo(f"✗ Error downloading stock returns: {e}", err=True)
        raise click.Abort()
    
    # Step 5: Save results
    click.echo(f"\n=== Step 5: Saving Results ===")
    
    # Save returns data
    returns_file = output_path / f'R_{model}.csv'
    click.echo(f"Saving returns data to: {returns_file}")
    returns_df.to_csv(returns_file)
    
    # Save successful tickers
    successful_file = output_path / f'successful_tickers_{model}.txt'
    click.echo(f"Saving successful tickers to: {successful_file}")
    with open(successful_file, 'w') as f:
        for ticker in successful_tickers:
            f.write(f'{ticker}\n')
    
    # Save failed tickers
    failed_file = output_path / f'failed_tickers_{model}.txt'
    click.echo(f"Saving failed tickers to: {failed_file}")
    with open(failed_file, 'w') as f:
        for ticker in failed_tickers:
            f.write(f'{ticker}\n')
    
    # Summary
    click.echo(f"\n{'='*60}")
    click.echo(f"✓ Download Complete!")
    click.echo(f"{'='*60}")
    click.echo(f"Model: {model}")
    click.echo(f"Total tickers: {len(tickers)}")
    click.echo(f"  ✓ Successful: {len(successful_tickers)} ({100*len(successful_tickers)/len(tickers):.1f}%)")
    click.echo(f"  ✗ Failed: {len(failed_tickers)} ({100*len(failed_tickers)/len(tickers):.1f}%)")
    click.echo(f"Trading days: {len(returns_df)}")
    click.echo(f"\nOutput files:")
    click.echo(f"  • {returns_file}")
    click.echo(f"  • {successful_file}")
    click.echo(f"  • {failed_file}")
    click.echo(f"{'='*60}\n")


@click.command('fetch-tickers')
@click.option(
    '--config',
    type=click.Path(exists=True, path_type=Path),
    help='Path to configuration file',
)
@click.option(
    '--input',
    type=click.Path(exists=True, path_type=Path),
    help='Input path for processed articles CSV',
)
@click.option(
    '--output',
    type=click.Path(path_type=Path),
    help='Output path for returns data CSV',
)
@click.option(
    '--start-date',
    type=str,
    help='Start date for stock data (YYYY-MM-DD)',
)
@click.option(
    '--end-date',
    type=str,
    help='End date for stock data (YYYY-MM-DD)',
)
@click.pass_context
def fetch_tickers(
    ctx: click.Context,
    config: Optional[Path],
    input: Optional[Path],
    output: Optional[Path],
    start_date: Optional[str],
    end_date: Optional[str],
) -> None:
    """(DEPRECATED) Use 'download-returns' command instead.
    
    This is a placeholder kept for backwards compatibility.
    Please use the 'download-returns' command for full functionality.
    """
    click.echo("⚠️  This command is deprecated.")
    click.echo("    Please use 'download-returns' instead:")
    click.echo("    pmrtn download-returns --rf-data data/raw/ESTR.csv --model KMeans")
    click.echo("\nFor help: pmrtn download-returns --help")


@click.command('generate-embeddings')
@click.option(
    '--config',
    type=click.Path(exists=True, path_type=Path),
    help='Path to configuration file',
)
@click.option(
    '--input',
    type=click.Path(exists=True, path_type=Path),
    help='Input path for processed articles CSV',
)
@click.option(
    '--output',
    type=click.Path(path_type=Path),
    help='Output path for embeddings CSV',
)
@click.option(
    '--model',
    type=click.Choice([
        'paraphrase-MiniLM-L6-v2',
        'paraphrase-multilingual-MiniLM-L12-v2',
        'distiluse-base-multilingual-cased-v1',
    ]),
    default='distiluse-base-multilingual-cased-v1',
    help='Sentence transformer model to use',
)
@click.option(
    '--batch-size',
    type=int,
    default=32,
    help='Batch size for embedding generation',
)
@click.pass_context
def generate_embeddings(
    ctx: click.Context,
    config: Optional[Path],
    input: Optional[Path],
    output: Optional[Path],
    model: str,
    batch_size: int,
) -> None:
    """Generate sentence embeddings for articles.
    
    This command loads processed articles and generates sentence embeddings
    using transformer models. The embeddings are used for clustering and
    similarity analysis.
    
    Equivalent to script: 3_data_embeddings.py
    """
    logger.info("Starting embedding generation")
    
    # Get configuration
    config_path = config or ctx.obj.get('config_path')
    settings = get_settings(config_path)
    path_manager = get_path_manager()
    
    # Determine paths
    input_path = input or (path_manager.processed_data / 'articles_with_tickers.csv')
    output_path = output or (path_manager.processed_data / 'articles_with_embeddings.csv')
    
    logger.info(f"Loading articles from: {input_path}")
    df = load_processed_articles(input_path)
    
    logger.info(f"Loaded {len(df)} articles")
    logger.info(f"Using model: {model}")
    logger.info(f"Batch size: {batch_size}")
    
    # Generate embeddings
    click.echo("Generating embeddings (this may take a while)...")
    df_with_embeddings = add_embeddings_to_dataframe(
        df,
        text_column='articles',
        model_name=model,
        batch_size=batch_size,
        show_progress=True,
    )
    
    # Save results
    logger.info(f"Saving embeddings to: {output_path}")
    save_processed_data(df_with_embeddings, output_path)
    
    click.echo(f"\n✓ Successfully generated embeddings for {len(df_with_embeddings)} articles")
    click.echo(f"  Model: {model}")
    click.echo(f"  Output saved to: {output_path}")
