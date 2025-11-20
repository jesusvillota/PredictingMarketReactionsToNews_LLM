"""CLI commands for data loading and processing."""

import logging
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from tqdm import tqdm

from news_market_analysis.config import get_settings, get_path_manager
from news_market_analysis.data import (
    load_raw_articles,
    process_articles,
    save_processed_data,
    load_processed_articles,
)
from news_market_analysis.embeddings import add_embeddings_to_dataframe
from news_market_analysis.visualization import (
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
    """Fetch stock price data from Yahoo Finance.
    
    This command extracts unique tickers from processed articles and downloads
    historical price data from Yahoo Finance. It calculates returns and prepares
    data for event study analysis.
    
    Equivalent to script: 2_data_tickers.py
    
    Note: This is a simplified implementation. Full functionality requires
    implementing yfinance integration and event study calculations.
    """
    logger.info("Starting ticker data fetching")
    
    # Get configuration
    config_path = config or ctx.obj.get('config_path')
    settings = get_settings(config_path)
    path_manager = get_path_manager()
    
    # Determine paths
    input_path = input or (path_manager.processed_data / 'articles_with_tickers.csv')
    output_path = output or (path_manager.processed_data / 'returns_data.csv')
    
    logger.info(f"Loading articles from: {input_path}")
    df = load_processed_articles(input_path)
    
    # Extract unique tickers
    unique_tickers = df['tickers'].explode().unique()
    logger.info(f"Found {len(unique_tickers)} unique tickers")
    
    click.echo(f"\n⚠️  Ticker fetching requires Yahoo Finance integration")
    click.echo(f"   This is a placeholder command for the full implementation")
    click.echo(f"   Tickers to fetch: {len(unique_tickers)}")
    
    # TODO: Implement full yfinance integration
    # - Download price data for each ticker
    # - Calculate returns
    # - Perform event study
    # - Save results
    
    click.echo(f"\n✓ Ticker analysis prepared")


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
