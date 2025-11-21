"""CLI commands for LLAMA-based news parsing and analysis."""

import logging
import os
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from PMRTN.config import get_settings, get_path_manager
from PMRTN.data import load_processed_articles
from PMRTN.models import create_parser

logger = logging.getLogger(__name__)


@click.command('llama-parse')
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
    help='Output path for parsed articles CSV',
)
@click.option(
    '--api-key',
    type=str,
    envvar='GROQ_API_KEY',
    help='Groq API key (or set GROQ_API_KEY environment variable)',
)
@click.option(
    '--model',
    type=click.Choice(['llama3-70b-8192', 'llama3-8b-8192', 'llama2-70b-4096']),
    default='llama3-70b-8192',
    help='LLAMA model to use',
)
@click.option(
    '--max-articles',
    type=int,
    help='Maximum number of articles to parse (for testing)',
)
@click.option(
    '--retry-attempts',
    type=int,
    default=3,
    help='Number of retry attempts for failed API calls',
)
@click.pass_context
def llama_parse(
    ctx: click.Context,
    config: Optional[Path],
    input: Optional[Path],
    output: Optional[Path],
    api_key: Optional[str],
    model: str,
    max_articles: Optional[int],
    retry_attempts: int,
) -> None:
    """Parse news articles using LLAMA models.
    
    This command uses LLAMA models via the Groq API to extract structured
    information from news articles, including firm names, stock tickers,
    shock types, magnitudes, and directions.
    
    Equivalent to script: 5_0_llama_news_parser.py
    
    Note: This requires a Groq API key. Set it via --api-key option or
    GROQ_API_KEY environment variable.
    
    Warning: This operation can be slow (>20 hours for full dataset) and
    may incur API costs.
    """
    logger.info("Starting LLAMA news parsing")
    
    # Validate API key
    if not api_key:
        click.echo("❌ Error: Groq API key is required", err=True)
        click.echo("   Set GROQ_API_KEY environment variable or use --api-key option", err=True)
        raise click.Abort()
    
    # Get configuration
    config_path = config or ctx.obj.get('config_path')
    settings = get_settings(config_path)
    path_manager = get_path_manager()
    
    # Determine paths
    input_path = input or (path_manager.processed_data / 'articles_with_tickers.csv')
    output_path = output or (path_manager.output_llama / 'parsed_articles.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading articles from: {input_path}")
    df = load_processed_articles(input_path)
    
    # Limit articles if specified (for testing)
    if max_articles:
        df = df.head(max_articles)
        logger.info(f"Limited to {max_articles} articles for testing")
    
    logger.info(f"Parsing {len(df)} articles with LLAMA model: {model}")
    
    # Create parser
    parser = create_parser(
        api_key=api_key,
        model_name=model,
        max_retries=retry_attempts,
    )
    
    # Parse articles
    click.echo(f"\n⚠️  Warning: This operation may take a long time!")
    click.echo(f"   Estimated time: {len(df) * 2} seconds ({len(df) * 2 / 3600:.1f} hours)")
    click.echo(f"   Progress will be displayed below...\n")
    
    if not click.confirm("Do you want to continue?"):
        click.echo("Operation cancelled")
        return
    
    # Parse with progress bar
    df_parsed = parser.parse_dataframe(
        df,
        text_column='articles',
        show_progress=True,
    )
    
    # Save results
    logger.info(f"Saving parsed articles to: {output_path}")
    df_parsed.to_csv(output_path, index=False)
    
    # Report statistics
    parsed_count = df_parsed['parsed_firms'].notna().sum()
    click.echo(f"\n✓ LLAMA parsing complete")
    click.echo(f"  Total articles: {len(df_parsed)}")
    click.echo(f"  Successfully parsed: {parsed_count}")
    click.echo(f"  Failed: {len(df_parsed) - parsed_count}")
    click.echo(f"  Output saved to: {output_path}")


@click.command('llama-clustering')
@click.option(
    '--config',
    type=click.Path(exists=True, path_type=Path),
    help='Path to configuration file',
)
@click.option(
    '--input',
    type=click.Path(exists=True, path_type=Path),
    help='Input path for parsed articles CSV',
)
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    help='Output directory for clustering results',
)
@click.pass_context
def llama_clustering(
    ctx: click.Context,
    config: Optional[Path],
    input: Optional[Path],
    output_dir: Optional[Path],
) -> None:
    """Perform clustering analysis on LLAMA-parsed articles.
    
    This command takes LLAMA-parsed articles and performs clustering analysis
    based on the extracted shock types and magnitudes. It generates trading
    signals and backtesting results.
    
    Equivalent to script: 5_llama_clustering.py
    
    Note: This is a placeholder. Full implementation requires integrating
    the LLAMA clustering logic from the original script.
    """
    logger.info("Starting LLAMA clustering analysis")
    
    # Get configuration
    config_path = config or ctx.obj.get('config_path')
    settings = get_settings(config_path)
    path_manager = get_path_manager()
    
    # Determine paths
    input_path = input or (path_manager.output_llama / 'parsed_articles.csv')
    output_path = output_dir or path_manager.output_llama
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading parsed articles from: {input_path}")
    
    if not input_path.exists():
        click.echo(f"❌ Error: Input file not found: {input_path}", err=True)
        click.echo("   Run 'pmrtn llama-parse' first", err=True)
        raise click.Abort()
    
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} parsed articles")
    
    # TODO: Implement full LLAMA clustering logic
    # - Group articles by shock types
    # - Assign clusters based on shocks
    # - Generate trading signals
    # - Perform backtesting
    # - Calculate portfolio statistics
    # - Generate visualizations
    
    click.echo(f"\n⚠️  LLAMA clustering requires full implementation")
    click.echo(f"   This is a placeholder command")
    click.echo(f"   Articles loaded: {len(df)}")
    
    click.echo(f"\n✓ LLAMA clustering analysis prepared")
    click.echo(f"  Output directory: {output_path}")
