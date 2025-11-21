"""Main CLI entry point for news market analysis.

This module provides the command-line interface for running the entire
news market analysis pipeline or individual components.
"""

import logging
from pathlib import Path
from typing import Optional

import click

from PMRTN.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option(
    '--config',
    type=click.Path(exists=True, path_type=Path),
    help='Path to configuration file (config.yaml)',
)
@click.option(
    '--verbose',
    '-v',
    is_flag=True,
    help='Enable verbose logging',
)
@click.pass_context
def cli(ctx: click.Context, config: Optional[Path], verbose: bool) -> None:
    """News Market Analysis - Predict market reactions to news articles.
    
    This CLI provides tools for analyzing how news articles affect stock market
    returns using machine learning and natural language processing.
    """
    # Ensure context dict exists
    ctx.ensure_object(dict)
    
    # Configure logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Store config path in context
    ctx.obj['config_path'] = config
    
    # Load settings if config provided
    if config:
        logger.info(f"Loading configuration from: {config}")
        ctx.obj['settings'] = get_settings(config)
    else:
        logger.debug("No config path provided, will use default search locations")


@cli.command()
@click.pass_context
def version(ctx: click.Context) -> None:
    """Display version information."""
    from PMRTN import __version__
    click.echo(f"PMRTN version {__version__}")


@cli.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """Display configuration and environment information."""
    from PMRTN.config import get_path_manager
    
    config_path = ctx.obj.get('config_path')
    settings = get_settings(config_path)
    path_manager = get_path_manager()
    
    click.echo("=" * 60)
    click.echo("News Market Analysis - Configuration Info")
    click.echo("=" * 60)
    click.echo(f"Config file: {settings._config_path}")
    click.echo(f"Base path: {path_manager.base_path}")
    click.echo(f"\nData directories:")
    click.echo(f"  Raw data: {path_manager.raw_data}")
    click.echo(f"  Processed data: {path_manager.processed_data}")
    click.echo(f"\nOutput directories:")
    click.echo(f"  Descriptives: {path_manager.output_descriptives}")
    click.echo(f"  KMeans: {path_manager.output_kmeans}")
    click.echo(f"  LLAMA: {path_manager.output_llama}")
    click.echo("=" * 60)


# Import subcommands from other modules
from PMRTN.cli.data_commands import (
    load_articles,
    describe_data,
    fetch_tickers,
    download_returns,
    generate_embeddings,
)
from PMRTN.cli.clustering_commands import (
    kmeans_clustering,
)
from PMRTN.cli.llama_commands import (
    llama_parse,
    llama_clustering,
)
from PMRTN.cli.pipeline_commands import (
    run_all,
)

# Register all subcommands
cli.add_command(load_articles)
cli.add_command(describe_data)
cli.add_command(fetch_tickers)
cli.add_command(download_returns)
cli.add_command(generate_embeddings)
cli.add_command(kmeans_clustering)
cli.add_command(llama_parse)
cli.add_command(llama_clustering)
cli.add_command(run_all)


if __name__ == '__main__':
    cli()
