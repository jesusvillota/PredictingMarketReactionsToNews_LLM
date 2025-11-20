"""CLI commands for running the full analysis pipeline."""

import logging
from pathlib import Path
from typing import Optional, List

import click

logger = logging.getLogger(__name__)


@click.command('run-all')
@click.option(
    '--config',
    type=click.Path(exists=True, path_type=Path),
    help='Path to configuration file',
)
@click.option(
    '--skip',
    multiple=True,
    type=click.Choice([
        'load-articles',
        'describe-data',
        'fetch-tickers',
        'generate-embeddings',
        'kmeans-clustering',
        'llama-parse',
        'llama-clustering',
    ]),
    help='Steps to skip (can be specified multiple times)',
)
@click.option(
    '--llama-api-key',
    type=str,
    envvar='GROQ_API_KEY',
    help='Groq API key for LLAMA steps',
)
@click.pass_context
def run_all(
    ctx: click.Context,
    config: Optional[Path],
    skip: tuple,
    llama_api_key: Optional[str],
) -> None:
    """Run the complete analysis pipeline.
    
    This command executes all steps of the analysis pipeline in sequence:
    
    1. load-articles: Load and process raw articles
    2. describe-data: Generate descriptive statistics
    3. fetch-tickers: Fetch stock data from Yahoo Finance
    4. generate-embeddings: Generate sentence embeddings
    5. kmeans-clustering: Perform KMeans clustering
    6. llama-parse: Parse articles with LLAMA (requires API key)
    7. llama-clustering: Perform LLAMA-based clustering
    
    You can skip specific steps using the --skip option. For example:
    
        news-analysis run-all --skip llama-parse --skip llama-clustering
    
    Note: LLAMA steps require a Groq API key and may take many hours to run.
    """
    logger.info("Starting full analysis pipeline")
    
    skip_steps = set(skip)
    
    # Display pipeline overview
    click.echo("\n" + "=" * 70)
    click.echo("NEWS MARKET ANALYSIS - FULL PIPELINE")
    click.echo("=" * 70)
    
    steps = [
        ('load-articles', 'Load and process raw articles'),
        ('describe-data', 'Generate descriptive statistics'),
        ('fetch-tickers', 'Fetch stock data from Yahoo Finance'),
        ('generate-embeddings', 'Generate sentence embeddings'),
        ('kmeans-clustering', 'Perform KMeans clustering analysis'),
        ('llama-parse', 'Parse articles with LLAMA (slow, requires API key)'),
        ('llama-clustering', 'Perform LLAMA clustering analysis'),
    ]
    
    click.echo("\nPipeline steps:")
    for i, (step_name, description) in enumerate(steps, 1):
        status = "SKIP" if step_name in skip_steps else "RUN"
        click.echo(f"  {i}. [{status:4}] {step_name}: {description}")
    
    click.echo("\n" + "=" * 70)
    
    # Confirm before running
    if not click.confirm("\nDo you want to proceed?"):
        click.echo("Pipeline cancelled")
        return
    
    # Import commands
    from news_market_analysis.cli.data_commands import (
        load_articles,
        describe_data,
        fetch_tickers,
        generate_embeddings,
    )
    from news_market_analysis.cli.clustering_commands import kmeans_clustering
    from news_market_analysis.cli.llama_commands import llama_parse, llama_clustering
    
    # Prepare context with config
    ctx.obj = ctx.obj or {}
    if config:
        ctx.obj['config_path'] = config
    
    # Execute steps in sequence
    step_number = 0
    total_steps = len([s for s in steps if s[0] not in skip_steps])
    
    try:
        # Step 1: Load articles
        if 'load-articles' not in skip_steps:
            step_number += 1
            click.echo(f"\n{'='*70}")
            click.echo(f"STEP {step_number}/{total_steps}: Load and process articles")
            click.echo('='*70)
            ctx.invoke(load_articles, config=config)
        
        # Step 2: Describe data
        if 'describe-data' not in skip_steps:
            step_number += 1
            click.echo(f"\n{'='*70}")
            click.echo(f"STEP {step_number}/{total_steps}: Generate descriptive statistics")
            click.echo('='*70)
            ctx.invoke(describe_data, config=config)
        
        # Step 3: Fetch tickers
        if 'fetch-tickers' not in skip_steps:
            step_number += 1
            click.echo(f"\n{'='*70}")
            click.echo(f"STEP {step_number}/{total_steps}: Fetch stock data")
            click.echo('='*70)
            ctx.invoke(fetch_tickers, config=config)
        
        # Step 4: Generate embeddings
        if 'generate-embeddings' not in skip_steps:
            step_number += 1
            click.echo(f"\n{'='*70}")
            click.echo(f"STEP {step_number}/{total_steps}: Generate embeddings")
            click.echo('='*70)
            ctx.invoke(generate_embeddings, config=config)
        
        # Step 5: KMeans clustering
        if 'kmeans-clustering' not in skip_steps:
            step_number += 1
            click.echo(f"\n{'='*70}")
            click.echo(f"STEP {step_number}/{total_steps}: KMeans clustering")
            click.echo('='*70)
            ctx.invoke(kmeans_clustering, config=config)
        
        # Step 6: LLAMA parse
        if 'llama-parse' not in skip_steps:
            step_number += 1
            click.echo(f"\n{'='*70}")
            click.echo(f"STEP {step_number}/{total_steps}: LLAMA news parsing")
            click.echo('='*70)
            
            if not llama_api_key:
                click.echo("⚠️  Warning: LLAMA API key not provided, skipping...")
            else:
                ctx.invoke(llama_parse, config=config, api_key=llama_api_key)
        
        # Step 7: LLAMA clustering
        if 'llama-clustering' not in skip_steps:
            step_number += 1
            click.echo(f"\n{'='*70}")
            click.echo(f"STEP {step_number}/{total_steps}: LLAMA clustering")
            click.echo('='*70)
            ctx.invoke(llama_clustering, config=config)
        
        # Success message
        click.echo(f"\n{'='*70}")
        click.echo("✓ PIPELINE COMPLETE!")
        click.echo('='*70)
        click.echo(f"\nAll {step_number} steps executed successfully.")
        click.echo("Check the output directories for results.")
        
    except Exception as e:
        click.echo(f"\n❌ Pipeline failed at step {step_number}", err=True)
        click.echo(f"Error: {str(e)}", err=True)
        logger.exception("Pipeline execution failed")
        raise click.Abort()
