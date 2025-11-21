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
        'download-returns',
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
@click.option(
    '--rf-data',
    type=click.Path(exists=True, path_type=Path),
    help='Path to risk-free rate CSV file (ESTR.csv) - required for download-returns step',
)
@click.option(
    '--model-type',
    type=click.Choice(['KMeans', 'LLAMA', 'both'], case_sensitive=False),
    default='both',
    help='Which model type to run: KMeans, LLAMA, or both (default: both)',
)
@click.pass_context
def run_all(
    ctx: click.Context,
    config: Optional[Path],
    skip: tuple,
    llama_api_key: Optional[str],
    rf_data: Optional[Path],
    model_type: str,
) -> None:
    """Run the complete analysis pipeline.
    
    This command executes all steps of the analysis pipeline in sequence:
    
    1. load-articles: Load and process raw articles
    2. describe-data: Generate descriptive statistics
    3. download-returns: Download stock returns data from Yahoo Finance
    4. generate-embeddings: Generate sentence embeddings
    5. kmeans-clustering: Perform KMeans clustering (if model-type is KMeans or both)
    6. llama-parse: Parse articles with LLAMA (if model-type is LLAMA or both, requires API key)
    7. llama-clustering: Perform LLAMA-based clustering (if model-type is LLAMA or both)
    
    You can skip specific steps using the --skip option. For example:
    
        pmrtn run-all --skip llama-parse --skip llama-clustering --rf-data data/raw/ESTR.csv
    
    Note: LLAMA steps require a Groq API key and may take many hours to run.
    Note: The download-returns step requires --rf-data to be specified unless skipped.
    """
    logger.info("Starting full analysis pipeline")
    
    skip_steps = set(skip)
    
    # Validate required parameters
    if 'download-returns' not in skip_steps and not rf_data:
        click.echo("❌ Error: --rf-data is required for download-returns step", err=True)
        click.echo("  Either provide --rf-data or skip the step with --skip download-returns", err=True)
        raise click.Abort()
    
    # Determine which model steps to run
    run_kmeans = model_type.lower() in ['kmeans', 'both']
    run_llama = model_type.lower() in ['llama', 'both']
    
    # Display pipeline overview
    click.echo("\n" + "=" * 70)
    click.echo("NEWS MARKET ANALYSIS - FULL PIPELINE")
    click.echo("=" * 70)
    
    steps = [
        ('load-articles', 'Load and process raw articles', True),
        ('describe-data', 'Generate descriptive statistics', True),
        ('download-returns', 'Download stock returns from Yahoo Finance', True),
        ('generate-embeddings', 'Generate sentence embeddings', True),
        ('kmeans-clustering', 'Perform KMeans clustering analysis', run_kmeans),
        ('llama-parse', 'Parse articles with LLAMA (slow, requires API key)', run_llama),
        ('llama-clustering', 'Perform LLAMA clustering analysis', run_llama),
    ]
    
    click.echo("\nPipeline steps:")
    for i, (step_name, description, should_run) in enumerate(steps, 1):
        if not should_run:
            status = "SKIP (model)"
        elif step_name in skip_steps:
            status = "SKIP (user)"
        else:
            status = "RUN"
        click.echo(f"  {i}. [{status:12}] {step_name}: {description}")
    
    click.echo("\n" + "=" * 70)
    
    # Confirm before running
    if not click.confirm("\nDo you want to proceed?"):
        click.echo("Pipeline cancelled")
        return
    
    # Import commands
    from PMRTN.cli.data_commands import (
        load_articles,
        describe_data,
        download_returns,
        generate_embeddings,
    )
    from PMRTN.cli.clustering_commands import kmeans_clustering
    from PMRTN.cli.llama_commands import llama_parse, llama_clustering
    
    # Prepare context with config
    ctx.obj = ctx.obj or {}
    if config:
        ctx.obj['config_path'] = config
    
    # Execute steps in sequence
    step_number = 0
    total_steps = len([s for s, _, should_run in steps if should_run and s[0] not in skip_steps])
    
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
        
        # Step 3: Download returns (replaces fetch-tickers)
        if 'download-returns' not in skip_steps:
            step_number += 1
            click.echo(f"\n{'='*70}")
            click.echo(f"STEP {step_number}/{total_steps}: Download stock returns")
            click.echo('='*70)
            
            # Download for KMeans if needed
            if run_kmeans:
                click.echo("\n→ Downloading returns for KMeans model...")
                ctx.invoke(download_returns, config=config, rf_data=rf_data, model='KMeans')
            
            # Download for LLAMA if needed
            if run_llama:
                click.echo("\n→ Downloading returns for LLAMA model...")
                ctx.invoke(download_returns, config=config, rf_data=rf_data, model='LLAMA')
        
        # Step 4: Generate embeddings
        if 'generate-embeddings' not in skip_steps:
            step_number += 1
            click.echo(f"\n{'='*70}")
            click.echo(f"STEP {step_number}/{total_steps}: Generate embeddings")
            click.echo('='*70)
            ctx.invoke(generate_embeddings, config=config)
        
        # Step 5: KMeans clustering
        if run_kmeans and 'kmeans-clustering' not in skip_steps:
            step_number += 1
            click.echo(f"\n{'='*70}")
            click.echo(f"STEP {step_number}/{total_steps}: KMeans clustering")
            click.echo('='*70)
            ctx.invoke(kmeans_clustering, config=config)
        
        # Step 6: LLAMA parse
        if run_llama and 'llama-parse' not in skip_steps:
            step_number += 1
            click.echo(f"\n{'='*70}")
            click.echo(f"STEP {step_number}/{total_steps}: LLAMA news parsing")
            click.echo('='*70)
            
            if not llama_api_key:
                click.echo("⚠️  Warning: LLAMA API key not provided, skipping...")
            else:
                ctx.invoke(llama_parse, config=config, api_key=llama_api_key)
        
        # Step 7: LLAMA clustering
        if run_llama and 'llama-clustering' not in skip_steps:
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
