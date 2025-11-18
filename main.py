#!/usr/bin/env python3
"""Main entry point for the replication pipeline."""

import argparse
import sys
from pathlib import Path

from src.config import setup_logger, get_logger, get_paths
from src.data.load_articles import load_and_process_articles
from src.data.process_tickers import process_ticker_data
from src.embeddings.generate import generate_embeddings
from src.analysis.descriptives import generate_descriptives
from src.analysis.kmeans_clustering import perform_kmeans_clustering
from src.analysis.llama_clustering import perform_llama_clustering
from src.llm.llama_parser import parse_articles_with_llama


logger = get_logger(__name__)


def run_pipeline_step(step: str, **kwargs):
    """Run a specific pipeline step."""
    steps = {
        'load_articles': lambda: load_and_process_articles(**kwargs),
        'process_tickers': lambda: process_ticker_data(**kwargs),
        'generate_embeddings': lambda: generate_embeddings(**kwargs),
        'descriptives': lambda: generate_descriptives(**kwargs),
        'kmeans_clustering': lambda: perform_kmeans_clustering(**kwargs),
        'llama_clustering': lambda: perform_llama_clustering(**kwargs),
        'llama_parser': lambda: parse_articles_with_llama(**kwargs),
    }
    
    if step not in steps:
        raise ValueError(f"Unknown step: {step}. Available steps: {list(steps.keys())}")
    
    return steps[step]()


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Replication pipeline for Predicting Market Reactions to News"
    )
    
    parser.add_argument(
        '--step',
        type=str,
        choices=[
            'load_articles',
            'process_tickers',
            'generate_embeddings',
            'descriptives',
            'kmeans_clustering',
            'llama_clustering',
            'llama_parser',
            'all'
        ],
        default='all',
        help='Pipeline step to run (default: all)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['KMeans', 'LLAMA'],
        help='Model to use for ticker processing (KMeans or LLAMA)'
    )
    
    parser.add_argument(
        '--raw-data-path',
        type=Path,
        help='Path to raw data directory (overrides config)'
    )
    
    parser.add_argument(
        '--processed-data-path',
        type=Path,
        help='Path to processed data directory (overrides config)'
    )
    
    parser.add_argument(
        '--output-path',
        type=Path,
        help='Path to output directory (overrides config)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save outputs to files'
    )
    
    parser.add_argument(
        '--groq-api-key',
        type=str,
        help='Groq API key for LLAMA parser (or set GROQ_API_KEY env var)'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logger()
    logger.info("=" * 80)
    logger.info("Starting replication pipeline")
    logger.info("=" * 80)
    
    # Initialize path manager
    path_manager = get_paths()
    path_manager.create_directories()
    
    # Prepare kwargs for pipeline steps
    kwargs = {
        'save_output': not args.no_save
    }
    
    if args.raw_data_path:
        kwargs['raw_data_path'] = args.raw_data_path
    if args.processed_data_path:
        kwargs['processed_data_path'] = args.processed_data_path
    if args.output_path:
        kwargs['output_path'] = args.output_path
    
    try:
        if args.step == 'all':
            # Run full pipeline
            logger.info("Running full pipeline")
            
            # Step 1: Load and process articles
            logger.info("\n" + "=" * 80)
            logger.info("Step 1: Loading and processing articles")
            logger.info("=" * 80)
            load_and_process_articles(**kwargs)
            
            # Step 2: Process tickers for KMeans
            logger.info("\n" + "=" * 80)
            logger.info("Step 2: Processing tickers for KMeans")
            logger.info("=" * 80)
            process_ticker_data('KMeans', **kwargs)
            
            # Step 3: Generate embeddings
            logger.info("\n" + "=" * 80)
            logger.info("Step 3: Generating embeddings")
            logger.info("=" * 80)
            generate_embeddings(**kwargs)
            
            # Step 4: Generate descriptives
            logger.info("\n" + "=" * 80)
            logger.info("Step 4: Generating descriptive statistics")
            logger.info("=" * 80)
            generate_descriptives(**kwargs)
            
            # Step 5: KMeans clustering
            logger.info("\n" + "=" * 80)
            logger.info("Step 5: Performing KMeans clustering")
            logger.info("=" * 80)
            perform_kmeans_clustering(**kwargs)
            
            # Step 6: LLAMA parser (optional, requires API key)
            logger.info("\n" + "=" * 80)
            logger.info("Step 6: LLAMA parser (skipped - requires API key)")
            logger.info("=" * 80)
            logger.info("To run LLAMA parser, use: python main.py --step llama_parser --groq-api-key YOUR_KEY")
            
            # Step 7: LLAMA clustering (requires LLAMA parsed data)
            logger.info("\n" + "=" * 80)
            logger.info("Step 7: LLAMA clustering (skipped - requires LLAMA parsed data)")
            logger.info("=" * 80)
            logger.info("To run LLAMA clustering, first run LLAMA parser")
            
        elif args.step == 'process_tickers':
            if not args.model:
                logger.error("--model is required for process_tickers step")
                sys.exit(1)
            run_pipeline_step(args.step, model=args.model, **kwargs)
        
        elif args.step == 'llama_parser':
            if args.groq_api_key:
                kwargs['api_key'] = args.groq_api_key
            run_pipeline_step(args.step, **kwargs)
        
        else:
            run_pipeline_step(args.step, **kwargs)
        
        logger.info("\n" + "=" * 80)
        logger.info("Pipeline completed successfully")
        logger.info("=" * 80)
    
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

