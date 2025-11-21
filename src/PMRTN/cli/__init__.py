"""Command-line interface for PMRTN (Predicting Market Reactions to News).

This module provides the CLI for running the PMRTN analysis pipeline.
The main entry point is the `cli` function in main.py, which is registered
as the `pmrtn` command via pyproject.toml (with `news-analysis` kept as a
temporary alias for backward compatibility).

Available commands:
- load-articles: Load and process raw articles
- describe-data: Generate descriptive statistics
- fetch-tickers: Fetch stock data from Yahoo Finance
- generate-embeddings: Generate sentence embeddings
- kmeans-clustering: Perform KMeans clustering analysis
- llama-parse: Parse articles with LLAMA models
- llama-clustering: Perform LLAMA-based clustering
- run-all: Execute the complete pipeline

Usage:
    pmrtn --help
    pmrtn load-articles --help
    pmrtn run-all --config config.yaml
"""

from PMRTN.cli.main import cli

__all__ = ['cli']

from .main import cli

__all__ = ["cli"]
