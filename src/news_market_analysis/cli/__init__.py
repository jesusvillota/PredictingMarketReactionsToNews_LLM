"""Command-line interface for news market analysis.

This module provides the CLI for running the news market analysis pipeline.
The main entry point is the `cli` function in main.py, which is registered
as the `news-analysis` command via pyproject.toml.

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
    news-analysis --help
    news-analysis load-articles --help
    news-analysis run-all --config config.yaml
"""

from news_market_analysis.cli.main import cli

__all__ = ['cli']

from .main import cli

__all__ = ["cli"]
