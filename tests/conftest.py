"""Pytest configuration and fixtures for testing."""

import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_data() -> dict:
    """Provide sample configuration data for testing."""
    return {
        "directories": {
            "raw_data": "data/raw",
            "processed_data": "data/processed",
            "output": "output",
            "output_descriptives": "output/descriptives",
            "output_kmeans": "output/kmeans",
            "output_llama": "output/llama",
        },
        "data_files": {
            "raw_articles": "ibex_sample.pqt.gzip",
            "processed_articles": "D.csv",
            "embeddings": "D_embeddings.csv",
            "returns_kmeans": "R_KMeans.csv",
            "returns_llama": "R_LLAMA.csv",
        },
    }
