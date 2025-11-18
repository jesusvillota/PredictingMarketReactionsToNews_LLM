# Predicting Market Reactions to News: An LLM-Based Approach Using Spanish Business Articles

## Project Description
This project predicts market reactions to news by analyzing Spanish business articles using a large language model (LLM). The workflow involves data collection, preprocessing, clustering, and evaluation using various machine learning techniques.

## Directory Structure
The project follows this directory structure:

- `data/raw` - Contains raw data files
- `data/processed` - Contains processed data files
- `notebooks` - Jupyter notebooks for each step of the analysis
- `output/descriptives` - Output related to data description
- `output/kmeans` - Output related to KMeans clustering
- `output/llama` - Output related to LLM-based clustering
- `config.yaml` - Configuration file for directory paths and other settings
- `requirements.txt` - List of dependencies
- `main.py` - Main script to run all notebooks sequentially


## Requirements

### Prerequisites
- Python 3.x
- Jupyter Notebook

## Runnning the project

This project is composed of a set of 6 notebooks

- `0_data_articles.ipynb`
- `1_data_description.ipynb`
- `2_data_tickers.ipynb`
- `3_data_embeddings.ipynb`
- `4_kmeans_clustering.ipynb`
- `5_0_llama_news_parser.ipynb`
- `5_llama_clustering.ipynb`

The `main.py` script executes the notebooks sequentially. 
    - Note that notebook `5_0_llama_news_parser.ipynb` is commented out because it takes >20 hours to run.
    - If you also want to run this notebook, uncomment it from the `notebook` list in `main.py`

