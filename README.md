# Predicting Market Reactions to News: Replication Code

## Overview

This is the replication repository for the paper "Predicting Market Reactions to News: An LLM-Based Approach Using Spanish Business Articles". The repository contains all code, data processing pipelines, and analysis scripts necessary to replicate the results presented in the paper.

## Contact

- Website: [jesusvillota.github.io](https://jesusvillota.github.io/)
- Email: jesus.villota@cemfi.edu.es
- LinkedIn: [jesusvillotamiranda](https://www.linkedin.com/in/jesusvillotamiranda/)

## Project Structure

```
PredictingMarketReactionsToNews_LLM/
├── pyproject.toml          # Project configuration and dependencies
├── README.md               # This file
├── config.yaml             # Configuration file (paths, parameters)
├── main.py                 # Main entry point
├── src/
│   ├── config/             # Configuration management
│   │   ├── config_settings.py
│   │   ├── logger.py
│   │   └── paths.py
│   ├── data/               # Data processing modules
│   │   ├── load_articles.py
│   │   ├── process_tickers.py
│   │   └── utils.py
│   ├── embeddings/         # Embedding generation
│   │   └── generate.py
│   ├── analysis/           # Analysis modules
│   │   ├── descriptives.py
│   │   ├── kmeans_clustering.py
│   │   └── llama_clustering.py
│   ├── llm/                # LLM parsing
│   │   └── llama_parser.py
│   └── utils/              # Utility modules
│       ├── text_processing.py
│       └── visualization.py
├── data/
│   ├── raw/                 # Raw data (gitignored)
│   └── processed/           # Processed data (gitignored)
├── output/                  # Analysis outputs (gitignored)
│   ├── descriptives/
│   ├── kmeans/
│   └── llama/
└── logs/                     # Log files (gitignored)
```

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for package management.

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Sync dependencies

This creates a virtual environment and installs all required packages:

```bash
uv sync
```

### 3. Activate the virtual environment

```bash
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows
```

### 4. Configure the project

Edit `config.yaml` to set up paths and parameters:

```yaml
directories:
  raw_data: "data/raw"
  processed_data: "data/processed"
  # ...

embedding:
  model: "distiluse-base-multilingual-cased-v1"
  # ...

clustering:
  kmeans:
    n_clusters: 10
```

## Usage

### Running the Full Pipeline

To run all steps of the replication pipeline:

```bash
uv run python main.py --step all
```

### Running Individual Steps

You can run specific steps of the pipeline:

```bash
# Load and process articles
uv run python main.py --step load_articles

# Process tickers for KMeans model
uv run python main.py --step process_tickers --model KMeans

# Generate embeddings
uv run python main.py --step generate_embeddings

# Generate descriptive statistics
uv run python main.py --step descriptives

# Perform KMeans clustering
uv run python main.py --step kmeans_clustering

# Parse articles with LLAMA (requires API key)
uv run python main.py --step llama_parser --groq-api-key YOUR_API_KEY
# Or set environment variable: export GROQ_API_KEY=YOUR_API_KEY

# Perform LLAMA clustering
uv run python main.py --step llama_clustering
```

### Command-Line Options

- `--step`: Pipeline step to run (default: `all`)
- `--model`: Model to use for ticker processing (`KMeans` or `LLAMA`)
- `--raw-data-path`: Override raw data directory path
- `--processed-data-path`: Override processed data directory path
- `--output-path`: Override output directory path
- `--no-save`: Do not save outputs to files
- `--groq-api-key`: Groq API key for LLAMA parser

## Pipeline Steps

1. **load_articles**: Load raw article data, filter, clean, and extract tickers
2. **process_tickers**: Fetch stock return data for tickers and prepare return datasets
3. **generate_embeddings**: Generate sentence embeddings for articles
4. **descriptives**: Generate descriptive statistics
5. **kmeans_clustering**: Perform KMeans clustering on article embeddings
6. **llama_parser**: Parse articles using LLAMA via Groq API (requires API key)
7. **llama_clustering**: Perform clustering analysis using LLAMA-parsed news

## Configuration

The `config.yaml` file contains all configuration settings:

- **directories**: Paths to data and output directories
- **logging**: Logging level and file settings
- **embedding**: Embedding model configuration
- **clustering**: Clustering parameters
- **data**: Data processing settings (closing time, market index, etc.)

## Dependencies

Key dependencies include:

- pandas, numpy: Data manipulation
- scikit-learn: Machine learning (clustering)
- sentence-transformers: Text embeddings
- yfinance: Stock market data
- groq: LLAMA API access
- matplotlib, seaborn: Visualization

See `pyproject.toml` for the complete list.

## Data Requirements

The pipeline expects the following raw data files:

- `ibex_sample.pqt.gziq`: Raw article data (parquet format)
- `ESTR.csv`: Risk-free rate data (€STR)
- `LLAMA_parsed_news.csv`: LLAMA-parsed news (if using LLAMA clustering)

Place these files in the `data/raw/` directory.

### Data Availability

**Important:** The article data used in this study is proprietary and cannot be publicly shared due to licensing restrictions. However, interested researchers can access the same data from [Factiva](https://www.dowjones.com/professional/factiva/) for a fee. The data consists of Spanish business news articles that can be obtained through Factiva's subscription service.

## Outputs

The pipeline generates outputs in the following directories:

- `data/processed/`: Processed datasets (D.csv, D_embeddings.csv, R_KMeans.csv, etc.)
- `output/descriptives/`: Descriptive statistics
- `output/kmeans/`: KMeans clustering results
- `output/llama/`: LLAMA clustering results
- `logs/`: Log files

## Logging

The project uses structured logging with colored console output. Logs are saved to `logs/predicting-market-reactions.log`. Logging level can be configured in `config.yaml`.

## Development

### Adding Dependencies

Edit `pyproject.toml` and run:

```bash
uv sync
```

### Running Scripts

```bash
uv run python <script_name>.py
```

### Code Structure

- Each module is self-contained with clear responsibilities
- Configuration is centralized in `src/config/`
- Logging is set up consistently across modules
- Path management is handled through `PathManager`

## Notes

- The LLAMA parser requires a Groq API key. Get one at https://console.groq.com/
- Some steps depend on previous steps (e.g., embeddings require processed articles)
- The pipeline preserves the original notebook logic while improving code organization
- Data files are gitignored - ensure you have the required raw data files

## Troubleshooting

### Missing Data Files

If you encounter file not found errors, ensure:
1. Raw data files are in `data/raw/`
2. File names match those expected by the code
3. Paths in `config.yaml` are correct

### API Key Issues

For LLAMA parser:
1. Set `GROQ_API_KEY` environment variable, or
2. Pass `--groq-api-key` argument

### Import Errors

Ensure the virtual environment is activated and dependencies are installed:

```bash
uv sync
source .venv/bin/activate
```

## License

All code and work associated with this project are solely created and authored by Jesus Villota Miranda. © 2024
