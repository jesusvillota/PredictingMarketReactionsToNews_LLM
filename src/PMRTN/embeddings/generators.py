"""Embedding generation utilities using transformer models."""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# Model registry - lazy-loaded models to save memory
_model_cache: Dict[str, SentenceTransformer] = {}


# Available transformer models for embedding generation
AVAILABLE_MODELS = [
    'paraphrase-MiniLM-L6-v2',
    'paraphrase-multilingual-MiniLM-L12-v2',
    'distiluse-base-multilingual-cased-v1',
]

DEFAULT_MODEL = 'distiluse-base-multilingual-cased-v1'


class EmbeddingGeneratorError(Exception):
    """Raised when embedding generation fails."""
    pass


def get_model(model_name: str) -> SentenceTransformer:
    """Get or load a SentenceTransformer model.
    
    Models are cached after first load to avoid reloading. This is particularly
    important when processing many articles, as model loading is expensive.
    
    Args:
        model_name: Name of the SentenceTransformer model to load.
                   Must be one of AVAILABLE_MODELS.
    
    Returns:
        SentenceTransformer: The loaded model instance.
    
    Raises:
        EmbeddingGeneratorError: If model_name is not in AVAILABLE_MODELS.
    
    Example:
        >>> model = get_model('paraphrase-MiniLM-L6-v2')
        >>> embeddings = model.encode(['Hello world'])
    """
    if model_name not in AVAILABLE_MODELS:
        raise EmbeddingGeneratorError(
            f"Model '{model_name}' not available. "
            f"Available models: {', '.join(AVAILABLE_MODELS)}"
        )
    
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    
    return _model_cache[model_name]


def clear_model_cache() -> None:
    """Clear the model cache to free memory.
    
    Useful when switching between different models or when memory is constrained.
    Models will be reloaded on next use.
    
    Example:
        >>> generate_embeddings(texts, 'paraphrase-MiniLM-L6-v2')
        >>> clear_model_cache()  # Free memory
        >>> generate_embeddings(texts, 'distiluse-base-multilingual-cased-v1')
    """
    global _model_cache
    _model_cache.clear()


def get_embedding(
    article: str,
    model_name: str = DEFAULT_MODEL
) -> List[float]:
    """Generate embedding vector for a single article.
    
    Args:
        article: The text content to generate embeddings for.
        model_name: Name of the transformer model to use. Defaults to
                   'distiluse-base-multilingual-cased-v1'.
    
    Returns:
        List[float]: Embedding vector as a list of floats. Length depends on
                    the model (typically 384 or 512 dimensions).
    
    Raises:
        EmbeddingGeneratorError: If article is empty or model fails.
    
    Example:
        >>> article = "Apple stock surges after earnings report."
        >>> embedding = get_embedding(article)
        >>> len(embedding)
        512
    """
    if not article or not article.strip():
        raise EmbeddingGeneratorError("Article text cannot be empty")
    
    try:
        model = get_model(model_name)
        embedding = model.encode(article)
        return embedding.tolist()
    except Exception as e:
        raise EmbeddingGeneratorError(
            f"Failed to generate embedding for article: {str(e)}"
        ) from e


def generate_embeddings(
    texts: Union[List[str], pd.Series],
    model_name: str = DEFAULT_MODEL,
    show_progress: bool = True,
    batch_size: int = 32
) -> List[List[float]]:
    """Generate embeddings for multiple texts efficiently.
    
    Uses batch processing for better performance when generating embeddings
    for many texts.
    
    Args:
        texts: List or Series of text strings to generate embeddings for.
        model_name: Name of the transformer model to use.
        show_progress: Whether to show progress bar during generation.
        batch_size: Number of texts to process in each batch.
    
    Returns:
        List[List[float]]: List of embedding vectors, one per input text.
    
    Raises:
        EmbeddingGeneratorError: If texts is empty or generation fails.
    
    Example:
        >>> articles = ["Article 1 text", "Article 2 text", "Article 3 text"]
        >>> embeddings = generate_embeddings(articles)
        >>> len(embeddings)
        3
    """
    # Check if texts is empty (handle both list and Series)
    if isinstance(texts, pd.Series):
        if texts.empty:
            raise EmbeddingGeneratorError("No texts provided for embedding generation")
        texts = texts.tolist()
    elif not texts or len(texts) == 0:
        raise EmbeddingGeneratorError("No texts provided for embedding generation")
    
    # Validate all texts are non-empty strings
    for i, text in enumerate(texts):
        if not text or not isinstance(text, str) or not text.strip():
            raise EmbeddingGeneratorError(
                f"Invalid text at index {i}: must be non-empty string"
            )
    
    try:
        model = get_model(model_name)
        embeddings = model.encode(
            texts,
            show_progress_bar=show_progress,
            batch_size=batch_size,
            convert_to_numpy=True
        )
        return [emb.tolist() for emb in embeddings]
    except Exception as e:
        raise EmbeddingGeneratorError(
            f"Failed to generate embeddings: {str(e)}"
        ) from e


def add_embeddings_to_dataframe(
    df: pd.DataFrame,
    text_column: str = 'articles',
    embedding_column: str = 'embeddings',
    model_name: str = DEFAULT_MODEL,
    show_progress: bool = True,
    batch_size: int = 32
) -> pd.DataFrame:
    """Add embeddings column to a DataFrame containing text data.
    
    Creates a new column with embeddings for the specified text column.
    Returns a copy of the DataFrame with the new column added.
    
    Args:
        df: DataFrame containing text data.
        text_column: Name of column containing text to embed.
        embedding_column: Name for the new embeddings column.
        model_name: Name of the transformer model to use.
        show_progress: Whether to show progress bar during generation.
        batch_size: Number of texts to process in each batch.
    
    Returns:
        pd.DataFrame: Copy of input DataFrame with embeddings column added.
    
    Raises:
        EmbeddingGeneratorError: If text_column doesn't exist or generation fails.
    
    Example:
        >>> df = pd.DataFrame({'articles': ['Text 1', 'Text 2']})
        >>> df_with_embeddings = add_embeddings_to_dataframe(df)
        >>> 'embeddings' in df_with_embeddings.columns
        True
    """
    if text_column not in df.columns:
        raise EmbeddingGeneratorError(
            f"Column '{text_column}' not found in DataFrame. "
            f"Available columns: {', '.join(df.columns)}"
        )
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Generate embeddings for all texts
    embeddings = generate_embeddings(
        df[text_column],
        model_name=model_name,
        show_progress=show_progress,
        batch_size=batch_size
    )
    
    # Add to DataFrame
    df_copy[embedding_column] = embeddings
    
    return df_copy


def get_embedding_dimension(model_name: str = DEFAULT_MODEL) -> int:
    """Get the dimensionality of embeddings produced by a model.
    
    Args:
        model_name: Name of the transformer model.
    
    Returns:
        int: Number of dimensions in the embedding vector.
    
    Raises:
        EmbeddingGeneratorError: If model cannot be loaded.
    
    Example:
        >>> dim = get_embedding_dimension('paraphrase-MiniLM-L6-v2')
        >>> dim
        384
    """
    try:
        model = get_model(model_name)
        # Get dimension by encoding a test string
        test_embedding = model.encode(["test"])
        return test_embedding.shape[1]
    except Exception as e:
        raise EmbeddingGeneratorError(
            f"Failed to determine embedding dimension: {str(e)}"
        ) from e
