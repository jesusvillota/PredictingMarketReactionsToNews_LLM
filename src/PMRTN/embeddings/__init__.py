"""Embeddings module for generating article embeddings."""

from .generators import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    EmbeddingGeneratorError,
    add_embeddings_to_dataframe,
    clear_model_cache,
    generate_embeddings,
    get_embedding,
    get_embedding_dimension,
    get_model,
)

__all__ = [
    'AVAILABLE_MODELS',
    'DEFAULT_MODEL',
    'EmbeddingGeneratorError',
    'add_embeddings_to_dataframe',
    'clear_model_cache',
    'generate_embeddings',
    'get_embedding',
    'get_embedding_dimension',
    'get_model',
]
