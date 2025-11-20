"""Tests for embeddings module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from news_market_analysis.embeddings import (
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


class TestGetModel:
    """Tests for get_model function."""

    def test_get_model_valid(self):
        """Test getting a valid model."""
        with patch('news_market_analysis.embeddings.generators.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model
            
            model = get_model('paraphrase-MiniLM-L6-v2')
            
            assert model == mock_model
            mock_st.assert_called_once_with('paraphrase-MiniLM-L6-v2')

    def test_get_model_invalid(self):
        """Test getting an invalid model raises error."""
        with pytest.raises(EmbeddingGeneratorError) as exc_info:
            get_model('invalid-model-name')
        
        assert 'not available' in str(exc_info.value)
        assert 'Available models' in str(exc_info.value)

    def test_get_model_caching(self):
        """Test that models are cached after first load."""
        with patch('news_market_analysis.embeddings.generators.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model
            
            # Clear cache first
            clear_model_cache()
            
            # First call should load model
            model1 = get_model('paraphrase-MiniLM-L6-v2')
            
            # Second call should use cached model
            model2 = get_model('paraphrase-MiniLM-L6-v2')
            
            # Should only be called once due to caching
            assert mock_st.call_count == 1
            assert model1 == model2


class TestClearModelCache:
    """Tests for clear_model_cache function."""

    def test_clear_model_cache(self):
        """Test clearing the model cache."""
        with patch('news_market_analysis.embeddings.generators.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model
            
            # Clear cache first to start fresh
            clear_model_cache()
            
            # Load a model
            get_model('paraphrase-MiniLM-L6-v2')
            call_count_1 = mock_st.call_count
            
            # Clear cache
            clear_model_cache()
            
            # Load again - should reload
            get_model('paraphrase-MiniLM-L6-v2')
            call_count_2 = mock_st.call_count
            
            # Should be called twice (once before and once after clear)
            assert call_count_2 == call_count_1 + 1


class TestGetEmbedding:
    """Tests for get_embedding function."""

    def test_get_embedding_success(self):
        """Test generating embedding for a single article."""
        with patch('news_market_analysis.embeddings.generators.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_embedding = np.array([0.1, 0.2, 0.3])
            mock_model.encode.return_value = mock_embedding
            mock_get_model.return_value = mock_model
            
            article = "Apple stock surges after earnings."
            embedding = get_embedding(article)
            
            assert isinstance(embedding, list)
            assert len(embedding) == 3
            assert embedding == [0.1, 0.2, 0.3]
            mock_model.encode.assert_called_once_with(article)

    def test_get_embedding_with_custom_model(self):
        """Test generating embedding with custom model."""
        with patch('news_market_analysis.embeddings.generators.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_embedding = np.array([0.1, 0.2])
            mock_model.encode.return_value = mock_embedding
            mock_get_model.return_value = mock_model
            
            embedding = get_embedding("Test article", model_name='paraphrase-MiniLM-L6-v2')
            
            mock_get_model.assert_called_once_with('paraphrase-MiniLM-L6-v2')
            assert len(embedding) == 2

    def test_get_embedding_empty_article(self):
        """Test that empty article raises error."""
        with pytest.raises(EmbeddingGeneratorError) as exc_info:
            get_embedding("")
        
        assert "cannot be empty" in str(exc_info.value)

    def test_get_embedding_whitespace_only(self):
        """Test that whitespace-only article raises error."""
        with pytest.raises(EmbeddingGeneratorError) as exc_info:
            get_embedding("   \n\t  ")
        
        assert "cannot be empty" in str(exc_info.value)

    def test_get_embedding_model_error(self):
        """Test handling of model encoding errors."""
        with patch('news_market_analysis.embeddings.generators.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.encode.side_effect = RuntimeError("Model error")
            mock_get_model.return_value = mock_model
            
            with pytest.raises(EmbeddingGeneratorError) as exc_info:
                get_embedding("Test article")
            
            assert "Failed to generate embedding" in str(exc_info.value)


class TestGenerateEmbeddings:
    """Tests for generate_embeddings function."""

    def test_generate_embeddings_list(self):
        """Test generating embeddings for list of texts."""
        with patch('news_market_analysis.embeddings.generators.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
            mock_model.encode.return_value = mock_embeddings
            mock_get_model.return_value = mock_model
            
            texts = ["Article 1", "Article 2", "Article 3"]
            embeddings = generate_embeddings(texts, show_progress=False)
            
            assert isinstance(embeddings, list)
            assert len(embeddings) == 3
            assert all(isinstance(emb, list) for emb in embeddings)
            assert embeddings[0] == [0.1, 0.2]
            assert embeddings[1] == [0.3, 0.4]
            assert embeddings[2] == [0.5, 0.6]

    def test_generate_embeddings_series(self):
        """Test generating embeddings for pandas Series."""
        with patch('news_market_analysis.embeddings.generators.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
            mock_model.encode.return_value = mock_embeddings
            mock_get_model.return_value = mock_model
            
            texts = pd.Series(["Article 1", "Article 2"])
            embeddings = generate_embeddings(texts, show_progress=False)
            
            assert len(embeddings) == 2
            assert embeddings[0] == [0.1, 0.2]

    def test_generate_embeddings_empty_list(self):
        """Test that empty list raises error."""
        with pytest.raises(EmbeddingGeneratorError) as exc_info:
            generate_embeddings([])
        
        assert "No texts provided" in str(exc_info.value)

    def test_generate_embeddings_with_empty_text(self):
        """Test that list with empty text raises error."""
        with pytest.raises(EmbeddingGeneratorError) as exc_info:
            generate_embeddings(["Valid text", "", "Another valid text"])
        
        assert "Invalid text at index" in str(exc_info.value)

    def test_generate_embeddings_with_non_string(self):
        """Test that non-string in list raises error."""
        with pytest.raises(EmbeddingGeneratorError) as exc_info:
            generate_embeddings(["Valid text", None, "Another valid text"])
        
        assert "Invalid text at index" in str(exc_info.value)

    def test_generate_embeddings_custom_batch_size(self):
        """Test generating embeddings with custom batch size."""
        with patch('news_market_analysis.embeddings.generators.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
            mock_model.encode.return_value = mock_embeddings
            mock_get_model.return_value = mock_model
            
            texts = ["Article 1", "Article 2"]
            generate_embeddings(texts, show_progress=False, batch_size=1)
            
            # Check that batch_size was passed
            call_kwargs = mock_model.encode.call_args[1]
            assert call_kwargs['batch_size'] == 1

    def test_generate_embeddings_show_progress(self):
        """Test that progress bar parameter is passed correctly."""
        with patch('news_market_analysis.embeddings.generators.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_embeddings = np.array([[0.1, 0.2]])
            mock_model.encode.return_value = mock_embeddings
            mock_get_model.return_value = mock_model
            
            texts = ["Article 1"]
            generate_embeddings(texts, show_progress=True)
            
            call_kwargs = mock_model.encode.call_args[1]
            assert call_kwargs['show_progress_bar'] is True


class TestAddEmbeddingsToDataframe:
    """Tests for add_embeddings_to_dataframe function."""

    def test_add_embeddings_default_columns(self):
        """Test adding embeddings with default column names."""
        with patch('news_market_analysis.embeddings.generators.generate_embeddings') as mock_gen:
            mock_gen.return_value = [[0.1, 0.2], [0.3, 0.4]]
            
            df = pd.DataFrame({'articles': ['Text 1', 'Text 2']})
            result = add_embeddings_to_dataframe(df, show_progress=False)
            
            assert 'embeddings' in result.columns
            assert len(result) == 2
            assert result['embeddings'].tolist() == [[0.1, 0.2], [0.3, 0.4]]

    def test_add_embeddings_custom_columns(self):
        """Test adding embeddings with custom column names."""
        with patch('news_market_analysis.embeddings.generators.generate_embeddings') as mock_gen:
            mock_gen.return_value = [[0.1, 0.2]]
            
            df = pd.DataFrame({'text_col': ['Text 1']})
            result = add_embeddings_to_dataframe(
                df,
                text_column='text_col',
                embedding_column='emb_col',
                show_progress=False
            )
            
            assert 'emb_col' in result.columns
            assert result['emb_col'].tolist() == [[0.1, 0.2]]

    def test_add_embeddings_missing_column(self):
        """Test error when text column doesn't exist."""
        df = pd.DataFrame({'other_col': ['Text 1']})
        
        with pytest.raises(EmbeddingGeneratorError) as exc_info:
            add_embeddings_to_dataframe(df, text_column='articles')
        
        assert "not found in DataFrame" in str(exc_info.value)
        assert "Available columns" in str(exc_info.value)

    def test_add_embeddings_preserves_original(self):
        """Test that original DataFrame is not modified."""
        with patch('news_market_analysis.embeddings.generators.generate_embeddings') as mock_gen:
            mock_gen.return_value = [[0.1, 0.2]]
            
            df = pd.DataFrame({'articles': ['Text 1']})
            original_cols = df.columns.tolist()
            
            result = add_embeddings_to_dataframe(df, show_progress=False)
            
            # Original should be unchanged
            assert df.columns.tolist() == original_cols
            assert 'embeddings' not in df.columns
            
            # Result should have new column
            assert 'embeddings' in result.columns

    def test_add_embeddings_custom_model(self):
        """Test adding embeddings with custom model."""
        with patch('news_market_analysis.embeddings.generators.generate_embeddings') as mock_gen:
            mock_gen.return_value = [[0.1, 0.2]]
            
            df = pd.DataFrame({'articles': ['Text 1']})
            add_embeddings_to_dataframe(
                df,
                model_name='paraphrase-MiniLM-L6-v2',
                show_progress=False
            )
            
            # Check that model_name was passed
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs['model_name'] == 'paraphrase-MiniLM-L6-v2'


class TestGetEmbeddingDimension:
    """Tests for get_embedding_dimension function."""

    def test_get_embedding_dimension_default(self):
        """Test getting embedding dimension for default model."""
        with patch('news_market_analysis.embeddings.generators.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_embedding = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
            mock_model.encode.return_value = mock_embedding
            mock_get_model.return_value = mock_model
            
            dim = get_embedding_dimension()
            
            assert dim == 5
            mock_model.encode.assert_called_once()

    def test_get_embedding_dimension_custom_model(self):
        """Test getting embedding dimension for custom model."""
        with patch('news_market_analysis.embeddings.generators.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_embedding = np.array([[0.1, 0.2, 0.3]])
            mock_model.encode.return_value = mock_embedding
            mock_get_model.return_value = mock_model
            
            dim = get_embedding_dimension('paraphrase-MiniLM-L6-v2')
            
            assert dim == 3
            mock_get_model.assert_called_once_with('paraphrase-MiniLM-L6-v2')

    def test_get_embedding_dimension_error(self):
        """Test error handling when dimension cannot be determined."""
        with patch('news_market_analysis.embeddings.generators.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.encode.side_effect = RuntimeError("Model error")
            mock_get_model.return_value = mock_model
            
            with pytest.raises(EmbeddingGeneratorError) as exc_info:
                get_embedding_dimension()
            
            assert "Failed to determine embedding dimension" in str(exc_info.value)


class TestConstants:
    """Tests for module constants."""

    def test_available_models(self):
        """Test that AVAILABLE_MODELS is defined correctly."""
        assert isinstance(AVAILABLE_MODELS, list)
        assert len(AVAILABLE_MODELS) == 3
        assert 'paraphrase-MiniLM-L6-v2' in AVAILABLE_MODELS
        assert 'paraphrase-multilingual-MiniLM-L12-v2' in AVAILABLE_MODELS
        assert 'distiluse-base-multilingual-cased-v1' in AVAILABLE_MODELS

    def test_default_model(self):
        """Test that DEFAULT_MODEL is set correctly."""
        assert DEFAULT_MODEL == 'distiluse-base-multilingual-cased-v1'
        assert DEFAULT_MODEL in AVAILABLE_MODELS
