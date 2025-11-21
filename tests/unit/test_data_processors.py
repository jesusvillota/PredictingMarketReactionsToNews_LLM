"""Tests for data processing utilities."""

from datetime import datetime

import pandas as pd
import pytest

from PMRTN.data.processors import (
    clean_article_text,
    convert_to_datetime,
    eliminate_text_after_word,
    extract_datetime,
    extract_tickers_from_article,
    merge_article_components,
    process_articles,
)


class TestEliminateTextAfterWord:
    """Tests for eliminate_text_after_word function."""

    def test_word_found(self):
        """Test that text after word is eliminated when word is found."""
        text = "Hello world. Goodbye world."
        result = eliminate_text_after_word(text, "Goodbye")
        assert result == "Hello world. "

    def test_word_not_found(self):
        """Test that original text is returned when word is not found."""
        text = "Hello world."
        result = eliminate_text_after_word(text, "Missing")
        assert result == "Hello world."

    def test_word_at_beginning(self):
        """Test when word is at the beginning of text."""
        text = "Start with this. Rest of text."
        result = eliminate_text_after_word(text, "Start")
        assert result == ""

    def test_empty_string(self):
        """Test with empty string input."""
        result = eliminate_text_after_word("", "word")
        assert result == ""

    def test_case_sensitive(self):
        """Test that function is case sensitive."""
        text = "Hello WORLD. goodbye world."
        result = eliminate_text_after_word(text, "goodbye")
        assert result == "Hello WORLD. "
        result = eliminate_text_after_word(text, "Goodbye")
        assert result == text  # Should not find it


class TestExtractDatetime:
    """Tests for extract_datetime function."""

    def test_valid_datetime_pattern(self):
        """Test extraction of valid datetime pattern."""
        text = "Published on 01-05-23 1430GMT by author"
        result = extract_datetime(text)
        assert result == "01-05-23 1430GMT"

    def test_no_datetime_pattern(self):
        """Test when no datetime pattern exists."""
        text = "No datetime here"
        result = extract_datetime(text)
        assert result is None

    def test_multiple_datetime_patterns(self):
        """Test that first pattern is returned when multiple exist."""
        text = "First: 01-05-23 1430GMT and second: 02-05-23 1500GMT"
        result = extract_datetime(text)
        assert result == "01-05-23 1430GMT"

    def test_invalid_format(self):
        """Test with invalid datetime format."""
        text = "Date: 2023-05-01 14:30 GMT"
        result = extract_datetime(text)
        assert result is None


class TestConvertToDatetime:
    """Tests for convert_to_datetime function."""

    def test_valid_timestamp(self):
        """Test conversion of valid timestamp."""
        # 2021-01-01 00:00:00 UTC
        timestamp_ms = 1609459200000
        result = convert_to_datetime(timestamp_ms)
        assert isinstance(result, datetime)
        assert result.year == 2021
        assert result.month == 1
        assert result.day == 1

    def test_zero_timestamp(self):
        """Test conversion of zero timestamp (EPOCH start)."""
        result = convert_to_datetime(0)
        assert isinstance(result, datetime)
        assert result.year == 1970

    def test_recent_timestamp(self):
        """Test with a recent timestamp."""
        # 2024-01-01 00:00:00 UTC
        timestamp_ms = 1704067200000
        result = convert_to_datetime(timestamp_ms)
        assert result.year == 2024


class TestExtractTickersFromArticle:
    """Tests for extract_tickers_from_article function."""

    def test_single_ticker(self):
        """Test extraction of single ticker."""
        article = "Telefónica (TEF.MC) announced results."
        result = extract_tickers_from_article(article)
        assert result == ['TEF.MC']

    def test_multiple_tickers(self):
        """Test extraction of multiple tickers."""
        article = "Telefónica (TEF.MC) and Santander (SAN.MC) rose today."
        result = extract_tickers_from_article(article)
        assert set(result) == {'TEF.MC', 'SAN.MC'}

    def test_no_tickers(self):
        """Test when no tickers are present."""
        article = "No ticker symbols in this text."
        result = extract_tickers_from_article(article)
        assert result == []

    def test_duplicate_tickers(self):
        """Test that duplicate tickers are removed."""
        article = "TEF (TEF.MC) reported that TEF.MC rose. TEF (TEF.MC) again."
        result = extract_tickers_from_article(article)
        assert result == ['TEF.MC']

    def test_non_mc_tickers_ignored(self):
        """Test that non-.MC tickers are ignored."""
        article = "Apple (AAPL) and TEF (TEF.MC) in the news."
        result = extract_tickers_from_article(article)
        assert result == ['TEF.MC']

    def test_lowercase_not_matched(self):
        """Test that lowercase tickers are not matched."""
        article = "Mention of (tef.mc) should not be captured."
        result = extract_tickers_from_article(article)
        assert result == []


class TestMergeArticleComponents:
    """Tests for merge_article_components function."""

    def test_basic_merge(self):
        """Test basic merging of title, snippet, and body."""
        df = pd.DataFrame({
            'title': ['Title 1'],
            'snippet': ['Snippet 1'],
            'body': ['Body text 1']
        })
        result = merge_article_components(df)
        assert 'articles' in result.columns
        assert result['articles'].iloc[0] == 'Title 1. Snippet 1. Body text 1'

    def test_with_nan_values(self):
        """Test merging when some components are NaN."""
        df = pd.DataFrame({
            'title': ['Title 1'],
            'snippet': [None],
            'body': ['Body text 1']
        })
        result = merge_article_components(df)
        assert result['articles'].iloc[0] == 'Title 1. . Body text 1'

    def test_custom_column_names(self):
        """Test with custom column names."""
        df = pd.DataFrame({
            'my_title': ['Title'],
            'my_snippet': ['Snippet'],
            'my_body': ['Body']
        })
        result = merge_article_components(
            df,
            title_col='my_title',
            snippet_col='my_snippet',
            body_col='my_body',
            output_col='merged'
        )
        assert 'merged' in result.columns
        assert result['merged'].iloc[0] == 'Title. Snippet. Body'

    def test_multiple_rows(self):
        """Test merging multiple rows."""
        df = pd.DataFrame({
            'title': ['Title 1', 'Title 2'],
            'snippet': ['Snippet 1', 'Snippet 2'],
            'body': ['Body 1', 'Body 2']
        })
        result = merge_article_components(df)
        assert len(result) == 2
        assert result['articles'].iloc[0] == 'Title 1. Snippet 1. Body 1'
        assert result['articles'].iloc[1] == 'Title 2. Snippet 2. Body 2'


class TestCleanArticleText:
    """Tests for clean_article_text function."""

    def test_eliminate_after_phrase(self):
        """Test that text after specific phrases is removed."""
        article = "News content here. -Escriba a author@email.com"
        result = clean_article_text(article)
        assert "-Escriba a" not in result
        assert "author@email.com" not in result
        assert "News content here." in result

    def test_remove_expressions(self):
        """Test that specific expressions are removed."""
        article = "MARKET TALK: Some news content here."
        result = clean_article_text(article)
        assert "MARKET TALK: " not in result
        assert "Some news content here." in result

    def test_remove_city_names(self):
        """Test that city names are removed."""
        article = "MADRID News about Spanish stocks."
        result = clean_article_text(article)
        assert "MADRID" not in result

    def test_remove_email_patterns(self):
        """Test that email patterns are removed."""
        article = "Content (author@example.com) more content"
        result = clean_article_text(article)
        assert "@example.com" not in result

    def test_remove_author_attributions(self):
        """Test that author attributions are removed."""
        article = "News content\nRedactada por Juan Pérez"
        result = clean_article_text(article)
        assert "Redactada por" not in result

    def test_empty_string(self):
        """Test with empty string."""
        result = clean_article_text("")
        assert result == ""

    def test_no_cleaning_needed(self):
        """Test when text doesn't need cleaning."""
        article = "Clean news content without problematic patterns."
        result = clean_article_text(article)
        # Text should be unchanged
        assert "Clean news content without problematic patterns" in result


class TestProcessArticles:
    """Tests for process_articles function."""

    def test_full_pipeline(self):
        """Test the complete article processing pipeline."""
        df = pd.DataFrame({
            'publication_datetime': [datetime(2023, 1, 1)],
            'title': ['Telefónica Results'],
            'snippet': ['Company reports (TEF.MC)'],
            'body': ['Detailed information about Telefónica (TEF.MC).'],
            'company_codes_about': ['TELEFONICA']
        })
        result = process_articles(df)
        
        assert 'publ_datetime' in result.columns
        assert 'articles' in result.columns
        assert 'tickers' in result.columns
        assert len(result) == 1
        assert result['tickers'].iloc[0] == ['TEF.MC']

    def test_filters_articles_without_tickers(self):
        """Test that articles without tickers are filtered out."""
        df = pd.DataFrame({
            'publication_datetime': [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            'title': ['With Ticker', 'Without Ticker'],
            'snippet': ['Company (TEF.MC)', 'No ticker here'],
            'body': ['Body text', 'More text'],
            'company_codes_about': ['TELEFONICA', 'COMPANY']
        })
        result = process_articles(df)
        
        assert len(result) == 1
        assert 'TEF.MC' in result['tickers'].iloc[0]

    def test_cleans_text(self):
        """Test that text cleaning is applied."""
        df = pd.DataFrame({
            'publication_datetime': [datetime(2023, 1, 1)],
            'title': ['MARKET TALK: TEF Update'],
            'snippet': ['News (TEF.MC)'],
            'body': ['Body text about Telefónica (TEF.MC).'],
            'company_codes_about': ['TELEFONICA']
        })
        result = process_articles(df)
        
        # MARKET TALK should be removed
        assert 'MARKET TALK' not in result['articles'].iloc[0]

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({
            'publication_datetime': [],
            'title': [],
            'snippet': [],
            'body': [],
            'company_codes_about': []
        })
        result = process_articles(df)
        
        assert len(result) == 0
        assert 'tickers' in result.columns
