"""Tests for text processing utilities module."""

import pytest

from news_market_analysis.utils.text_processing import (
    calculate_text_statistics,
    capitalize_first_letter,
    contains_keywords,
    count_words,
    extract_numbers,
    extract_quoted_text,
    extract_sentences,
    normalize_whitespace,
    remove_email_addresses,
    remove_repeated_punctuation,
    remove_special_characters,
    remove_urls,
    truncate_text,
)


class TestNormalizeWhitespace:
    """Tests for whitespace normalization."""

    def test_normalize_multiple_spaces(self):
        """Test collapsing multiple spaces."""
        text = "Hello   world  test"
        result = normalize_whitespace(text)
        assert result == "Hello world test"

    def test_normalize_newlines(self):
        """Test removing newlines."""
        text = "Hello\nworld\n\ntest"
        result = normalize_whitespace(text)
        assert result == "Hello world test"

    def test_normalize_mixed_whitespace(self):
        """Test normalizing mixed whitespace."""
        text = "  Hello \t world \n test  "
        result = normalize_whitespace(text)
        assert result == "Hello world test"

    def test_normalize_empty_string(self):
        """Test with empty string."""
        result = normalize_whitespace("")
        assert result == ""


class TestRemoveUrls:
    """Tests for URL removal."""

    def test_remove_http_url(self):
        """Test removing http URLs."""
        text = "Check http://example.com for more"
        result = remove_urls(text)
        assert "http://example.com" not in result
        assert "Check" in result and "for more" in result

    def test_remove_https_url(self):
        """Test removing https URLs."""
        text = "Visit https://secure.example.com/path?query=1"
        result = remove_urls(text)
        assert "https://" not in result

    def test_remove_multiple_urls(self):
        """Test removing multiple URLs."""
        text = "Visit http://example.com and https://another.com"
        result = remove_urls(text)
        assert "http://" not in result
        assert "https://" not in result

    def test_remove_urls_no_urls(self):
        """Test with text containing no URLs."""
        text = "Just plain text"
        result = remove_urls(text)
        assert result == text


class TestRemoveEmailAddresses:
    """Tests for email address removal."""

    def test_remove_basic_email(self):
        """Test removing basic email address."""
        text = "Contact me at user@example.com"
        result = remove_email_addresses(text)
        assert "user@example.com" not in result
        assert "Contact me at" in result

    def test_remove_complex_email(self):
        """Test removing complex email address."""
        text = "Email: john.doe+tag@sub.example.co.uk"
        result = remove_email_addresses(text)
        assert "@" not in result or result == "Email: "

    def test_remove_multiple_emails(self):
        """Test removing multiple email addresses."""
        text = "Contact alice@example.com or bob@test.org"
        result = remove_email_addresses(text)
        assert "alice@example.com" not in result
        assert "bob@test.org" not in result

    def test_remove_emails_no_emails(self):
        """Test with text containing no emails."""
        text = "Just plain text with @ symbol"
        result = remove_email_addresses(text)
        assert result == text


class TestTruncateText:
    """Tests for text truncation."""

    def test_truncate_long_text(self):
        """Test truncating long text."""
        text = "This is a very long text that needs truncation"
        result = truncate_text(text, max_length=20)
        assert len(result) == 20
        assert result.endswith("...")

    def test_truncate_short_text(self):
        """Test truncating text shorter than max_length."""
        text = "Short"
        result = truncate_text(text, max_length=20)
        assert result == text

    def test_truncate_custom_suffix(self):
        """Test truncation with custom suffix."""
        text = "This is a long text"
        result = truncate_text(text, max_length=15, suffix=">>")
        assert len(result) == 15
        assert result.endswith(">>")

    def test_truncate_exact_length(self):
        """Test truncating text at exact max_length."""
        text = "Exactly20characters!"
        result = truncate_text(text, max_length=20)
        assert result == text


class TestExtractSentences:
    """Tests for sentence extraction."""

    def test_extract_basic_sentences(self):
        """Test extracting basic sentences."""
        text = "Hello world. How are you? Fine!"
        sentences = extract_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "Hello world."
        assert sentences[1] == "How are you?"
        assert sentences[2] == "Fine!"

    def test_extract_single_sentence(self):
        """Test with single sentence."""
        text = "Just one sentence."
        sentences = extract_sentences(text)
        assert len(sentences) == 1
        assert sentences[0] == "Just one sentence."

    def test_extract_no_sentences(self):
        """Test with text without proper sentence boundaries."""
        text = "no capital letters here"
        sentences = extract_sentences(text)
        # Should return the whole text as one sentence
        assert len(sentences) >= 1

    def test_extract_sentences_with_whitespace(self):
        """Test sentence extraction with extra whitespace."""
        text = "First sentence.  Second sentence!   Third sentence?"
        sentences = extract_sentences(text)
        assert len(sentences) == 3


class TestCountWords:
    """Tests for word counting."""

    def test_count_basic_words(self):
        """Test counting basic words."""
        text = "Hello world this is a test"
        count = count_words(text)
        assert count == 6

    def test_count_with_punctuation(self):
        """Test counting words with punctuation."""
        text = "Hello, world! This is a test."
        count = count_words(text)
        assert count == 6

    def test_count_empty_string(self):
        """Test counting words in empty string."""
        count = count_words("")
        assert count == 0

    def test_count_whitespace_only(self):
        """Test counting words with only whitespace."""
        count = count_words("   ")
        assert count == 0


class TestExtractNumbers:
    """Tests for number extraction."""

    def test_extract_integers(self):
        """Test extracting integers."""
        text = "The count is 100 and 200"
        numbers = extract_numbers(text)
        assert 100.0 in numbers
        assert 200.0 in numbers

    def test_extract_floats(self):
        """Test extracting floating point numbers."""
        text = "Growth of 3.5% and revenue of 1000.75"
        numbers = extract_numbers(text)
        assert 3.5 in numbers
        assert 1000.75 in numbers

    def test_extract_negative_numbers(self):
        """Test extracting negative numbers."""
        text = "Loss of -50 and -2.5"
        numbers = extract_numbers(text)
        assert -50.0 in numbers
        assert -2.5 in numbers

    def test_extract_no_numbers(self):
        """Test with text containing no numbers."""
        text = "No numbers here"
        numbers = extract_numbers(text)
        assert len(numbers) == 0


class TestContainsKeywords:
    """Tests for keyword detection."""

    def test_contains_single_keyword(self):
        """Test detecting single keyword."""
        text = "Apple stock rises today"
        assert contains_keywords(text, ["apple"])
        assert contains_keywords(text, ["stock"])

    def test_contains_multiple_keywords(self):
        """Test detecting multiple keywords."""
        text = "Apple and Google stocks"
        assert contains_keywords(text, ["apple", "google"])

    def test_contains_no_keywords(self):
        """Test with no matching keywords."""
        text = "Microsoft announces product"
        assert not contains_keywords(text, ["apple", "google"])

    def test_contains_case_sensitive(self):
        """Test case-sensitive keyword matching."""
        text = "Apple stock rises"
        assert contains_keywords(text, ["Apple"], case_sensitive=True)
        assert not contains_keywords(text, ["apple"], case_sensitive=True)

    def test_contains_case_insensitive(self):
        """Test case-insensitive keyword matching."""
        text = "Apple stock rises"
        assert contains_keywords(text, ["apple"], case_sensitive=False)
        assert contains_keywords(text, ["APPLE"], case_sensitive=False)


class TestRemoveSpecialCharacters:
    """Tests for special character removal."""

    def test_remove_basic_special_chars(self):
        """Test removing basic special characters."""
        text = "Hello @world! #test"
        result = remove_special_characters(text)
        assert "@" not in result
        assert "#" not in result
        assert "!" not in result

    def test_remove_keep_punctuation(self):
        """Test removing special chars but keeping punctuation."""
        text = "Hello @world! #test."
        result = remove_special_characters(text, keep_punctuation=True)
        assert "@" not in result
        assert "#" not in result
        assert "!" in result
        assert "." in result

    def test_remove_keep_spaces(self):
        """Test keeping spaces."""
        text = "Hello @world #test"
        result = remove_special_characters(text, keep_spaces=True)
        assert " " in result

    def test_remove_spanish_chars(self):
        """Test preserving Spanish characters."""
        text = "España! Niño. Año."
        result = remove_special_characters(text, keep_punctuation=True)
        assert "ñ" in result
        assert "!" in result


class TestCapitalizeFirstLetter:
    """Tests for first letter capitalization."""

    def test_capitalize_lowercase(self):
        """Test capitalizing lowercase first letter."""
        text = "hello world"
        result = capitalize_first_letter(text)
        assert result == "Hello world"

    def test_capitalize_already_capitalized(self):
        """Test with already capitalized text."""
        text = "Hello world"
        result = capitalize_first_letter(text)
        assert result == "Hello world"

    def test_capitalize_empty_string(self):
        """Test with empty string."""
        result = capitalize_first_letter("")
        assert result == ""

    def test_capitalize_single_char(self):
        """Test with single character."""
        result = capitalize_first_letter("h")
        assert result == "H"


class TestRemoveRepeatedPunctuation:
    """Tests for repeated punctuation removal."""

    def test_remove_repeated_exclamation(self):
        """Test removing repeated exclamation marks."""
        text = "Hello!!! World"
        result = remove_repeated_punctuation(text)
        assert result == "Hello! World"

    def test_remove_repeated_question(self):
        """Test removing repeated question marks."""
        text = "What??? How???"
        result = remove_repeated_punctuation(text)
        assert result == "What? How?"

    def test_remove_repeated_periods(self):
        """Test removing repeated periods."""
        text = "End... Start"
        result = remove_repeated_punctuation(text)
        assert result == "End. Start"

    def test_remove_mixed_repeated(self):
        """Test removing various repeated punctuation."""
        text = "Wow!!! Really??? Yes..."
        result = remove_repeated_punctuation(text)
        assert result == "Wow! Really? Yes."


class TestExtractQuotedText:
    """Tests for quoted text extraction."""

    def test_extract_single_quote(self):
        """Test extracting single quoted text."""
        text = 'He said "hello"'
        quoted = extract_quoted_text(text)
        assert len(quoted) == 1
        assert quoted[0] == "hello"

    def test_extract_multiple_quotes(self):
        """Test extracting multiple quoted texts."""
        text = 'He said "hello" and she replied "goodbye"'
        quoted = extract_quoted_text(text)
        assert len(quoted) == 2
        assert "hello" in quoted
        assert "goodbye" in quoted

    def test_extract_no_quotes(self):
        """Test with no quoted text."""
        text = "No quotes here"
        quoted = extract_quoted_text(text)
        assert len(quoted) == 0

    def test_extract_empty_quotes(self):
        """Test extracting empty quotes."""
        text = 'Empty: ""'
        quoted = extract_quoted_text(text)
        assert len(quoted) == 1
        assert quoted[0] == ""


class TestCalculateTextStatistics:
    """Tests for text statistics calculation."""

    def test_calculate_basic_statistics(self):
        """Test calculating basic text statistics."""
        text = "Hello world. This is a test."
        stats = calculate_text_statistics(text)

        assert stats['char_count'] == len(text)
        assert stats['word_count'] == 6
        assert stats['sentence_count'] == 2
        assert stats['avg_word_length'] > 0
        assert stats['avg_sentence_length'] > 0

    def test_calculate_single_sentence(self):
        """Test statistics for single sentence."""
        text = "Single sentence here."
        stats = calculate_text_statistics(text)

        assert stats['sentence_count'] == 1
        assert stats['word_count'] == 3
        assert stats['avg_sentence_length'] == 3

    def test_calculate_empty_text(self):
        """Test statistics for empty text."""
        text = ""
        stats = calculate_text_statistics(text)

        assert stats['char_count'] == 0
        assert stats['word_count'] == 0
        assert stats['avg_word_length'] == 0

    def test_calculate_statistics_structure(self):
        """Test that all expected keys are present."""
        text = "Test text."
        stats = calculate_text_statistics(text)

        assert 'char_count' in stats
        assert 'word_count' in stats
        assert 'sentence_count' in stats
        assert 'avg_word_length' in stats
        assert 'avg_sentence_length' in stats


class TestIntegration:
    """Integration tests for text processing utilities."""

    def test_full_text_cleaning_pipeline(self):
        """Test complete text cleaning pipeline."""
        text = """
        Check http://example.com!!!   
        Contact: user@example.com
        "This is a quote"
        Growth of 3.5% @mention #hashtag
        """

        # Clean the text
        cleaned = remove_urls(text)
        cleaned = remove_email_addresses(cleaned)
        cleaned = remove_special_characters(cleaned, keep_punctuation=True)
        cleaned = normalize_whitespace(cleaned)
        cleaned = remove_repeated_punctuation(cleaned)

        # Verify cleaning
        assert "http://" not in cleaned
        assert "@" not in cleaned or "mention" not in cleaned
        assert "#" not in cleaned

    def test_text_analysis_workflow(self):
        """Test text analysis workflow."""
        text = "Apple stock rises 15.5%. Google announces new product. Microsoft revenue reaches $1000."

        # Extract information
        numbers = extract_numbers(text)
        sentences = extract_sentences(text)
        stats = calculate_text_statistics(text)

        # Verify analysis
        assert len(numbers) > 0
        assert len(sentences) == 3
        assert stats['word_count'] > 0
        assert stats['sentence_count'] == 3

    def test_keyword_detection_workflow(self):
        """Test keyword detection workflow."""
        text = "Apple and Google stocks rise while Microsoft falls"

        tech_keywords = ["apple", "google", "microsoft"]
        assert contains_keywords(text, tech_keywords)

        finance_keywords = ["stock", "rise", "fall"]
        assert contains_keywords(text, finance_keywords)

        unrelated_keywords = ["banana", "orange"]
        assert not contains_keywords(text, unrelated_keywords)
