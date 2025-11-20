"""Text processing utilities for news articles and financial text.

This module provides additional text processing utilities beyond those
in data/processors.py, focused on normalization, formatting, and analysis.
"""

import re
from typing import List, Optional


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text (collapse multiple spaces, trim).

    Args:
        text: Input text string.

    Returns:
        Text with normalized whitespace.

    Example:
        >>> normalize_whitespace("Hello   world  \\n  test")
        'Hello world test'
    """
    # Replace multiple whitespace (including newlines) with single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def remove_urls(text: str) -> str:
    """Remove URLs from text.

    Args:
        text: Input text string.

    Returns:
        Text with URLs removed.

    Example:
        >>> remove_urls("Check https://example.com for more")
        'Check  for more'
    """
    # Pattern matches http/https URLs
    url_pattern = r'https?://\S+'
    text = re.sub(url_pattern, '', text)
    return text


def remove_email_addresses(text: str) -> str:
    """Remove email addresses from text.

    Args:
        text: Input text string.

    Returns:
        Text with email addresses removed.

    Example:
        >>> remove_email_addresses("Contact me at user@example.com")
        'Contact me at '
    """
    # Pattern matches email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    text = re.sub(email_pattern, '', text)
    return text


def truncate_text(
    text: str, max_length: int = 512, suffix: str = '...'
) -> str:
    """Truncate text to maximum length with optional suffix.

    Args:
        text: Input text string.
        max_length: Maximum length (including suffix).
        suffix: Suffix to add when truncating (default '...').

    Returns:
        Truncated text.

    Example:
        >>> truncate_text("This is a long text", 10)
        'This is...'
    """
    if len(text) <= max_length:
        return text

    # Account for suffix length
    truncate_at = max_length - len(suffix)
    return text[:truncate_at] + suffix


def extract_sentences(text: str) -> List[str]:
    """Extract sentences from text.

    Uses simple sentence boundary detection based on punctuation
    followed by whitespace and capital letter.

    Args:
        text: Input text string.

    Returns:
        List of sentences.

    Example:
        >>> extract_sentences("Hello world. How are you? Fine!")
        ['Hello world.', 'How are you?', 'Fine!']
    """
    # Simple sentence splitting on .!? followed by space and capital
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_pattern, text)
    # Clean up and remove empty strings
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def count_words(text: str) -> int:
    """Count words in text.

    Args:
        text: Input text string.

    Returns:
        Number of words.

    Example:
        >>> count_words("Hello world, this is a test")
        6
    """
    # Split on whitespace and count non-empty tokens
    words = text.split()
    return len(words)


def extract_numbers(text: str) -> List[float]:
    """Extract numeric values from text.

    Args:
        text: Input text string.

    Returns:
        List of numeric values found in text.

    Example:
        >>> extract_numbers("Growth of 3.5% and revenue of 1000")
        [3.5, 1000.0]
    """
    # Pattern matches integers and floats (including negative)
    number_pattern = r'-?\d+\.?\d*'
    matches = re.findall(number_pattern, text)
    numbers = [float(match) for match in matches if match]
    return numbers


def contains_keywords(
    text: str, keywords: List[str], case_sensitive: bool = False
) -> bool:
    """Check if text contains any of the given keywords.

    Args:
        text: Input text string.
        keywords: List of keywords to search for.
        case_sensitive: Whether to perform case-sensitive search (default False).

    Returns:
        True if any keyword is found, False otherwise.

    Example:
        >>> contains_keywords("Apple stock rises", ["apple", "stock"])
        True
    """
    if not case_sensitive:
        text = text.lower()
        keywords = [k.lower() for k in keywords]

    return any(keyword in text for keyword in keywords)


def remove_special_characters(
    text: str, keep_spaces: bool = True, keep_punctuation: bool = False
) -> str:
    """Remove special characters from text.

    Args:
        text: Input text string.
        keep_spaces: Whether to keep spaces (default True).
        keep_punctuation: Whether to keep basic punctuation .,!? (default False).

    Returns:
        Text with special characters removed.

    Example:
        >>> remove_special_characters("Hello @world! #test", keep_punctuation=True)
        'Hello world! test'
    """
    if keep_punctuation:
        pattern = r'[^a-zA-Z0-9áéíóúñÁÉÍÓÚÑ.,!?\s]' if keep_spaces else r'[^a-zA-Z0-9áéíóúñÁÉÍÓÚÑ.,!?]'
    else:
        pattern = r'[^a-zA-Z0-9áéíóúñÁÉÍÓÚÑ\s]' if keep_spaces else r'[^a-zA-Z0-9áéíóúñÁÉÍÓÚÑ]'

    text = re.sub(pattern, '', text)
    return text


def capitalize_first_letter(text: str) -> str:
    """Capitalize first letter of text.

    Args:
        text: Input text string.

    Returns:
        Text with first letter capitalized.

    Example:
        >>> capitalize_first_letter("hello world")
        'Hello world'
    """
    if not text:
        return text
    return text[0].upper() + text[1:] if len(text) > 1 else text.upper()


def remove_repeated_punctuation(text: str) -> str:
    """Remove repeated punctuation marks.

    Args:
        text: Input text string.

    Returns:
        Text with repeated punctuation collapsed to single occurrence.

    Example:
        >>> remove_repeated_punctuation("Hello!!! World???")
        'Hello! World?'
    """
    # Replace 2+ repeated punctuation with single
    text = re.sub(r'([!?.,])\1+', r'\1', text)
    return text


def extract_quoted_text(text: str) -> List[str]:
    """Extract quoted text segments from text.

    Args:
        text: Input text string.

    Returns:
        List of quoted text segments (without quotes).

    Example:
        >>> extract_quoted_text('He said "hello" and she replied "goodbye"')
        ['hello', 'goodbye']
    """
    # Match text within double quotes
    quoted_pattern = r'"([^"]*)"'
    matches = re.findall(quoted_pattern, text)
    return matches


def calculate_text_statistics(text: str) -> dict:
    """Calculate various statistics about text.

    Args:
        text: Input text string.

    Returns:
        Dictionary containing:
            - char_count: Total character count
            - word_count: Total word count
            - sentence_count: Estimated sentence count
            - avg_word_length: Average word length
            - avg_sentence_length: Average sentence length (in words)

    Example:
        >>> stats = calculate_text_statistics("Hello world. This is a test.")
        >>> stats['word_count']
        6
    """
    char_count = len(text)
    word_count = count_words(text)
    sentences = extract_sentences(text)
    sentence_count = len(sentences)

    words = text.split()
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

    return {
        'char_count': char_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': avg_word_length,
        'avg_sentence_length': avg_sentence_length,
    }
