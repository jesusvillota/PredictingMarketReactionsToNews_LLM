"""Text processing utilities for article cleaning and extraction."""

import re
from datetime import datetime
from typing import List


def eliminate_text_after_word(text: str, word_x: str) -> str:
    """
    Eliminate all text after a specific word.
    
    Args:
        text: Input text
        word_x: Word to search for
    
    Returns:
        Text up to (but not including) the word, or original text if word not found
    """
    index = text.find(word_x)
    if index != -1:
        return text[:index]
    else:
        return text


def extract_datetime(text: str) -> str:
    """
    Extract datetime string from text using regex pattern.
    
    Args:
        text: Input text containing datetime
    
    Returns:
        Datetime string in format 'DD-MM-YY HHMMGMT' or None if not found
    """
    datetime_pattern = r'\d{2}-\d{2}-\d{2} \d{4}GMT'
    datetime_match = re.search(datetime_pattern, text)
    if datetime_match:
        datetime_str = datetime_match.group(0)
        return datetime_str
    else:
        return None


def convert_to_datetime(timestamp_ms: int) -> datetime:
    """
    Convert milliseconds timestamp to datetime object.
    
    Args:
        timestamp_ms: Timestamp in milliseconds
    
    Returns:
        Datetime object
    """
    timestamp_seconds = timestamp_ms / 1000
    date_time = datetime.fromtimestamp(timestamp_seconds)
    return date_time


def extract_tickers_from_article(article: str) -> List[str]:
    """
    Extract ticker symbols from article text.
    
    Pattern matches '(WHATEVER.MC)' where WHATEVER is any upper-case ticker symbol.
    
    Args:
        article: Article text
    
    Returns:
        List of unique ticker symbols found
    """
    pattern = r'\(([A-Z]+\.MC)\)'
    matches = re.findall(pattern, article)
    unique_tickers = list(set(matches))
    return unique_tickers


def clean_article_text(text: str) -> str:
    """
    Clean article text by removing unwanted expressions and patterns.
    
    Args:
        text: Raw article text
    
    Returns:
        Cleaned article text
    """
    cleaned = text
    
    # Eliminate text after specific words
    eliminate_text_after_these_words = [
        "-Escriba a", "Editado por", "(END)", 
        "Versión española de", "Escriba a", "Traductores:"
    ]
    for word in eliminate_text_after_these_words:
        cleaned = eliminate_text_after_word(cleaned, word)
    
    # Remove specific expressions
    expressions_to_remove = [
        'MARKET TALK: ', 'MADRID', 'BARCELONA', 'LONDRES', 'MÉXICO', 
        'ROMA', 'BRUSELAS', 'FRÁNCFORT', 'SÍDNEY', 'PARÍS', 'RÍO DE JANEIRO',
        '(EFE Dow Jones)--', '(EFE Dow Jones).-', '(EFE Dow Jones)', 
        '(MORE TO FOLLOW)', 'Dow Jones Newswires', 'GMT', 'gmt', 'Gmt',
        '(rodrigo.demiguelroncal@dowjones.com )', 
        '(Reenfoca titular y añade detalles a lo largo del texto)', 'Rodríguez',
        '--Giulia Petroni contribuyó a esta nota', 
        '--Mauro Orrù contribuyó a esta nota', 
        '-Ben Otto contribuyó a esta nota'
    ]
    for expression in expressions_to_remove:
        cleaned = cleaned.replace(expression, '')
    
    # Remove text patterns using regex
    patterns = [
        r'\([^)]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\)',  # eliminates (name@example.domain)
        r'\(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\b\)',  # eliminates (name@example)
        r'\nRedactada por [A-Za-záéíóúÁÉÍÓÚ]+\s[A-Za-záéíóúÁÉÍÓÚ]+',  # eliminates Redactada por Name Surname
        r'Redactada por [A-Za-záéíóúÁÉÍÓÚ]+\s[A-Za-záéíóúÁÉÍÓÚ]\sy\s[A-Za-záéíóúÁÉÍÓÚ]+\s[A-Za-záéíóúÁÉÍÓÚ]',  # eliminates Redactada por Name Surname y Name Surname
        r'\([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}; @\w+\)',  # eliminates (email; @twitter)
        r'Por\s[A-Za-záéíóúÁÉÍÓÚ]+\s[A-Za-záéíóúÁÉÍÓÚ]+\s[A-Za-záéíóúÁÉÍÓÚ]+'  # eliminates Por Name1 Name2 Name3
    ]
    
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned)
    
    return cleaned

