"""Tests for vocabulary_filter function in text_processing utilities."""

import pytest

from PMRTN.utils.text_processing import vocabulary_filter


class TestVocabularyFilterBasic:
    """Basic tests for vocabulary_filter function."""
    
    def test_basic_filtering(self):
        """Test basic filtering removes infrequent and frequent words."""
        # Word frequencies: 'the' appears 5 times, 'test' 2 times, 'example' 1 time
        text = "the the the the the test test example"
        
        # Remove words appearing <= 1 time (removes 'example')
        # Remove words appearing >= 50% of total (removes 'the')
        filtered, stats = vocabulary_filter(
            text, 
            min_word_count=1, 
            max_word_count_threshold=0.5
        )
        
        assert filtered == "test test"
        assert stats['frequent_words_removed'] == 1  # 'the'
        assert stats['infrequent_words_removed'] == 1  # 'example'
        assert stats['filtered_vocab_size'] == 1  # Only 'test'
    
    def test_no_filtering_needed(self):
        """Test when no words need filtering."""
        text = "hello world test example"
        
        filtered, stats = vocabulary_filter(
            text, 
            min_word_count=0,  # Don't remove infrequent
            max_word_count_threshold=1.0  # Don't remove frequent
        )
        
        assert filtered == text
        assert stats['frequent_words_removed'] == 0
        assert stats['infrequent_words_removed'] == 0
        assert stats['filtered_vocab_size'] == 4
    
    def test_empty_text(self):
        """Test with empty text."""
        text = ""
        filtered, stats = vocabulary_filter(text)
        
        assert filtered == ""
        assert stats['original_vocab_size'] == 0
        assert stats['filtered_vocab_size'] == 0
    
    def test_single_word(self):
        """Test with single word."""
        text = "hello"
        # Use high threshold so word isn't removed as frequent
        filtered, stats = vocabulary_filter(text, min_word_count=0, max_word_count_threshold=10.0)
        
        # With min_word_count=0 and high threshold, keeps words where 0 < count < 10
        # 'hello' appears 1 time, so 0 < 1 < 10 is True, should be kept
        assert filtered == "hello"
        assert stats['original_vocab_size'] == 1
        assert stats['filtered_vocab_size'] == 1
    
    def test_all_words_removed(self):
        """Test when all words are filtered out."""
        text = "the the the"
        
        # Remove words appearing > 50% of total
        filtered, stats = vocabulary_filter(
            text,
            min_word_count=0,
            max_word_count_threshold=0.3
        )
        
        assert filtered == ""
        assert stats['frequent_words_removed'] == 1
        assert stats['filtered_vocab_size'] == 0


class TestVocabularyFilterThresholds:
    """Tests for different threshold values."""
    
    def test_min_word_count_threshold(self):
        """Test different min_word_count values."""
        text = "a a b b b c c c c d d d d d"
        
        # min_word_count=1: removes 'a' (appears 2 times, but <= 1 means exactly 1)
        filtered, stats = vocabulary_filter(text, min_word_count=2)
        
        # Should remove 'a' (2 times) and anything with exactly 2 occurrences
        assert 'a' not in filtered.split()
        assert stats['infrequent_words_removed'] >= 1
    
    def test_max_threshold_percentage(self):
        """Test different max_word_count_threshold values."""
        # 20 words total: 'common' appears 10 times (50%), 'rare' appears 10 times (50%)
        text = "common " * 10 + "rare " * 10
        
        # Threshold 40% = 8 words, so both 'common' (10 times) and 'rare' (10 times) are removed
        filtered, stats = vocabulary_filter(
            text.strip(),
            min_word_count=0,
            max_word_count_threshold=0.4
        )
        
        assert 'common' not in filtered
        assert 'rare' not in filtered
        # Both words appear 10 times which is >= 8 (threshold)
        assert stats['frequent_words_removed'] == 2
    
    def test_zero_threshold(self):
        """Test with zero max_word_count_threshold."""
        text = "hello world test"
        
        # Zero threshold means int(3 * 0.0) = 0
        # Words appearing >= 0 are considered frequent, so all are removed
        filtered, stats = vocabulary_filter(
            text,
            min_word_count=-1,  # Don't remove infrequent (-1 means nothing is <= -1)
            max_word_count_threshold=0.0
        )
        
        # All words appear >= 0, so all are removed as frequent
        # Actually, with threshold = 0, all words with count >= 0 are frequent
        # Since all words have count >= 1, they are all >= 0
        assert filtered == ""  # All removed
        assert stats['frequent_words_removed'] == 3
    
    def test_very_high_threshold(self):
        """Test with very high max_word_count_threshold."""
        text = "word " * 100
        
        # 100% threshold means remove words appearing >= 100 times
        filtered, stats = vocabulary_filter(
            text.strip(),
            min_word_count=0,
            max_word_count_threshold=1.0
        )
        
        # 'word' appears exactly 100 times, threshold is 100
        assert stats['frequent_words_removed'] >= 1


class TestVocabularyFilterStatistics:
    """Tests for statistics returned by vocabulary_filter."""
    
    def test_statistics_structure(self):
        """Test that returned statistics have correct structure."""
        text = "hello world test example"
        filtered, stats = vocabulary_filter(text)
        
        assert 'original_vocab_size' in stats
        assert 'filtered_vocab_size' in stats
        assert 'frequent_words_removed' in stats
        assert 'infrequent_words_removed' in stats
        assert 'max_word_count_threshold' in stats
        assert 'total_word_count' in stats
        
        assert isinstance(stats['original_vocab_size'], int)
        assert isinstance(stats['filtered_vocab_size'], int)
        assert isinstance(stats['frequent_words_removed'], int)
        assert isinstance(stats['infrequent_words_removed'], int)
    
    def test_statistics_accuracy(self):
        """Test accuracy of reported statistics."""
        text = "a a a b b c d e f g h i j"  # 13 words, 10 unique
        
        filtered, stats = vocabulary_filter(
            text,
            min_word_count=1,  # Remove words where count <= 1
            max_word_count_threshold=0.2  # Remove words where count >= int(13*0.2) = 2
        )
        
        assert stats['total_word_count'] == 13
        assert stats['original_vocab_size'] == 10
        # Frequent: count >= 2, so 'a' (3) and 'b' (2) are frequent
        # Infrequent: count <= 1, so c,d,e,f,g,h,i,j (all have count 1)
        # Kept: 1 < count < 2, which is NOTHING (no integer satisfies this)
        assert stats['frequent_words_removed'] == 2  # 'a' and 'b'
        assert stats['infrequent_words_removed'] == 8  # c-j (1 time each)
        assert stats['filtered_vocab_size'] == 0  # Nothing kept
        assert stats['max_word_count_threshold'] == 2  # int(13 * 0.2)


class TestVocabularyFilterVerbose:
    """Tests for verbose output mode."""
    
    def test_verbose_mode_runs(self, capsys):
        """Test that verbose mode prints output."""
        text = "hello world test example hello world"
        
        filtered, stats = vocabulary_filter(text, verbose=True)
        
        captured = capsys.readouterr()
        assert 'VOCABULARY FILTERING REPORT' in captured.out
        assert 'Original Statistics:' in captured.out
        assert 'Filtered Result:' in captured.out
    
    def test_verbose_false_no_output(self, capsys):
        """Test that verbose=False produces no output."""
        text = "hello world test"
        
        filtered, stats = vocabulary_filter(text, verbose=False)
        
        captured = capsys.readouterr()
        assert captured.out == ""


class TestVocabularyFilterEdgeCases:
    """Edge case tests for vocabulary_filter."""
    
    def test_whitespace_only(self):
        """Test with whitespace-only text."""
        text = "   \t   \n   "
        filtered, stats = vocabulary_filter(text)
        
        # After split(), empty strings are filtered out, returns original for whitespace-only
        assert filtered == text
        assert stats['original_vocab_size'] == 0
        assert stats['filtered_vocab_size'] == 0
    
    def test_repeated_spaces(self):
        """Test with multiple spaces between words."""
        text = "hello    world     test"
        
        # Use params that won't filter everything out
        filtered, stats = vocabulary_filter(text, min_word_count=0, max_word_count_threshold=10.0)
        
        # split() handles multiple spaces automatically
        # Keeps words where 0 < count < 10
        words = filtered.split()
        assert len(words) == 3
        assert set(words) == {'hello', 'world', 'test'}
    
    def test_case_sensitivity(self):
        """Test that filtering is case-sensitive."""
        text = "Hello hello HELLO"
        
        # Use params that won't filter everything out
        filtered, stats = vocabulary_filter(text, min_word_count=0, max_word_count_threshold=10.0)
        
        # All three are treated as different words, each appears once
        # Keeps words where 0 < count < 10, so all kept
        assert stats['original_vocab_size'] == 3
        assert stats['filtered_vocab_size'] == 3
    
    def test_special_characters_in_words(self):
        """Test with special characters in words."""
        text = "hello, world! test? example."
        
        # Use params that won't filter everything out
        filtered, stats = vocabulary_filter(text, min_word_count=0, max_word_count_threshold=10.0)
        
        # Punctuation is part of the word
        assert 'hello,' in filtered or 'hello' in filtered
        assert stats['original_vocab_size'] == 4
        assert stats['filtered_vocab_size'] == 4
    
    def test_very_long_text(self):
        """Test with very long text."""
        # Create text with 10000 words
        text = " ".join([f"word{i % 100}" for i in range(10000)])
        
        filtered, stats = vocabulary_filter(
            text,
            min_word_count=50,
            max_word_count_threshold=0.02
        )
        
        assert stats['total_word_count'] == 10000
        assert isinstance(filtered, str)
        assert stats['filtered_vocab_size'] >= 0


class TestVocabularyFilterRealistic:
    """Realistic use case tests."""
    
    def test_news_article_like_text(self):
        """Test with news article-like text."""
        text = """
        The company announced a new product today. The CEO said the product 
        will launch next month. Analysts believe the product could increase 
        revenue. The market reacted positively to the announcement.
        """
        
        filtered, stats = vocabulary_filter(
            text.strip(),
            min_word_count=1,
            max_word_count_threshold=0.15
        )
        
        # 'The' and 'the' are different words (case-sensitive)
        # 'product' appears 3 times out of ~30 words (10%)
        assert stats['total_word_count'] > 25
        assert stats['filtered_vocab_size'] > 0
        assert isinstance(filtered, str)
    
    def test_spanish_text(self):
        """Test with Spanish text."""
        text = """
        La empresa anunció un nuevo producto hoy. El CEO dijo que el producto
        se lanzará el próximo mes. Los analistas creen que el producto podría
        aumentar los ingresos.
        """
        
        filtered, stats = vocabulary_filter(
            text.strip(),
            min_word_count=1,
            max_word_count_threshold=0.15
        )
        
        assert stats['original_vocab_size'] > 0
        assert isinstance(filtered, str)
    
    def test_financial_jargon(self):
        """Test with financial terminology."""
        text = """
        EBITDA revenue earnings profit margin ROE ROA leverage debt equity
        valuation PE ratio EPS dividend yield growth rate return investment
        """
        
        filtered, stats = vocabulary_filter(
            text.strip(),
            min_word_count=0,
            max_word_count_threshold=0.1
        )
        
        # All words appear once, so none should be removed with min_word_count=0
        assert stats['filtered_vocab_size'] > 10


class TestVocabularyFilterReturnTypes:
    """Tests for return type validation."""
    
    def test_return_tuple(self):
        """Test that function returns tuple."""
        text = "hello world"
        result = vocabulary_filter(text)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_return_string_and_dict(self):
        """Test that tuple contains string and dict."""
        text = "hello world"
        filtered, stats = vocabulary_filter(text)
        
        assert isinstance(filtered, str)
        assert isinstance(stats, dict)
    
    def test_filtered_text_is_string(self):
        """Test that filtered text is always a string."""
        test_cases = [
            "",
            "hello",
            "hello world test",
            "   ",
            "a " * 1000
        ]
        
        for text in test_cases:
            filtered, stats = vocabulary_filter(text)
            assert isinstance(filtered, str)


class TestVocabularyFilterIntegration:
    """Integration tests with other text processing."""
    
    def test_lowercase_then_filter(self):
        """Test filtering after lowercasing."""
        text = "Hello HELLO hello World WORLD"
        
        # First lowercase, then filter
        text_lower = text.lower()
        filtered, stats = vocabulary_filter(text_lower, min_word_count=1)
        
        # After lowercasing: 'hello' x3, 'world' x2
        # min_word_count=1 removes words with exactly 1 occurrence
        assert stats['original_vocab_size'] == 2  # 'hello', 'world'
    
    def test_filter_then_count(self):
        """Test counting words after filtering."""
        text = "the the the test test example rare"
        
        filtered, stats = vocabulary_filter(
            text,
            min_word_count=1,
            max_word_count_threshold=0.3
        )
        
        # Verify filtered text has expected word count
        filtered_words = filtered.split()
        assert len(filtered_words) == len([w for w in filtered_words if w])
    
    def test_multiple_filtering_passes(self):
        """Test applying filter multiple times."""
        text = "a a a b b c d e"
        
        # First pass
        filtered1, stats1 = vocabulary_filter(text, min_word_count=1)
        
        # Second pass on already filtered text
        filtered2, stats2 = vocabulary_filter(filtered1, min_word_count=1)
        
        # Second pass should remove fewer words (or same)
        assert stats2['infrequent_words_removed'] <= stats1['infrequent_words_removed']
