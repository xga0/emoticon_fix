#!/usr/bin/env python3
"""
Pytest-compatible test suite for emoticon_fix functionality.
This complements the existing test files and ensures compatibility with pytest.
"""

import pytest
from emoticon_fix import (
    emoticon_fix,
    remove_emoticons,
    replace_emoticons,
    analyze_sentiment,
    get_sentiment_score,
    classify_sentiment,
    extract_emotions,
    batch_analyze,
    SentimentAnalysis
)


class TestBasicFunctionality:
    """Test basic emoticon transformation functionality."""
    
    def test_basic_emoticon_transformation(self):
        """Test basic emoticon to text transformation."""
        assert emoticon_fix("Hello :)") == "Hello Smile"
        assert emoticon_fix("Sad :(") == "Sad Sad"
        assert emoticon_fix("Laugh :D") == "Laugh Laugh"
    
    def test_multiple_emoticons(self):
        """Test handling multiple emoticons."""
        assert emoticon_fix("Happy :) and laughing :D") == "Happy Smile and laughing Laugh"
    
    def test_no_emoticons(self):
        """Test text without emoticons."""
        text = "Just regular text"
        assert emoticon_fix(text) == text
    
    def test_empty_string(self):
        """Test empty string handling."""
        assert emoticon_fix("") == ""


class TestRemoveEmoticons:
    """Test emoticon removal functionality."""
    
    def test_remove_simple_emoticons(self):
        """Test removing simple emoticons."""
        result = remove_emoticons("Hello :) World")
        assert "Hello" in result
        assert "World" in result
        assert ":)" not in result
    
    def test_remove_multiple_emoticons(self):
        """Test removing multiple emoticons."""
        result = remove_emoticons("Happy :) and sad :(")
        assert "Happy" in result
        assert "and" in result
        assert "sad" in result
        assert ":)" not in result
        assert ":(" not in result


class TestReplaceEmoticons:
    """Test emoticon replacement functionality."""
    
    def test_default_replacement(self):
        """Test default emoticon replacement format."""
        result = replace_emoticons("Happy :)")
        assert "__EMO_Smile__" in result
    
    def test_custom_replacement_format(self):
        """Test custom emoticon replacement format."""
        result = replace_emoticons("Happy :)", tag_format="<EMO:{tag}>")
        assert "<EMO:Smile>" in result


class TestSentimentAnalysis:
    """Test sentiment analysis functionality."""
    
    def test_positive_sentiment(self):
        """Test positive sentiment detection."""
        score = get_sentiment_score("Happy :)")
        assert score > 0
        
        classification = classify_sentiment("Happy :)")
        assert classification in ["Positive", "Very Positive"]
    
    def test_negative_sentiment(self):
        """Test negative sentiment detection."""
        score = get_sentiment_score("Sad :(")
        assert score < 0
        
        classification = classify_sentiment("Sad :(")
        assert classification in ["Negative", "Very Negative"]
    
    def test_neutral_sentiment(self):
        """Test neutral sentiment detection."""
        score = get_sentiment_score("No emoticons here")
        assert score == 0
        
        classification = classify_sentiment("No emoticons here")
        assert classification == "Neutral"
    
    def test_sentiment_analysis_object(self):
        """Test SentimentAnalysis object."""
        analysis = analyze_sentiment("Happy :) but sad :(")
        assert isinstance(analysis, SentimentAnalysis)
        assert analysis.total_count > 0
        assert len(analysis.emoticons) == 2
        assert len(analysis.sentiments) == 2
        assert len(analysis.scores) == 2
    
    def test_extract_emotions(self):
        """Test emotion extraction."""
        emotions = extract_emotions("Happy :) and sad :(")
        assert len(emotions) == 2
        
        # Check the format: (emoticon, emotion, score)
        for emoticon, emotion, score in emotions:
            assert isinstance(emoticon, str)
            assert isinstance(emotion, str)
            assert isinstance(score, (int, float))
    
    def test_batch_analyze(self):
        """Test batch sentiment analysis."""
        texts = ["Happy :)", "Sad :(", "No emoticons"]
        results = batch_analyze(texts)
        assert len(results) == 3
        
        # All should be SentimentAnalysis objects
        for result in results:
            assert isinstance(result, SentimentAnalysis)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_none_input(self):
        """Test None input handling."""
        with pytest.raises((TypeError, AttributeError)):
            emoticon_fix(None)
    
    def test_non_string_input(self):
        """Test non-string input handling."""
        with pytest.raises((TypeError, AttributeError)):
            emoticon_fix(123)
    
    def test_unicode_emoticons(self):
        """Test Unicode emoticons (kaomoji)."""
        # This should not crash
        result = emoticon_fix("Happy (｡♥‿♥｡)")
        assert isinstance(result, str)
    
    def test_very_long_text(self):
        """Test very long text with emoticons."""
        long_text = "Happy :) " * 1000
        result = emoticon_fix(long_text)
        assert isinstance(result, str)
        assert "Smile" in result


if __name__ == "__main__":
    pytest.main([__file__]) 