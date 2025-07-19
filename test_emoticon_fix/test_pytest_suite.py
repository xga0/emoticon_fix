#!/usr/bin/env python3
"""
Pytest-compatible test suite for emoticon_fix functionality.
This complements the existing test files and ensures compatibility with pytest.
"""

import pytest
import os
import json
import tempfile
from emoticon_fix import (
    emoticon_fix,
    remove_emoticons,
    replace_emoticons,
    analyze_sentiment,
    get_sentiment_score,
    classify_sentiment,
    extract_emotions,
    batch_analyze,
    SentimentAnalysis,
    # New analytics functionality
    EmoticonStats,
    EmoticonProfile,
    get_emoticon_statistics,
    create_emotion_profile,
    compare_emotion_profiles,
    export_analysis,
    get_emoticon_trends
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


class TestEmoticonStats:
    """Test EmoticonStats functionality."""
    
    def test_basic_stats_creation(self):
        """Test basic EmoticonStats creation."""
        text = "Happy :) and laughing :D and happy :) again!"
        stats = EmoticonStats(text)
        
        assert stats.total_emoticons == 3
        assert stats.unique_emoticons == 2
        assert stats.dominant_emotion in ["Smile", "Laugh"]
        assert isinstance(stats.average_sentiment, float)
    
    def test_get_emoticon_statistics(self):
        """Test get_emoticon_statistics function."""
        text = "Great :D day with smiles :) and :D!"
        stats = get_emoticon_statistics(text)
        
        assert isinstance(stats, EmoticonStats)
        assert stats.total_emoticons == 3
        assert stats.unique_emoticons == 2
    
    def test_emoticon_frequency(self):
        """Test emoticon frequency counting."""
        text = "Happy :) very :) extremely :D"
        stats = EmoticonStats(text)
        
        top_emoticons = stats.get_top_emoticons(2)
        assert len(top_emoticons) == 2
        assert top_emoticons[0][1] == 2  # :) appears twice
    
    def test_emotion_frequency(self):
        """Test emotion frequency counting."""
        text = "Happy :) very :) extremely :D"
        stats = EmoticonStats(text)
        
        top_emotions = stats.get_top_emotions(2)
        assert len(top_emotions) == 2
        assert top_emotions[0][1] == 2  # Smile appears twice
    
    def test_emoticon_density(self):
        """Test emoticon density calculation."""
        text = ":)"  # 2 characters, 1 emoticon
        stats = EmoticonStats(text)
        density = stats.get_emoticon_density()
        assert density == 50.0  # 1/2 * 100 = 50
    
    def test_sentiment_distribution(self):
        """Test sentiment distribution calculation."""
        text = "Happy :) sad :( neutral :|"
        stats = EmoticonStats(text)
        
        assert stats.sentiment_distribution["positive"] >= 1
        assert stats.sentiment_distribution["negative"] >= 1
        assert stats.sentiment_distribution["neutral"] >= 1
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        text = "Happy :) and sad :("
        stats = EmoticonStats(text)
        
        data = stats.to_dict()
        assert isinstance(data, dict)
        assert "total_emoticons" in data
        assert "emoticon_frequency" in data
        assert "analysis_timestamp" in data
    
    def test_empty_text_stats(self):
        """Test stats with no emoticons."""
        text = "No emoticons here"
        stats = EmoticonStats(text)
        
        assert stats.total_emoticons == 0
        assert stats.unique_emoticons == 0
        assert stats.dominant_emotion == "None"
        assert stats.average_sentiment == 0.0


class TestEmoticonProfile:
    """Test EmoticonProfile functionality."""
    
    def test_profile_creation(self):
        """Test basic profile creation."""
        profile = EmoticonProfile("Test User")
        assert profile.name == "Test User"
        assert profile.texts_analyzed == 0
        assert profile.total_emoticons == 0
    
    def test_create_emotion_profile_single_text(self):
        """Test creating profile with single text."""
        text = "Happy :) day with :D"
        profile = create_emotion_profile(text, "Single Text Test")
        
        assert isinstance(profile, EmoticonProfile)
        assert profile.name == "Single Text Test"
        assert profile.texts_analyzed == 1
        assert profile.total_emoticons == 2
    
    def test_create_emotion_profile_multiple_texts(self):
        """Test creating profile with multiple texts."""
        texts = ["Happy :)", "Sad :(", "Excited :D :D"]
        profile = create_emotion_profile(texts, "Multi Text Test")
        
        assert profile.texts_analyzed == 3
        assert profile.total_emoticons == 4
        assert len(profile.text_stats) == 3
    
    def test_add_text_to_profile(self):
        """Test adding texts to existing profile."""
        profile = EmoticonProfile("Test")
        
        stats1 = profile.add_text("Happy :)")
        stats2 = profile.add_text("Sad :(")
        
        assert profile.texts_analyzed == 2
        assert profile.total_emoticons == 2
        assert isinstance(stats1, EmoticonStats)
        assert isinstance(stats2, EmoticonStats)
    
    def test_overall_sentiment(self):
        """Test overall sentiment calculation."""
        texts = ["Very happy :D", "Happy :)", "Sad :("]
        profile = create_emotion_profile(texts)
        
        sentiment = profile.get_overall_sentiment()
        assert isinstance(sentiment, float)
        assert sentiment > -1.0 and sentiment < 1.0
    
    def test_dominant_emotions(self):
        """Test dominant emotions extraction."""
        texts = ["Happy :)", "Happy :)", "Sad :("]
        profile = create_emotion_profile(texts)
        
        dominant = profile.get_dominant_emotions(2)
        assert len(dominant) <= 2
        assert dominant[0][0] == "Smile"  # Most frequent
        assert dominant[0][1] == 2  # Appears twice
    
    def test_emotion_diversity(self):
        """Test emotion diversity calculation."""
        # All same emotion
        texts1 = ["Happy :)", "Happy :)", "Happy :)"]
        profile1 = create_emotion_profile(texts1)
        diversity1 = profile1.get_emotion_diversity()
        
        # Different emotions
        texts2 = ["Happy :)", "Sad :(", "Excited :D"]
        profile2 = create_emotion_profile(texts2)
        diversity2 = profile2.get_emotion_diversity()
        
        assert diversity2 > diversity1  # More diverse emotions
    
    def test_sentiment_consistency(self):
        """Test sentiment consistency calculation."""
        # Consistent sentiment
        texts1 = ["Happy :)", "Happy :)", "Happy :)"]
        profile1 = create_emotion_profile(texts1)
        consistency1 = profile1.get_sentiment_consistency()
        
        # Inconsistent sentiment
        texts2 = ["Very happy :D", "Very sad :(", "Neutral :|"]
        profile2 = create_emotion_profile(texts2)
        consistency2 = profile2.get_sentiment_consistency()
        
        assert consistency2 > consistency1  # Less consistent
    
    def test_profile_to_dict(self):
        """Test profile conversion to dictionary."""
        texts = ["Happy :)", "Sad :("]
        profile = create_emotion_profile(texts, "Test Profile")
        
        data = profile.to_dict()
        assert isinstance(data, dict)
        assert data["name"] == "Test Profile"
        assert "texts_analyzed" in data
        assert "text_stats" in data


class TestCompareEmotionProfiles:
    """Test emotion profile comparison functionality."""
    
    def test_compare_multiple_profiles(self):
        """Test comparing multiple emotion profiles."""
        profile1 = create_emotion_profile(["Happy :) :)"], "Happy User")
        profile2 = create_emotion_profile(["Sad :( :("], "Sad User")
        profile3 = create_emotion_profile(["Mixed :) :("], "Mixed User")
        
        comparison = compare_emotion_profiles([profile1, profile2, profile3])
        
        assert isinstance(comparison, dict)
        assert comparison["profiles_compared"] == 3
        assert "profile_summaries" in comparison
        assert "overall_comparison" in comparison
        assert len(comparison["profile_summaries"]) == 3
    
    def test_comparison_statistics(self):
        """Test comparison statistics calculation."""
        profile1 = create_emotion_profile(["Very happy :D :D"], "Profile1")
        profile2 = create_emotion_profile(["Very sad :( :("], "Profile2")
        
        comparison = compare_emotion_profiles([profile1, profile2])
        
        overall = comparison["overall_comparison"]
        assert "sentiment_range" in overall
        assert "diversity_range" in overall
        assert "consistency_range" in overall
        
        # Should have positive and negative sentiment range
        assert overall["sentiment_range"]["max"] > 0
        assert overall["sentiment_range"]["min"] < 0
    
    def test_empty_profiles_comparison(self):
        """Test comparison with empty profiles list."""
        with pytest.raises(ValueError):
            compare_emotion_profiles([])


class TestEmoticonTrends:
    """Test emoticon trends analysis functionality."""
    
    def test_basic_trends_analysis(self):
        """Test basic trends analysis."""
        texts = ["Happy :)", "Very happy :D", "Sad :("]
        trends = get_emoticon_trends(texts)
        
        assert isinstance(trends, dict)
        assert trends["total_texts"] == 3
        assert "text_analyses" in trends
        assert "trend_summary" in trends
        assert len(trends["text_analyses"]) == 3
    
    def test_trends_with_labels(self):
        """Test trends analysis with custom labels."""
        texts = ["Happy :)", "Sad :("]
        labels = ["Text A", "Text B"]
        
        trends = get_emoticon_trends(texts, labels)
        
        analyses = trends["text_analyses"]
        assert analyses[0]["label"] == "Text A"
        assert analyses[1]["label"] == "Text B"
    
    def test_trends_sentiment_tracking(self):
        """Test sentiment trend tracking."""
        texts = ["Happy :)", "Very happy :D", "Extremely happy :D :D"]
        trends = get_emoticon_trends(texts)
        
        sentiment_trend = trends["trend_summary"]["sentiment_trend"]
        assert len(sentiment_trend) == 3
        assert all(isinstance(s, float) for s in sentiment_trend)
    
    def test_trends_emotion_frequency(self):
        """Test overall emotion frequency in trends."""
        texts = ["Happy :) :)", "Happy :)", "Sad :("]
        trends = get_emoticon_trends(texts)
        
        emotion_freq = trends["trend_summary"]["overall_emotion_frequency"]
        assert "Smile" in emotion_freq
        assert emotion_freq["Smile"] == 3  # Appears 3 times total
        assert emotion_freq["Sad"] == 1


class TestExportAnalysis:
    """Test analysis export functionality."""
    
    def test_export_emoticon_stats_json(self):
        """Test exporting EmoticonStats to JSON."""
        text = "Happy :) and sad :("
        stats = get_emoticon_statistics(text)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "test_stats.json")
            result_filename = export_analysis(stats, "json", filename)
            
            assert os.path.exists(result_filename)
            
            # Verify JSON content
            with open(result_filename, 'r') as f:
                data = json.load(f)
                assert "total_emoticons" in data
                assert data["total_emoticons"] == 2
    
    def test_export_emoticon_stats_csv(self):
        """Test exporting EmoticonStats to CSV."""
        text = "Happy :) and sad :("
        stats = get_emoticon_statistics(text)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "test_stats.csv")
            result_filename = export_analysis(stats, "csv", filename)
            
            assert os.path.exists(result_filename)
            
            # Verify CSV content
            with open(result_filename, 'r') as f:
                content = f.read()
                assert "Total Emoticons" in content
                assert "Emoticon" in content
                assert "Frequency" in content
    
    def test_export_emotion_profile_json(self):
        """Test exporting EmoticonProfile to JSON."""
        texts = ["Happy :)", "Sad :("]
        profile = create_emotion_profile(texts, "Test Profile")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "test_profile.json")
            result_filename = export_analysis(profile, "json", filename)
            
            assert os.path.exists(result_filename)
            
            # Verify JSON content
            with open(result_filename, 'r') as f:
                data = json.load(f)
                assert "name" in data
                assert data["name"] == "Test Profile"
                assert "texts_analyzed" in data
    
    def test_export_auto_filename(self):
        """Test automatic filename generation."""
        text = "Happy :)"
        stats = get_emoticon_statistics(text)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            result_filename = export_analysis(stats, "json")
            
            assert result_filename.startswith("emoticon_analysis_")
            assert result_filename.endswith(".json")
            assert os.path.exists(result_filename)
    
    def test_export_invalid_format(self):
        """Test export with invalid format."""
        text = "Happy :)"
        stats = get_emoticon_statistics(text)
        
        with pytest.raises(ValueError):
            export_analysis(stats, "invalid_format")


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
    
    def test_invalid_input_types_for_analytics(self):
        """Test analytics functions with invalid input types."""
        with pytest.raises(TypeError):
            get_emoticon_statistics(123)
        
        with pytest.raises(TypeError):
            create_emotion_profile(123)
        
        with pytest.raises(TypeError):
            get_emoticon_trends("not a list")


if __name__ == "__main__":
    pytest.main([__file__]) 