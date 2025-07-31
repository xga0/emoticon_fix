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


class TestPerformanceOptimizations:
    """Test performance-related optimizations."""
    
    def test_large_input_performance(self):
        """Test performance with large inputs."""
        large_text = "Happy :) " * 10000  # 10k repetitions
        result = emoticon_fix(large_text)
        assert "Smile" in result
        assert result.count("Smile") == 10000
    
    def test_many_different_emoticons(self):
        """Test text with many different types of emoticons."""
        emoticons = [':)', ':D', ':(', ':P', ';)', ':*', 'XD', 'QQ', '^_^', '(^_^)']
        text = " ".join(emoticons * 100)  # 1000 emoticons total
        result = emoticon_fix(text)
        assert len(result.split()) >= 1000
    
    def test_regex_pattern_efficiency(self):
        """Test that regex patterns match correctly and efficiently."""
        # Test longest emoticons are matched first (greedy matching)
        text = ":-) :)"  # Both should be detected
        result = emoticon_fix(text)
        assert result == "Smile Smile"
        
        # Test complex emoticons
        text = "(╯°□°）╯︵ ┻━┻"
        result = emoticon_fix(text)
        assert "Table Flip" in result
    
    def test_precompiled_patterns(self):
        """Test that precompiled patterns work correctly."""
        from emoticon_fix.emoticon_fix import _COMPILED_TOKEN_PATTERN, _PUNCTUATION_SET
        
        # Test compiled pattern exists
        assert _COMPILED_TOKEN_PATTERN is not None
        
        # Test punctuation set optimization
        assert '.' in _PUNCTUATION_SET
        assert 'a' not in _PUNCTUATION_SET


class TestStressTests:
    """Stress tests with extreme inputs."""
    
    def test_extremely_long_text(self):
        """Test with extremely long text."""
        base_text = "This is a test with :) and :D emoticons. "
        long_text = base_text * 1000  # ~40k characters
        result = emoticon_fix(long_text)
        assert "Smile" in result
        assert "Laugh" in result
        assert len(result) > len(long_text)  # Should be longer due to replacements
    
    def test_dense_emoticon_text(self):
        """Test text that's mostly emoticons."""
        emoticon_text = ":):D:(:P;):*XD" * 500
        result = emoticon_fix(emoticon_text)
        assert "Smile" in result
        assert "Laugh" in result
        assert "Sad" in result
    
    def test_mixed_content_stress(self):
        """Test mixed content with URLs, hashtags, mentions, and emoticons."""
        mixed_text = (
            "Check out https://example.com :) #happy @user123 "
            "More text with :D and https://another-url.com/path?param=value "
            "@mention #hashtag :( end"
        ) * 200
        
        result = emoticon_fix(mixed_text)
        assert "https://example.com" in result  # URLs preserved
        assert "#happy" in result  # Hashtags preserved
        assert "@user123" in result  # Mentions preserved
        assert "Smile" in result
        assert "Laugh" in result
        assert "Sad" in result
    
    def test_no_emoticons_large_text(self):
        """Test large text with no emoticons."""
        no_emoticon_text = "This is regular text without any emoticons at all. " * 2000
        result = emoticon_fix(no_emoticon_text)
        # Should handle large text correctly and preserve content
        assert "This is regular text" in result
        assert "without any emoticons" in result
        assert len(result) > 1000  # Should be substantial


class TestUnicodeAndSpecialCharacters:
    """Test Unicode and special character handling."""
    
    def test_unicode_emoticons(self):
        """Test Unicode-based emoticons (kaomoji)."""
        unicode_text = "Happy (◕‿◕) and excited ＼(^o^)／ today!"
        result = emoticon_fix(unicode_text)
        assert "Happy" in result
        assert "today" in result
        # Should handle Unicode emoticons properly
    
    def test_mixed_unicode_and_ascii(self):
        """Test mixing Unicode and ASCII emoticons."""
        mixed_text = "ASCII :) and Unicode (◕‿◕) together"
        result = emoticon_fix(mixed_text)
        assert "Smile" in result
        assert "ASCII" in result
        assert "together" in result
    
    def test_special_characters_preservation(self):
        """Test that special characters are preserved correctly."""
        special_text = "Money $100 :) and €200 :D plus ¥300 :("
        result = emoticon_fix(special_text)
        # Currency symbols are tokenized separately from numbers
        assert "$" in result and "100" in result
        assert "€" in result and "200" in result
        assert "¥" in result and "300" in result
        assert "Smile" in result
        assert "Laugh" in result
        assert "Sad" in result
    
    def test_emoji_vs_emoticon(self):
        """Test that emoji don't interfere with emoticon processing."""
        emoji_text = "Happy 😀 with emoticon :) and sad 😢 with emoticon :("
        result = emoticon_fix(emoji_text)
        assert "😀" in result  # Emoji preserved
        assert "😢" in result  # Emoji preserved
        assert "Smile" in result  # Emoticon converted
        assert "Sad" in result  # Emoticon converted


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple functions."""
    
    def test_full_pipeline_analysis(self):
        """Test complete analysis pipeline."""
        text = "Great day :D with friends :) but got sad :( later, now neutral :|"
        
        # Test all functions work together
        fixed = emoticon_fix(text)
        removed = remove_emoticons(text)
        replaced = replace_emoticons(text)
        sentiment = analyze_sentiment(text)
        stats = get_emoticon_statistics(text)
        profile = create_emotion_profile(text)
        
        assert "Laugh" in fixed
        assert "Smile" in fixed
        assert len(removed) < len(text)
        assert "__EMO_" in replaced
        assert sentiment.total_count == 4
        assert stats.total_emoticons == 4
        assert profile.total_emoticons == 4
    
    def test_batch_processing_integration(self):
        """Test batch processing with multiple texts."""
        texts = [
            "Happy day :) :D",
            "Sad day :( :(", 
            "Neutral day :| :|",
            "Mixed emotions :) :( :D"
        ]
        
        # Batch analyze
        batch_results = batch_analyze(texts)
        assert len(batch_results) == 4
        
        # Trends analysis
        trends = get_emoticon_trends(texts, ['Text1', 'Text2', 'Text3', 'Text4'])
        assert trends['total_texts'] == 4
        
        # Profile comparison
        profiles = [create_emotion_profile(text, f"Profile{i}") for i, text in enumerate(texts)]
        comparison = compare_emotion_profiles(profiles)
        assert comparison['profiles_compared'] == 4
    
    def test_export_import_cycle(self):
        """Test export and verify exported data."""
        text = "Testing :) export :D functionality :("
        stats = get_emoticon_statistics(text)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Export to JSON
            json_file = os.path.join(temp_dir, "test_export.json")
            result_file = export_analysis(stats, "json", json_file)
            
            # Verify export
            assert os.path.exists(result_file)
            
            # Read and verify content
            with open(result_file, 'r') as f:
                data = json.load(f)
                assert data['total_emoticons'] == 3
                assert 'Smile' in data['emotion_frequency']


class TestBoundaryConditions:
    """Test boundary conditions and edge cases."""
    
    def test_single_character_inputs(self):
        """Test single character inputs."""
        assert emoticon_fix("a") == "a"
        assert emoticon_fix(":") == ":"
        assert emoticon_fix(")") == ")"
        assert emoticon_fix(" ") == ""
    
    def test_only_punctuation(self):
        """Test inputs with only punctuation."""
        punct_text = ".,!?;:"
        result = emoticon_fix(punct_text)
        assert len(result) > 0
    
    def test_only_emoticons(self):
        """Test inputs with only emoticons."""
        emoticon_only = ":):D:("
        result = emoticon_fix(emoticon_only)
        assert "Smile" in result
        assert "Laugh" in result
        assert "Sad" in result
    
    def test_repeated_emoticons(self):
        """Test repeated identical emoticons."""
        repeated = ":)" * 100
        result = emoticon_fix(repeated)
        assert result.count("Smile") == 100
    
    def test_whitespace_variations(self):
        """Test various whitespace scenarios."""
        test_cases = [
            "  :)  ",
            "\t:)\t",
            "\n:)\n",
            "  :)  :D  ",
            ":)\n\n:D"
        ]
        
        for text in test_cases:
            result = emoticon_fix(text)
            assert "Smile" in result
    
    def test_malformed_emoticons(self):
        """Test handling of malformed or partial emoticons."""
        malformed_cases = [
            ": )",  # Space in middle
            ":-",   # Incomplete
            "):(",  # Reversed
            "::)",  # Extra colon
            ":))",  # Extra parenthesis
        ]
        
        for text in malformed_cases:
            result = emoticon_fix(text)
            # Should not crash and should handle gracefully
            assert isinstance(result, str)


class TestErrorHandling:
    """Test comprehensive error handling."""
    
    def test_type_errors_comprehensive(self):
        """Test type errors for all functions."""
        invalid_inputs = [None, 123, [], {}, set()]
        
        for invalid_input in invalid_inputs:
            with pytest.raises((TypeError, AttributeError)):
                emoticon_fix(invalid_input)
            
            with pytest.raises((TypeError, AttributeError)):
                remove_emoticons(invalid_input)
            
            with pytest.raises((TypeError, AttributeError)):
                analyze_sentiment(invalid_input)
    
    def test_invalid_tag_format(self):
        """Test invalid tag formats for replace_emoticons."""
        with pytest.raises(ValueError):
            replace_emoticons("test :)", "invalid_format")
        
        with pytest.raises(ValueError):
            replace_emoticons("test :)", "no_placeholder_here")
    
    def test_empty_inputs_all_functions(self):
        """Test empty inputs across all functions."""
        empty_text = ""
        
        assert emoticon_fix(empty_text) == ""
        assert remove_emoticons(empty_text) == ""
        assert replace_emoticons(empty_text) == ""
        
        sentiment = analyze_sentiment(empty_text)
        assert sentiment.total_count == 0
        
        stats = get_emoticon_statistics(empty_text)
        assert stats.total_emoticons == 0


class TestMemoryEfficiency:
    """Test memory efficiency improvements."""
    
    def test_slots_usage(self):
        """Test that __slots__ is working for SentimentAnalysis."""
        sentiment = analyze_sentiment("Happy :)")
        
        # Should have __slots__ defined
        assert hasattr(sentiment.__class__, '__slots__')
        
        # Should not be able to add arbitrary attributes
        with pytest.raises(AttributeError):
            sentiment.arbitrary_attribute = "test"
    
    def test_large_data_memory_usage(self):
        """Test memory usage with large datasets."""
        # Create large dataset
        large_texts = [f"Text {i} with :) emoticon" for i in range(1000)]
        
        # Should handle large batch without memory issues
        results = batch_analyze(large_texts)
        assert len(results) == 1000
        
        # All should have detected the emoticon
        for result in results:
            assert result.total_count == 1


class TestRegexPatterns:
    """Test regex pattern functionality."""
    
    def test_pattern_matching_order(self):
        """Test that longer patterns are matched first."""
        # Test emoticons that could be subsets of others
        text = ":-) :)"
        result = emoticon_fix(text)
        assert result == "Smile Smile"
    
    def test_url_preservation(self):
        """Test that URLs are preserved correctly."""
        # Test supported URL patterns (http/https/www)
        supported_urls = [
            "https://example.com",
            "http://test.org", 
            "www.example.com"
        ]
        
        for url in supported_urls:
            text = f"Check {url} :) for info"
            result = emoticon_fix(text)
            assert url in result
            assert "Smile" in result
        
        # Test that emoticons still work with URLs
        text = "Visit https://example.com :) for more info :D"
        result = emoticon_fix(text)
        assert "https://example.com" in result
        assert "Smile" in result
        assert "Laugh" in result
    
    def test_hashtag_mention_preservation(self):
        """Test hashtags and mentions are preserved."""
        text = "Follow @user #hashtag :) and @another #tag :D"
        result = emoticon_fix(text)
        
        assert "@user" in result
        assert "#hashtag" in result
        assert "@another" in result
        assert "#tag" in result
        assert "Smile" in result
        assert "Laugh" in result
    
    def test_case_insensitive_matching(self):
        """Test case insensitive matching where applicable."""
        # Test case variations of emoticons
        test_cases = ["xd", "XD", "Xd", "xD"]
        
        for case_var in test_cases:
            result = emoticon_fix(f"Laughing {case_var}")
            assert "Laugh" in result


class TestPipelineSteps:
    """Test individual pipeline step functionality."""
    
    def test_emoticon_fix_step(self):
        """Test EmoticonFixStep."""
        from emoticon_fix import EmoticonFixStep
        
        step = EmoticonFixStep()
        assert step.name == "emoticon_fix"
        
        result = step.process("Happy :)")
        assert result == "Happy Smile"
        
        metadata = step.get_metadata("Happy :) and sad :(")
        assert metadata["emoticons_found"] == 2
        assert metadata["unique_emoticons"] == 2
    
    def test_remove_emoticon_step(self):
        """Test RemoveEmoticonStep."""
        from emoticon_fix import RemoveEmoticonStep
        
        step = RemoveEmoticonStep()
        assert step.name == "remove_emoticons"
        
        result = step.process("Happy :) World")
        assert "Happy" in result
        assert "World" in result
        assert ":)" not in result
        
        metadata = step.get_metadata("Happy :)")
        assert metadata["emoticons_removed"] == 1
    
    def test_replace_emoticon_step(self):
        """Test ReplaceEmoticonStep."""
        from emoticon_fix import ReplaceEmoticonStep
        
        step = ReplaceEmoticonStep()
        result = step.process("Happy :)")
        assert "__EMO_Smile__" in result
        
        # Test custom format
        custom_step = ReplaceEmoticonStep("<EMO:{tag}>")
        result = custom_step.process("Happy :)")
        assert "<EMO:Smile>" in result
    
    def test_sentiment_analysis_step(self):
        """Test SentimentAnalysisStep."""
        from emoticon_fix import SentimentAnalysisStep
        
        step = SentimentAnalysisStep()
        context = {}
        result = step.process("Happy :)", context)
        
        assert result == "Happy :)"  # Text unchanged
        assert "sentiment_analysis_result" in context
        
        metadata = step.get_metadata("Happy :)")
        assert metadata["sentiment_score"] > 0
        assert metadata["sentiment_classification"] in ["Positive", "Very Positive"]
    
    def test_text_cleaning_step(self):
        """Test TextCleaningStep."""
        from emoticon_fix import TextCleaningStep
        
        # Test whitespace normalization
        step = TextCleaningStep(normalize_whitespace=True)
        result = step.process("Text   with    extra   spaces")
        assert result == "Text with extra spaces"
        
        # Test punctuation cleaning
        step = TextCleaningStep(remove_extra_punctuation=True)
        result = step.process("What!!! Really???")
        assert result == "What! Really?"
        
        # Test both
        step = TextCleaningStep(normalize_whitespace=True, remove_extra_punctuation=True)
        result = step.process("Text   with!!!   extra   spaces???")
        assert result == "Text with! extra spaces?"
    
    def test_custom_step(self):
        """Test CustomStep."""
        from emoticon_fix import CustomStep
        
        def uppercase_func(text, prefix=""):
            return prefix + text.upper()
        
        step = CustomStep("uppercase", uppercase_func, prefix=">>> ")
        result = step.process("hello world")
        assert result == ">>> HELLO WORLD"
        
        metadata = step.get_metadata("test")
        assert metadata["function"] == "uppercase_func"
        assert metadata["kwargs"] == {"prefix": ">>> "}


class TestTextPreprocessingPipeline:
    """Test TextPreprocessingPipeline functionality."""
    
    def test_pipeline_creation(self):
        """Test pipeline creation and configuration."""
        from emoticon_fix import TextPreprocessingPipeline
        
        pipeline = TextPreprocessingPipeline("TestPipeline")
        assert pipeline.name == "TestPipeline"
        assert len(pipeline.steps) == 0
        assert not pipeline.enable_caching
        assert not pipeline.enable_metadata
    
    def test_add_steps_fluent_interface(self):
        """Test fluent interface for adding steps."""
        from emoticon_fix import TextPreprocessingPipeline
        
        pipeline = (TextPreprocessingPipeline("Test")
                   .add_emoticon_fix()
                   .add_sentiment_analysis()
                   .enable_cache()
                   .enable_metadata_collection())
        
        assert len(pipeline.steps) == 2
        assert pipeline.enable_caching
        assert pipeline.enable_metadata
        assert pipeline.get_step_names() == ["emoticon_fix", "sentiment_analysis"]
    
    def test_pipeline_processing(self):
        """Test basic pipeline processing."""
        from emoticon_fix import TextPreprocessingPipeline
        
        pipeline = (TextPreprocessingPipeline("Test")
                   .add_text_cleaning()
                   .add_emoticon_fix())
        
        result = pipeline.process("Text   with   :)   emoticon")
        assert "Smile" in result
        assert "   " not in result  # Whitespace normalized
    
    def test_pipeline_with_metadata(self):
        """Test pipeline processing with metadata."""
        from emoticon_fix import TextPreprocessingPipeline
        
        pipeline = (TextPreprocessingPipeline("Test")
                   .add_emoticon_fix()
                   .add_sentiment_analysis())
        
        result, metadata = pipeline.process("Happy :)", collect_metadata=True)
        
        assert result == "Happy Smile"
        assert "pipeline" in metadata
        assert "steps" in metadata
        assert len(metadata["steps"]) == 2
        assert metadata["steps"][0]["step"] == "emoticon_fix"
        assert metadata["steps"][1]["step"] == "sentiment_analysis"
    
    def test_pipeline_caching(self):
        """Test pipeline caching functionality."""
        from emoticon_fix import TextPreprocessingPipeline
        
        pipeline = (TextPreprocessingPipeline("Test")
                   .add_emoticon_fix()
                   .enable_cache())
        
        # First processing
        result1 = pipeline.process("Happy :)")
        
        # Second processing (should use cache)
        result2 = pipeline.process("Happy :)")
        
        assert result1 == result2 == "Happy Smile"
        assert len(pipeline.results_cache) == 1
        
        # Clear cache
        pipeline.clear_cache()
        assert len(pipeline.results_cache) == 0
    
    def test_batch_processing(self):
        """Test batch processing."""
        from emoticon_fix import TextPreprocessingPipeline
        
        pipeline = TextPreprocessingPipeline("Test").add_emoticon_fix()
        
        texts = ["Happy :)", "Sad :(", "Excited :D"]
        results = pipeline.process_batch(texts)
        
        assert len(results) == 3
        assert "Smile" in results[0]
        assert "Sad" in results[1]
        assert "Laugh" in results[2]
    
    def test_step_management(self):
        """Test step management functionality."""
        from emoticon_fix import TextPreprocessingPipeline, EmoticonFixStep
        
        pipeline = (TextPreprocessingPipeline("Test")
                   .add_emoticon_fix()
                   .add_sentiment_analysis())
        
        # Test step retrieval
        step = pipeline.get_step("emoticon_fix")
        assert isinstance(step, EmoticonFixStep)
        
        # Test step removal
        removed = pipeline.remove_step("sentiment_analysis")
        assert removed
        assert len(pipeline.steps) == 1
        
        # Test non-existent step removal
        removed = pipeline.remove_step("non_existent")
        assert not removed
    
    def test_pipeline_cloning(self):
        """Test pipeline cloning."""
        from emoticon_fix import TextPreprocessingPipeline
        
        original = (TextPreprocessingPipeline("Original")
                   .add_emoticon_fix()
                   .enable_cache())
        
        clone = original.clone()
        
        assert clone.name == "Original_copy"
        assert len(clone.steps) == len(original.steps)
        assert clone.enable_caching == original.enable_caching
        
        # Modifications to clone shouldn't affect original
        clone.add_sentiment_analysis()
        assert len(clone.steps) == 2
        assert len(original.steps) == 1
    
    def test_pipeline_serialization(self):
        """Test pipeline serialization to dict."""
        from emoticon_fix import TextPreprocessingPipeline
        
        pipeline = (TextPreprocessingPipeline("Test")
                   .add_emoticon_fix()
                   .add_sentiment_analysis()
                   .enable_cache())
        
        data = pipeline.to_dict()
        
        assert data["name"] == "Test"
        assert data["step_count"] == 2
        assert data["caching_enabled"]
        assert len(data["steps"]) == 2
        assert data["steps"][0]["name"] == "emoticon_fix"
    
    def test_empty_pipeline(self):
        """Test empty pipeline behavior."""
        from emoticon_fix import TextPreprocessingPipeline
        
        pipeline = TextPreprocessingPipeline("Empty")
        
        result = pipeline.process("Test text")
        assert result == "Test text"
        
        result, metadata = pipeline.process("Test", collect_metadata=True)
        assert result == "Test"
        assert metadata["steps"] == []


class TestPrebuiltPipelines:
    """Test prebuilt pipeline factories."""
    
    def test_standard_pipeline(self):
        """Test standard pipeline creation."""
        from emoticon_fix import create_standard_pipeline
        
        pipeline = create_standard_pipeline("MyStandard")
        assert pipeline.name == "MyStandard"
        
        step_names = pipeline.get_step_names()
        assert "text_cleaning" in step_names
        assert "emoticon_fix" in step_names
        assert "sentiment_analysis" in step_names
        
        # Test processing
        result = pipeline.process("Text   with  :)  extra  spaces!!!")
        assert "Smile" in result
        assert "   " not in result
        assert "!!!" not in result  # Extra punctuation removed
    
    def test_ner_pipeline(self):
        """Test NER pipeline creation."""
        from emoticon_fix import create_ner_pipeline
        
        pipeline = create_ner_pipeline()
        assert pipeline.name == "NERPipeline"
        
        step_names = pipeline.get_step_names()
        assert "text_cleaning" in step_names
        assert "replace_emoticons" in step_names
        assert "sentiment_analysis" in step_names
        
        # Test processing
        result = pipeline.process("Happy  :)  day")
        assert "__EMO_Smile__" in result
        assert "   " not in result  # Whitespace normalized
    
    def test_analysis_pipeline(self):
        """Test analysis pipeline creation."""
        from emoticon_fix import create_analysis_pipeline
        
        pipeline = create_analysis_pipeline()
        assert pipeline.name == "AnalysisPipeline"
        assert pipeline.enable_metadata
        
        result, metadata = pipeline.process("Happy :)")
        assert result == "Happy :)"  # Text unchanged
        assert "sentiment_analysis" in [step["step"] for step in metadata["steps"]]


class TestPipelineErrorHandling:
    """Test pipeline error handling."""
    
    def test_invalid_step_type(self):
        """Test adding invalid step type."""
        from emoticon_fix import TextPreprocessingPipeline
        
        pipeline = TextPreprocessingPipeline("Test")
        
        with pytest.raises(TypeError):
            pipeline.add_step("not_a_step_object")
    
    def test_invalid_input_types(self):
        """Test invalid input types for pipeline."""
        from emoticon_fix import TextPreprocessingPipeline
        
        pipeline = TextPreprocessingPipeline("Test").add_emoticon_fix()
        
        with pytest.raises(TypeError):
            pipeline.process(123)
        
        with pytest.raises(TypeError):
            pipeline.process_batch("not_a_list")
    
    def test_pipeline_step_errors(self):
        """Test error handling in pipeline steps."""
        from emoticon_fix import TextPreprocessingPipeline, CustomStep
        
        def error_func(text):
            raise ValueError("Test error")
        
        pipeline = (TextPreprocessingPipeline("Test")
                   .add_step(CustomStep("error_step", error_func)))
        
        with pytest.raises(ValueError):
            pipeline.process("Test text")


class TestPipelinePerformance:
    """Test pipeline performance characteristics."""
    
    def test_large_text_processing(self):
        """Test pipeline with large text."""
        from emoticon_fix import create_standard_pipeline
        
        pipeline = create_standard_pipeline()
        large_text = "Happy :) " * 1000  # 9000 characters
        
        result = pipeline.process(large_text)
        assert "Smile" in result
        assert result.count("Smile") == 1000
    
    def test_batch_processing_performance(self):
        """Test batch processing performance."""
        from emoticon_fix import create_standard_pipeline
        
        pipeline = create_standard_pipeline()
        texts = [f"Text {i} with :) emoticon" for i in range(100)]
        
        results = pipeline.process_batch(texts)
        assert len(results) == 100
        
        for result in results:
            assert "Smile" in result
    
    def test_caching_performance(self):
        """Test caching improves performance on repeated inputs."""
        from emoticon_fix import TextPreprocessingPipeline
        
        pipeline = (TextPreprocessingPipeline("Test")
                   .add_emoticon_fix()
                   .add_sentiment_analysis()
                   .enable_cache())
        
        text = "Happy :) and excited :D!"
        
        # Process multiple times (should use cache after first)
        for _ in range(10):
            result = pipeline.process(text)
            assert "Smile" in result and "Laugh" in result
        
        # Should have cached result
        assert len(pipeline.results_cache) == 1


if __name__ == "__main__":
    pytest.main([__file__]) 