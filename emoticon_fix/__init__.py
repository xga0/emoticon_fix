from .emoticon_fix import (
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

__version__ = "0.2.4"

__all__ = [
    'emoticon_fix', 
    'remove_emoticons', 
    'replace_emoticons',
    'analyze_sentiment',
    'get_sentiment_score',
    'classify_sentiment',
    'extract_emotions',
    'batch_analyze',
    'SentimentAnalysis',
    # New analytics functionality
    'EmoticonStats',
    'EmoticonProfile',
    'get_emoticon_statistics',
    'create_emotion_profile',
    'compare_emotion_profiles',
    'export_analysis',
    'get_emoticon_trends'
]
