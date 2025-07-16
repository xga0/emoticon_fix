from .emoticon_fix import (
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

__version__ = "0.2.1"

__all__ = [
    'emoticon_fix', 
    'remove_emoticons', 
    'replace_emoticons',
    'analyze_sentiment',
    'get_sentiment_score',
    'classify_sentiment',
    'extract_emotions',
    'batch_analyze',
    'SentimentAnalysis'
]
