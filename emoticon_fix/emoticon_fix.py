#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A Python package to transform emoticons in text to their corresponding meanings.
This is useful for NLP preprocessing where emoticons need to be preserved as meaningful text.
"""

import re
import json
import csv
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter, defaultdict
from datetime import datetime

# Emoticon dictionary mapping emoticons to their meanings
EMOTICON_DICT: Dict[str, str] = {
    ':)': 'Smile', ':-)': 'Smile', ':-]': 'Smile', ':]': 'Smile', ':-3': 'Smile', ':3': 'Smile',
    ':->': 'Smile', ':>': 'Smile', '8-)': 'Smile', '8)': 'Smile', ':-}': 'Smile', ':}': 'Smile',
    ':o)': 'Smile', ':c)': 'Smile', ':^)': 'Smile', '=]': 'Smile', '=)': 'Smile',
    ':-D': 'Laugh', ':D': 'Laugh', '8-D': 'Laugh', '8D': 'Laugh', 'x-D': 'Laugh', 'xD': 'Laugh',
    'X-D': 'Laugh', 'XD': 'Laugh', '=D': 'Laugh', '=3': 'Laugh', 'B^D': 'Laugh', ':-))': 'Very Happy',
    ':-(': 'Sad', ':(': 'Sad', ':-c': 'Sad', ':c': 'Sad', ':-<': 'Sad', ':<': 'Sad',
    ':-[': 'Sad', ':[': 'Sad', ':-||': 'Angry', '>:[': 'Angry', ':{': 'Angry', ':@': 'Angry', '>:(': 'Angry',
    "('_')": 'Crying', '(/_;)': 'Crying', '(T_T)': 'Crying', '(;_;)': 'Crying', '(;_;': 'Crying',
    '(;_:)': 'Crying', '(;O;)': 'Crying', '(:_;)': 'Crying', '(ToT)': 'Crying', '(Ｔ▽Ｔ)': 'Crying',
    ';_;': 'Crying', ';-;': 'Crying', ';n;': 'Crying', ';;': 'Crying', 'Q.Q': 'Crying', 'T.T': 'Crying',
    'TnT': 'Crying', 'QQ': 'Crying', 'Q_Q': 'Crying', ":'-(": 'Crying', ":'(": 'Crying',
    '(ー_ー)!!': 'Annoyed', '(-.-)': 'Annoyed', '(-_-)': 'Annoyed', '(一一)': 'Annoyed', '(；一_一)': 'Annoyed',
    '"(-""-)"': 'Annoyed', '(ーー゛)': 'Annoyed', '(^_^メ)': 'Annoyed', '(-_-メ)': 'Annoyed', '(~_~メ)': 'Annoyed',
    '(－－〆)': 'Annoyed', '(・へ・)': 'Annoyed', '(｀´)': 'Annoyed', '<`～´>': 'Annoyed', '<`ヘ´>': 'Annoyed',
    '(ーー;)': 'Annoyed', '(＃￣ω￣)': 'Dissatisfied', '(；一_一)': 'Unimpressed',
    '＼(~o~)／': 'Excited', '＼(^o^)／': 'Excited', '＼(-o-)／': 'Excited', 'ヽ(^。^)ノ': 'Excited',
    'ヽ(^o^)丿': 'Excited', '(*^0^*)': 'Excited', '(╯✧▽✧)╯': 'Excited Jump', 'o(≧▽≦)o': 'Very Happy',
    ":'-)": 'Tears of Joy', ":')": 'Tears of Joy', "D-':": 'Horror', 'D:<': 'Horror', 'D:': 'Horror',
    'D8': 'Horror', 'D;': 'Horror', 'D=': 'Horror', 'DX': 'Horror',
    ':-O': 'Surprised', ':O': 'Surprised', ':-o': 'Surprised', ':o': 'Surprised', ':-0': 'Surprised',
    '8-0': 'Surprised', '>:O': 'Surprised', '(￣□￣;)': 'Surprised', '°o°': 'Surprised', '°O°': 'Surprised',
    'o_0': 'Surprised', 'o.O': 'Surprised', '(o.o)': 'Surprised', 'oO': 'Surprised', '(⊙ω⊙)': 'Wide Eyed',
    '(╬☉д⊙)⊰⊹ฺ': 'Shocked', '(∩╹□╹∩)': 'Surprised Joy', '(°°)': 'Shocked', '(°-°)': 'Shocked',
    '(°.°)': 'Shocked', '(°_°)': 'Shocked', '(°_°>)': 'Shocked', '(°レ°)': 'Shocked', '(°o°)': 'Shocked',
    ':-*': 'Kiss', ':*': 'Kiss', ':×': 'Kiss', ';-)': 'Wink', ';)': 'Wink', '*-)': 'Wink', '*)': 'Wink',
    ';-]': 'Wink', ';]': 'Wink', ';^)': 'Wink', ':-,': 'Wink', ';D': 'Wink',
    '(｡•̀ᴗ-)': 'Gentle Wink', '(｡•̀ᴗ-)✧': 'Wink Star', '(｡•̀‿-)✧': 'Playful Wink',
    ':-P': 'Tongue', ':P': 'Tongue', 'X-P': 'Tongue', 'XP': 'Tongue', 'x-p': 'Tongue', 'xp': 'Tongue',
    ':-p': 'Tongue', ':p': 'Tongue', ':-Þ': 'Tongue', ':Þ': 'Tongue', ':-þ': 'Tongue', ':þ': 'Tongue',
    ':-b': 'Tongue', ':b': 'Tongue', 'd:': 'Tongue', '=p': 'Tongue', '>:P': 'Tongue',
    ':-/': 'Skeptical', ':/': 'Skeptical', ':-.': 'Skeptical', '>:\\': 'Skeptical', '>:/': 'Skeptical',
    ':\\': 'Skeptical', '=/': 'Skeptical', '=\\': 'Skeptical', ':L': 'Skeptical', '=L': 'Skeptical',
    ':S': 'Skeptical', '(｡ì _ í｡)': 'Skeptical',
    ':-|': 'Neutral', ':|': 'Neutral', ':$': 'Embarrassed', '://)': 'Embarrassed', '://3': 'Embarrassed',
    '(^^ゞ': 'Embarrassed', '(^_^;)': 'Embarrassed', '(-_-;)': 'Embarrassed', '(~_~;)': 'Embarrassed',
    '(・.・;)': 'Embarrassed', '(・_・;)': 'Embarrassed', '(・・;)^^': 'Embarrassed', ';^_^;': 'Embarrassed',
    '(#^.^#)': 'Embarrassed', '(^^;)': 'Embarrassed', '(⁄ ⁄•⁄ω⁄•⁄ ⁄)': 'Embarrassed Blush', '(｡ﾉω＼｡)': 'Shy Embarrassed',
    ':-X': 'Sealed Lips', ':X': 'Sealed Lips', ':-#': 'Sealed Lips', ':#': 'Sealed Lips',
    ':-&': 'Sealed Lips', ':&': 'Sealed Lips',
    'O:-)': 'Angel', 'O:)': 'Angel', '0:-3': 'Angel', '0:3': 'Angel', '0:-)': 'Angel', '0:)': 'Angel', '0;^)': 'Angel',
    '>:-)': 'Devil', '>:)': 'Devil', '}:-)': 'Devil', '}:)': 'Devil', '3:-)': 'Devil', '3:)': 'Devil',
    '>;)': 'Devil', '>:3': 'Devil', '>;3': 'Devil', '(｀∀´)Ψ': 'Mischievous',
    ':‑J': 'Tongue in Cheek', '#‑)': 'Partied All Night', '|;-)': 'Cool', '|-O': 'Bored',
    '%-)': 'Confused', '%)': 'Confused', '⊙﹏⊙': 'Confused', '(；◔ิз◔ิ)': 'Confused Kiss',
    ':-###..': 'Being Sick', ':###.': 'Being Sick', '<:-|': 'Dump',
    '(>_<)': 'Troubled', '(>_<)>': 'Troubled', '((+_+))': 'Troubled', '(+o+)': 'Troubled',
    '^_^': 'Happy', '(^_^)/': 'Wave', '(^O^)／': 'Wave', '(^o^)／': 'Wave', '(^^)/': 'Wave',
    '(-∇-)/': 'Wave', '(/-ヮ-)/': 'Wave', '(^o^)丿': 'Wave', '(ノ•̀ o •́ )ノ': 'Excited Wave',
    '∩(·ω·)∩': 'Wagging', '(·ω·)': 'Wagging', '^ω^': 'Wagging',
    '(✿◠‿◠)': 'Happy', '(◠‿◠✿)': 'Flower Smile',
    '(づ｡◕‿‿◕｡)づ': 'Hug', '(ﾉ◕ヮ◕)ﾉ*:･ﾟ✧': 'Throwing Sparkles',
    '⊂((・▽・))⊃': 'Big Hug', '(⊃｡•́‿•̀｡)⊃': 'Want Hug', '(･ω･)つ⊂(･ω･)': 'Hugging',
    '(╯°□°）╯︵ ┻━┻': 'Table Flip', '┬─┬ノ( º _ ºノ)': 'Put Table Back',
    '(¬_¬)': 'Disapproval', '(；⌣̀_⌣́)': 'Nervous', '(╥﹏╥)': 'Sobbing', '(｡•́︿•̀｡)': 'Sad',
    '(◡‿◡✿)': 'Peaceful', '(︶｡︶✽)': 'Peaceful Sleep', '(✖╭╮✖)': 'Dead', '(◕д◕)': 'Amazed',
    '(｡◝‿◜｡)': 'Content', '(っ˘ω˘ς)': 'Sleepy', '(◍•ᴗ•◍)': 'Joyful', '(ノ°益°)ノ': 'Rage',
    '(๑•́ ₃ •̀๑)': 'Pouty', '(´･_･`)': 'Worried', '(｡ŏ﹏ŏ)': 'Worried', '(●´ω｀●)': 'Bashful',
    '(´｡• ᵕ •｡`)': 'Innocent', '( ･ω･)ﾉ': 'Hello', '(✿ヘᴥヘ)': 'Cute Animal', 'ʕ•ᴥ•ʔ': 'Bear',
    '(=^･ω･^=)': 'Cat', '(◕ᴥ◕ʋ)': 'Dog', '(ᵔᴥᵔ)': 'Happy Puppy', '₍ᐢ•ﻌ•ᐢ₎': 'Hamster',
    '(⁎˃ᆺ˂)': 'Cute Animal', '(◕দ◕)': 'Cute Plead',
    '(⊙_⊙)': 'Wide Eyes', '(✧ω✧)': 'Starry Eyes', '(╬ Ò﹏Ó)': 'Very Angry',
    '(◞≼◉ื≽◟ ;益;◞≼◉ื≽◟)': 'Super Angry', '(≖͞_≖̥)': 'Side Eye',
    '(＾▽＾)': 'Big Smile', '(〜￣△￣)〜': 'Dancing', '(~˘▾˘)~': 'Dancing Happy', '┏(＾0＾)┛': 'Dancing Joy',
    '┗(＾0＾)┓': 'Dancing Joy', '(◕‿◕)♡': 'Love', '(◍•ᴗ•◍)❤': 'Love Heart', '(｡♥‿♥｡)': 'In Love', '(◕‿◕✿)': 'Sweet Smile',
    '(●´□`)♡': 'Love Struck', '(´,,•ω•,,)♡': 'Loving Cute',
    '( ˘▽˘)っ♨': 'Having Tea', '(っ˘ڡ˘ς)': 'Yummy', '╮(╯▽╰)╭': 'Shrug', '(๑•̀ㅂ•́)و✧': 'Determined',
    '(ㆆ _ ㆆ)': 'Suspicious', '(´-ω-`)': 'Tired', '(´∩｡• ᵕ •｡∩`)': 'Shy Happy', '(´･ᴗ･ ` )': 'Serene',
    '( ˙▿˙ )': 'Simple Smile', '(◔◡◔)': 'Cute Smile', '(¬‿¬)': 'Smirking', '(｀皿´)': 'Angry Face',
    '(◕▿◕✿)': 'Happy Flower', '(◕ᴗ◕✿)': 'Sweet Smile', '(｡◕‿◕｡)': 'Sweet Eyes',
    '(๑˃ᴗ˂)ﻭ': 'Cheering'
}

# Sentiment mappings for emoticons
SENTIMENT_MAPPING: Dict[str, float] = {
    # Very Positive (0.8 to 1.0)
    'Laugh': 0.9, 'Very Happy': 1.0, 'Excited': 0.9, 'Excited Jump': 0.9, 'Excited Wave': 0.9,
    'Love': 1.0, 'Love Heart': 1.0, 'In Love': 1.0, 'Love Struck': 1.0, 'Loving Cute': 1.0,
    'Dancing Joy': 0.9, 'Throwing Sparkles': 0.9, 'Joyful': 0.9, 'Tears of Joy': 0.8, 'Cheering': 0.9,
    
    # Positive (0.3 to 0.7)
    'Smile': 0.7, 'Happy': 0.7, 'Big Smile': 0.7, 'Sweet Smile': 0.7, 'Cute Smile': 0.7,
    'Happy Flower': 0.7, 'Sweet Eyes': 0.7, 'Flower Smile': 0.7, 'Simple Smile': 0.5,
    'Gentle Wink': 0.6, 'Wink Star': 0.6, 'Playful Wink': 0.6, 'Wink': 0.5,
    'Content': 0.6, 'Peaceful': 0.6, 'Serene': 0.6, 'Determined': 0.5, 'Cool': 0.5,
    'Kiss': 0.6, 'Hug': 0.7, 'Big Hug': 0.7, 'Want Hug': 0.6, 'Hugging': 0.7,
    'Wave': 0.5, 'Hello': 0.5, 'Wagging': 0.5, 'Dancing': 0.6, 'Dancing Happy': 0.7,
    'Yummy': 0.6, 'Having Tea': 0.4, 'Happy Puppy': 0.7, 'Cute Animal': 0.6, 'Cute Plead': 0.5,
    'Bear': 0.4, 'Cat': 0.4, 'Dog': 0.4, 'Hamster': 0.4, 'Starry Eyes': 0.6,
    'Angel': 0.6, 'Innocent': 0.5, 'Bashful': 0.4, 'Shy Happy': 0.5, 'Shy Embarrassed': 0.3,
    'Amazed': 0.4, 'Surprised Joy': 0.6, 'Put Table Back': 0.3,
    
    # Neutral (0.0 to 0.2)
    'Neutral': 0.0, 'Tongue': 0.1, 'Tongue in Cheek': 0.1, 'Embarrassed': 0.1, 'Embarrassed Blush': 0.1,
    'Sealed Lips': 0.0, 'Sleepy': 0.1, 'Tired': 0.1, 'Bored': 0.1, 'Shrug': 0.0,
    'Confused': 0.0, 'Confused Kiss': 0.1, 'Skeptical': 0.1, 'Suspicious': 0.1, 'Wide Eyes': 0.1,
    'Surprised': 0.1, 'Shocked': 0.1, 'Partied All Night': 0.2, 'Peaceful Sleep': 0.2,
    'Side Eye': 0.1, 'Smirking': 0.1,
    
    # Negative (-0.2 to -0.7)
    'Sad': -0.6, 'Crying': -0.7, 'Sobbing': -0.7, 'Worried': -0.4, 'Nervous': -0.3,
    'Annoyed': -0.4, 'Dissatisfied': -0.5, 'Unimpressed': -0.3,
    'Disapproval': -0.4, 'Pouty': -0.3, 'Troubled': -0.5, 'Being Sick': -0.4,
    'Horror': -0.6, 'Dump': -0.2,
    
    # Very Negative (-0.8 to -1.0)
    'Angry': -0.8, 'Angry Face': -0.8, 'Very Angry': -0.9, 'Super Angry': -1.0,
    'Rage': -1.0, 'Table Flip': -0.8, 'Dead': -0.9, 'Mischievous': -0.3, 'Devil': -0.5,
}

# Pre-compiled regex patterns for optimal performance - optimized for speed
# Sort emoticons by length (longest first) to match greedily
_SORTED_EMOTICONS = sorted(EMOTICON_DICT.keys(), key=len, reverse=True)
_EMOTICON_PATTERN = '|'.join(re.escape(emoticon) for emoticon in _SORTED_EMOTICONS)

# Compile patterns once for better performance
_COMPILED_TOKEN_PATTERN = re.compile(
    f'({_EMOTICON_PATTERN})|'
    r'(https?://\S+|www\.\S+)|'
    r'(#\w+)|'
    r'(@\w+)|'
    r'(\w+)|'
    r'([^\w\s])',
    re.IGNORECASE
)

# Cache for commonly used patterns
_PUNCTUATION_SET = frozenset('.,!?;:')

def _extract_tokens(input_string: str) -> List[str]:
    """Extract tokens from input string using pre-compiled regex pattern - optimized version."""
    matches = _COMPILED_TOKEN_PATTERN.findall(input_string)
    # Use list comprehension with next() for better performance
    return [next((part for part in match if part), '') for match in matches]

def _rebuild_with_spacing(tokens: List[str], transform_fn=None) -> str:
    """Rebuild text from tokens with proper spacing - optimized version."""
    if not tokens:
        return ''
    
    # Pre-allocate list for better performance
    result = []
    
    for i, token in enumerate(tokens):
        if i > 0:
            prev_token = tokens[i-1]
            # Optimized spacing logic using set membership
            if not (prev_token in _PUNCTUATION_SET and token in _PUNCTUATION_SET):
                result.append(' ')
        
        if transform_fn:
            result.append(transform_fn(token))
        else:
            result.append(token)
    
    return ''.join(result).strip()

class SentimentAnalysis:
    """Container for sentiment analysis results."""
    
    __slots__ = ('emoticons', 'sentiments', 'scores', 'total_count', 'average_score', 'emotion_counts', 'classification')
    
    def __init__(self, emoticons: List[str], sentiments: List[str], scores: List[float]):
        self.emoticons = emoticons
        self.sentiments = sentiments
        self.scores = scores
        self.total_count = len(emoticons)
        self.average_score = sum(scores) / len(scores) if scores else 0.0
        self.emotion_counts = Counter(sentiments)
        
        # Optimized classification using direct comparison
        if self.average_score >= 0.6:
            self.classification = "Very Positive"
        elif self.average_score >= 0.3:
            self.classification = "Positive"
        elif self.average_score >= -0.3:
            self.classification = "Neutral"
        elif self.average_score >= -0.6:
            self.classification = "Negative"
        else:
            self.classification = "Very Negative"
    
    def __str__(self) -> str:
        return f"SentimentAnalysis(count={self.total_count}, avg_score={self.average_score:.3f}, classification='{self.classification}')"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def summary(self) -> str:
        """Get a detailed summary of the sentiment analysis."""
        if not self.emoticons:
            return "No emoticons found in text."
        
        summary = []
        summary.append(f"Emoticon Sentiment Analysis Summary:")
        summary.append(f"  Total emoticons: {self.total_count}")
        summary.append(f"  Average sentiment score: {self.average_score:.3f}")
        summary.append(f"  Overall classification: {self.classification}")
        summary.append(f"  Emotions detected: {dict(self.emotion_counts)}")
        
        return "\n".join(summary)

def emoticon_fix(input_string: str) -> str:
    """Transform emoticons in a string to their corresponding meanings - optimized version."""
    if not isinstance(input_string, str):
        raise TypeError("Input must be a string")
    
    tokens = _extract_tokens(input_string)
    # Use dict.get() to avoid repeated key checks
    return _rebuild_with_spacing(tokens, lambda t: EMOTICON_DICT.get(t, t))

def remove_emoticons(input_string: str) -> str:
    """Remove all emoticons from the input string - optimized version."""
    if not isinstance(input_string, str):
        raise TypeError("Input must be a string")
    
    tokens = _extract_tokens(input_string)
    # Use list comprehension with not in for better performance
    filtered_tokens = [t for t in tokens if t not in EMOTICON_DICT]
    return _rebuild_with_spacing(filtered_tokens)

def replace_emoticons(input_string: str, tag_format: str = "__EMO_{tag}__") -> str:
    """Replace emoticons with customizable NER-friendly tags - optimized version."""
    if not isinstance(input_string, str):
        raise TypeError("Input must be a string")
    
    if "{tag}" not in tag_format:
        raise ValueError("tag_format must contain the {tag} placeholder")
    
    tokens = _extract_tokens(input_string)
    
    # Use optimized transform function with get()
    def transform(token):
        emotion = EMOTICON_DICT.get(token)
        return tag_format.format(tag=emotion) if emotion else token
    
    return _rebuild_with_spacing(tokens, transform)

def analyze_sentiment(input_string: str) -> SentimentAnalysis:
    """Analyze the sentiment of emoticons in the input string - optimized version."""
    if not isinstance(input_string, str):
        raise TypeError("Input must be a string")
    
    tokens = _extract_tokens(input_string)
    
    # Pre-allocate lists for better performance
    emoticons = []
    sentiments = []
    scores = []
    
    # Single pass through tokens with optimized lookups
    for token in tokens:
        emotion = EMOTICON_DICT.get(token)
        if emotion:
            emoticons.append(token)
            sentiments.append(emotion)
            scores.append(SENTIMENT_MAPPING.get(emotion, 0.0))
    
    return SentimentAnalysis(emoticons, sentiments, scores)

def get_sentiment_score(input_string: str) -> float:
    """Get the average sentiment score for emoticons in the text."""
    analysis = analyze_sentiment(input_string)
    return analysis.average_score

def classify_sentiment(input_string: str) -> str:
    """Classify the overall sentiment of emoticons in the text."""
    analysis = analyze_sentiment(input_string)
    return analysis.classification

def extract_emotions(input_string: str) -> List[Tuple[str, str, float]]:
    """Extract all emoticons with their emotions and sentiment scores."""
    analysis = analyze_sentiment(input_string)
    return list(zip(analysis.emoticons, analysis.sentiments, analysis.scores))

def batch_analyze(texts: List[str]) -> List[SentimentAnalysis]:
    """Analyze sentiment for multiple texts efficiently - optimized version."""
    if not isinstance(texts, list):
        raise TypeError("Input must be a list of strings")
    
    # Use list comprehension for better performance
    return [analyze_sentiment(text) for text in texts]

class EmoticonStats:
    """Container for emoticon usage statistics."""
    
    def __init__(self, text: str = ""):
        self.text = text
        self.total_emoticons = 0
        self.unique_emoticons = 0
        self.emoticon_frequency = Counter()
        self.emotion_frequency = Counter()
        self.sentiment_distribution = {"positive": 0, "negative": 0, "neutral": 0}
        self.average_sentiment = 0.0
        self.dominant_emotion = "None"
        self.emoticon_positions = []  # List of (emoticon, position) tuples
        self.analysis_timestamp = datetime.now().isoformat()
        
        if text:
            self._analyze_text(text)
    
    def _analyze_text(self, text: str):
        """Analyze the text and populate statistics."""
        tokens = _extract_tokens(text)
        
        emotions_found = []
        for i, token in enumerate(tokens):
            if token in EMOTICON_DICT:
                emotion = EMOTICON_DICT[token]
                score = SENTIMENT_MAPPING.get(emotion, 0.0)
                
                self.emoticon_frequency[token] += 1
                self.emotion_frequency[emotion] += 1
                self.emoticon_positions.append((token, i))
                emotions_found.append(score)
                
                # Categorize sentiment
                if score > 0.3:
                    self.sentiment_distribution["positive"] += 1
                elif score < -0.3:
                    self.sentiment_distribution["negative"] += 1
                else:
                    self.sentiment_distribution["neutral"] += 1
        
        self.total_emoticons = len(emotions_found)
        self.unique_emoticons = len(self.emoticon_frequency)
        self.average_sentiment = sum(emotions_found) / len(emotions_found) if emotions_found else 0.0
        self.dominant_emotion = self.emotion_frequency.most_common(1)[0][0] if self.emotion_frequency else "None"
    
    def get_top_emoticons(self, n: int = 5) -> List[Tuple[str, int]]:
        """Get the top N most frequent emoticons."""
        return self.emoticon_frequency.most_common(n)
    
    def get_top_emotions(self, n: int = 5) -> List[Tuple[str, int]]:
        """Get the top N most frequent emotions."""
        return self.emotion_frequency.most_common(n)
    
    def get_emoticon_density(self) -> float:
        """Calculate emoticons per 100 characters."""
        if not self.text:
            return 0.0
        return (self.total_emoticons / len(self.text)) * 100
    
    def to_dict(self) -> Dict:
        """Convert statistics to dictionary format."""
        return {
            "total_emoticons": self.total_emoticons,
            "unique_emoticons": self.unique_emoticons,
            "emoticon_density": self.get_emoticon_density(),
            "emoticon_frequency": dict(self.emoticon_frequency),
            "emotion_frequency": dict(self.emotion_frequency),
            "sentiment_distribution": self.sentiment_distribution,
            "average_sentiment": self.average_sentiment,
            "dominant_emotion": self.dominant_emotion,
            "top_emoticons": self.get_top_emoticons(),
            "top_emotions": self.get_top_emotions(),
            "analysis_timestamp": self.analysis_timestamp
        }
    
    def __str__(self) -> str:
        return f"EmoticonStats(total={self.total_emoticons}, unique={self.unique_emoticons}, dominant='{self.dominant_emotion}', avg_sentiment={self.average_sentiment:.3f})"

class EmoticonProfile:
    """Container for emotion profiling of texts or users."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.texts_analyzed = 0
        self.total_emoticons = 0
        self.aggregate_emotions = Counter()
        self.aggregate_sentiments = []
        self.text_stats = []  # List of EmoticonStats for each text
        self.creation_timestamp = datetime.now().isoformat()
    
    def add_text(self, text: str, text_id: str = None) -> EmoticonStats:
        """Add a text to the profile and return its statistics."""
        stats = EmoticonStats(text)
        stats.text_id = text_id or f"text_{self.texts_analyzed + 1}"
        
        self.text_stats.append(stats)
        self.texts_analyzed += 1
        self.total_emoticons += stats.total_emoticons
        
        # Aggregate data
        self.aggregate_emotions.update(stats.emotion_frequency)
        self.aggregate_sentiments.extend([SENTIMENT_MAPPING.get(emotion, 0.0) 
                                        for emotion in stats.emotion_frequency.elements()])
        
        return stats
    
    def get_overall_sentiment(self) -> float:
        """Get the overall average sentiment across all texts."""
        return sum(self.aggregate_sentiments) / len(self.aggregate_sentiments) if self.aggregate_sentiments else 0.0
    
    def get_dominant_emotions(self, n: int = 5) -> List[Tuple[str, int]]:
        """Get the top N dominant emotions across all texts."""
        return self.aggregate_emotions.most_common(n)
    
    def get_emotion_diversity(self) -> float:
        """Calculate emotion diversity (number of unique emotions / total emotions)."""
        if self.total_emoticons == 0:
            return 0.0
        return len(self.aggregate_emotions) / self.total_emoticons
    
    def get_sentiment_consistency(self) -> float:
        """Calculate sentiment consistency (lower values indicate more consistent sentiment)."""
        if len(self.aggregate_sentiments) < 2:
            return 0.0
        
        import math
        mean_sentiment = self.get_overall_sentiment()
        variance = sum((s - mean_sentiment) ** 2 for s in self.aggregate_sentiments) / len(self.aggregate_sentiments)
        return math.sqrt(variance)
    
    def to_dict(self) -> Dict:
        """Convert profile to dictionary format."""
        return {
            "name": self.name,
            "texts_analyzed": self.texts_analyzed,
            "total_emoticons": self.total_emoticons,
            "overall_sentiment": self.get_overall_sentiment(),
            "dominant_emotions": self.get_dominant_emotions(),
            "emotion_diversity": self.get_emotion_diversity(),
            "sentiment_consistency": self.get_sentiment_consistency(),
            "aggregate_emotions": dict(self.aggregate_emotions),
            "text_stats": [stats.to_dict() for stats in self.text_stats],
            "creation_timestamp": self.creation_timestamp
        }
    
    def __str__(self) -> str:
        return f"EmoticonProfile(name='{self.name}', texts={self.texts_analyzed}, emoticons={self.total_emoticons}, sentiment={self.get_overall_sentiment():.3f})"

def get_emoticon_statistics(text: str) -> EmoticonStats:
    """Get comprehensive statistics about emoticon usage in text."""
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    
    return EmoticonStats(text)

def create_emotion_profile(texts: Union[str, List[str]], profile_name: str = "Profile") -> EmoticonProfile:
    """Create an emotion profile from one or more texts."""
    if isinstance(texts, str):
        texts = [texts]
    
    if not isinstance(texts, list):
        raise TypeError("Input must be a string or list of strings")
    
    profile = EmoticonProfile(profile_name)
    
    for i, text in enumerate(texts):
        profile.add_text(text, f"text_{i+1}")
    
    return profile

def compare_emotion_profiles(profiles: List[EmoticonProfile]) -> Dict:
    """Compare multiple emotion profiles and return comparison statistics."""
    if not profiles or not isinstance(profiles, list):
        raise ValueError("Must provide a list of EmoticonProfile objects")
    
    comparison = {
        "profiles_compared": len(profiles),
        "comparison_timestamp": datetime.now().isoformat(),
        "profile_summaries": [],
        "overall_comparison": {}
    }
    
    # Individual profile summaries
    sentiments = []
    diversities = []
    consistencies = []
    
    for profile in profiles:
        summary = {
            "name": profile.name,
            "texts_analyzed": profile.texts_analyzed,
            "total_emoticons": profile.total_emoticons,
            "overall_sentiment": profile.get_overall_sentiment(),
            "emotion_diversity": profile.get_emotion_diversity(),
            "sentiment_consistency": profile.get_sentiment_consistency(),
            "dominant_emotions": profile.get_dominant_emotions(3)
        }
        comparison["profile_summaries"].append(summary)
        
        sentiments.append(profile.get_overall_sentiment())
        diversities.append(profile.get_emotion_diversity())
        consistencies.append(profile.get_sentiment_consistency())
    
    # Overall comparison statistics
    comparison["overall_comparison"] = {
        "sentiment_range": {
            "min": min(sentiments) if sentiments else 0,
            "max": max(sentiments) if sentiments else 0,
            "average": sum(sentiments) / len(sentiments) if sentiments else 0
        },
        "diversity_range": {
            "min": min(diversities) if diversities else 0,
            "max": max(diversities) if diversities else 0,
            "average": sum(diversities) / len(diversities) if diversities else 0
        },
        "consistency_range": {
            "min": min(consistencies) if consistencies else 0,
            "max": max(consistencies) if consistencies else 0,
            "average": sum(consistencies) / len(consistencies) if consistencies else 0
        }
    }
    
    return comparison

def export_analysis(data: Union[EmoticonStats, EmoticonProfile, Dict], 
                   format: str = "json", 
                   filename: Optional[str] = None) -> str:
    """Export analysis results to JSON or CSV format."""
    
    # Convert to dictionary if needed
    if hasattr(data, 'to_dict'):
        data_dict = data.to_dict()
    elif isinstance(data, dict):
        data_dict = data
    else:
        raise TypeError("Data must be EmoticonStats, EmoticonProfile, or dict")
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"emoticon_analysis_{timestamp}.{format}"
    
    if format.lower() == "json":
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)
    
    elif format.lower() == "csv":
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Handle different data types
            if isinstance(data, EmoticonStats):
                # Write emoticon statistics as CSV
                writer.writerow(["Metric", "Value"])
                writer.writerow(["Total Emoticons", data_dict["total_emoticons"]])
                writer.writerow(["Unique Emoticons", data_dict["unique_emoticons"]])
                writer.writerow(["Emoticon Density", data_dict["emoticon_density"]])
                writer.writerow(["Average Sentiment", data_dict["average_sentiment"]])
                writer.writerow(["Dominant Emotion", data_dict["dominant_emotion"]])
                
                writer.writerow([])  # Empty row
                writer.writerow(["Emoticon", "Frequency"])
                for emoticon, freq in data_dict["emoticon_frequency"].items():
                    writer.writerow([emoticon, freq])
                
                writer.writerow([])  # Empty row
                writer.writerow(["Emotion", "Frequency"])
                for emotion, freq in data_dict["emotion_frequency"].items():
                    writer.writerow([emotion, freq])
            
            else:
                # Flatten nested dictionaries for CSV
                def flatten_dict(d, parent_key='', sep='_'):
                    items = []
                    for k, v in d.items():
                        new_key = f"{parent_key}{sep}{k}" if parent_key else k
                        if isinstance(v, dict):
                            items.extend(flatten_dict(v, new_key, sep=sep).items())
                        elif isinstance(v, list):
                            items.append((new_key, str(v)))
                        else:
                            items.append((new_key, v))
                    return dict(items)
                
                flattened = flatten_dict(data_dict)
                writer.writerow(["Key", "Value"])
                for key, value in flattened.items():
                    writer.writerow([key, value])
    
    else:
        raise ValueError("Format must be 'json' or 'csv'")
    
    return filename

def get_emoticon_trends(texts: List[str], text_labels: Optional[List[str]] = None) -> Dict:
    """Analyze emoticon trends across multiple texts."""
    if not isinstance(texts, list):
        raise TypeError("texts must be a list of strings")
    
    if text_labels and len(text_labels) != len(texts):
        raise ValueError("text_labels must have the same length as texts")
    
    trends = {
        "analysis_timestamp": datetime.now().isoformat(),
        "total_texts": len(texts),
        "text_analyses": [],
        "trend_summary": {}
    }
    
    all_sentiments = []
    all_emotions = Counter()
    
    for i, text in enumerate(texts):
        label = text_labels[i] if text_labels else f"text_{i+1}"
        stats = get_emoticon_statistics(text)
        
        analysis = {
            "label": label,
            "stats": stats.to_dict()
        }
        trends["text_analyses"].append(analysis)
        
        all_sentiments.append(stats.average_sentiment)
        all_emotions.update(stats.emotion_frequency)
    
    # Calculate trends
    trends["trend_summary"] = {
        "sentiment_trend": all_sentiments,
        "overall_emotion_frequency": dict(all_emotions),
        "most_common_emotions": all_emotions.most_common(10),
        "average_sentiment_across_texts": sum(all_sentiments) / len(all_sentiments) if all_sentiments else 0
    }
    
    return trends