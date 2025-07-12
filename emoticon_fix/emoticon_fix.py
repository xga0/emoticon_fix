#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A Python package to transform emoticons in text to their corresponding meanings.
This is useful for NLP preprocessing where emoticons need to be preserved as meaningful text.
"""

import re
from typing import Dict, List, Tuple
from collections import Counter

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

# Pre-compiled regex patterns for optimal performance
_EMOTICON_PATTERN = '|'.join(re.escape(emoticon) for emoticon in sorted(EMOTICON_DICT.keys(), key=len, reverse=True))
_TOKEN_PATTERN = re.compile(
    f'({_EMOTICON_PATTERN})|'
    r'(https?://\S+|www\.\S+)|'
    r'(#\w+)|'
    r'(@\w+)|'
    r'(\w+)|'
    r'([^\w\s])'
)

def _extract_tokens(input_string: str) -> List[str]:
    """Extract tokens from input string using pre-compiled regex pattern."""
    return [match[0] or match[1] or match[2] or match[3] or match[4] or match[5] 
            for match in _TOKEN_PATTERN.finditer(input_string)]

def _rebuild_with_spacing(tokens: List[str], transform_fn=None) -> str:
    """Rebuild text from tokens with proper spacing."""
    result = []
    for i, token in enumerate(tokens):
        if i > 0:
            prev_token = tokens[i-1]
            if prev_token not in '.,!?;:' or token not in '.,!?;:':
                if not (prev_token in '.,!?;:' and token in '.,!?;:'):
                    result.append(' ')
        
        if transform_fn:
            result.append(transform_fn(token))
        else:
            result.append(token)
    
    return ''.join(result).strip()

class SentimentAnalysis:
    """Container for sentiment analysis results."""
    
    def __init__(self, emoticons: List[str], sentiments: List[str], scores: List[float]):
        self.emoticons = emoticons
        self.sentiments = sentiments
        self.scores = scores
        self.total_count = len(emoticons)
        self.average_score = sum(scores) / len(scores) if scores else 0.0
        self.emotion_counts = Counter(sentiments)
        
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
    """Transform emoticons in a string to their corresponding meanings."""
    if not isinstance(input_string, str):
        raise TypeError("Input must be a string")
    
    tokens = _extract_tokens(input_string)
    return _rebuild_with_spacing(tokens, lambda t: EMOTICON_DICT.get(t, t))

def remove_emoticons(input_string: str) -> str:
    """Remove all emoticons from the input string."""
    if not isinstance(input_string, str):
        raise TypeError("Input must be a string")
    
    tokens = _extract_tokens(input_string)
    filtered_tokens = [t for t in tokens if t not in EMOTICON_DICT]
    return _rebuild_with_spacing(filtered_tokens)

def replace_emoticons(input_string: str, tag_format: str = "__EMO_{tag}__") -> str:
    """Replace emoticons with customizable NER-friendly tags."""
    if not isinstance(input_string, str):
        raise TypeError("Input must be a string")
    
    if "{tag}" not in tag_format:
        raise ValueError("tag_format must contain the {tag} placeholder")
    
    tokens = _extract_tokens(input_string)
    
    def transform(token):
        if token in EMOTICON_DICT:
            return tag_format.format(tag=EMOTICON_DICT[token])
        return token
    
    return _rebuild_with_spacing(tokens, transform)

def analyze_sentiment(input_string: str) -> SentimentAnalysis:
    """Analyze the sentiment of emoticons in the input string."""
    if not isinstance(input_string, str):
        raise TypeError("Input must be a string")
    
    tokens = _extract_tokens(input_string)
    
    emoticons = []
    sentiments = []
    scores = []
    
    for token in tokens:
        if token in EMOTICON_DICT:
            emoticons.append(token)
            sentiment = EMOTICON_DICT[token]
            sentiments.append(sentiment)
            scores.append(SENTIMENT_MAPPING.get(sentiment, 0.0))
    
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
    """Analyze sentiment for multiple texts efficiently."""
    if not isinstance(texts, list):
        raise TypeError("Input must be a list of strings")
    
    return [analyze_sentiment(text) for text in texts]