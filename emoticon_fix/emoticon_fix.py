#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A Python package to transform emoticons in text to their corresponding meanings.
This is useful for NLP preprocessing where emoticons need to be preserved as meaningful text.
"""

import re
from typing import Dict, List, Union

# Dictionary mapping emoticons to their meanings
EMOTICON_DICT: Dict[str, str] = {
    ':)': 'Smile',
    ':-)': 'Smile',
    ':-]': 'Smile',
    ':]': 'Smile',
    ':-3': 'Smile',
    ':3': 'Smile',
    ':->': 'Smile',
    ':>': 'Smile',
    '8-)': 'Smile',
    '8)': 'Smile',
    ':-}': 'Smile',
    ':}': 'Smile',
    ':o)': 'Smile',
    ':c)': 'Smile',
    ':^)': 'Smile',
    '=]': 'Smile',
    '=)': 'Smile',
    ':-D': 'Laugh',
    ':D': 'Laugh',
    '8-D': 'Laugh',
    '8D': 'Laugh',
    'x-D': 'Laugh',
    'xD': 'Laugh',
    'X-D': 'Laugh',
    'XD': 'Laugh',
    '=D': 'Laugh',
    '=3': 'Laugh',
    'B^D': 'Laugh',
    ':-))': 'Very Happy',
    ':-(': 'Sad',
    ':(': 'Sad',
    ':-c': 'Sad',
    ':c': 'Sad',
    ':-<': 'Sad',
    ':<': 'Sad',
    ':-[': 'Sad',
    ':[': 'Sad',
    ':-||': 'Angry',
    '>:[': 'Angry',
    ':{': 'Angry',
    ':@': 'Angry',
    '>:(': 'Angry',
    "('_')": 'Crying',
    '(/_;)': 'Crying',
    '(T_T)': 'Crying',
    '(;_;)': 'Crying',
    '(;_;': 'Crying',
    '(;_:)': 'Crying',
    '(;O;)': 'Crying',
    '(:_;)': 'Crying',
    '(ToT)': 'Crying',
    '(Ｔ▽Ｔ)': 'Crying',
    ';_;': 'Crying',
    ';-;': 'Crying',
    ';n;': 'Crying',
    ';;': 'Crying',
    'Q.Q': 'Crying',
    'T.T': 'Crying',
    'TnT': 'Crying',
    'QQ': 'Crying',
    'Q_Q': 'Crying',
    '(ー_ー)!!': 'Annoyed',
    '(-.-)': 'Annoyed',
    '(-_-)': 'Annoyed',
    '(一一)': 'Annoyed',
    '(；一_一)': 'Annoyed',
    '＼(~o~)／': 'Excited',
    '＼(^o^)／': 'Excited',
    '＼(-o-)／': 'Excited',
    'ヽ(^。^)ノ': 'Excited',
    'ヽ(^o^)丿': 'Excited',
    '(*^0^*)': 'Excited',
    ":'-(": 'Crying',
    ":'(": 'Crying',
    ":'-)": 'Tears of Joy',
    ":')": 'Tears of Joy',
    "D-':": 'Horror',
    'D:<': 'Horror',
    'D:': 'Horror',
    'D8': 'Horror',
    'D;': 'Horror',
    'D=': 'Horror',
    'DX': 'Horror',
    ':-O': 'Surprised',
    ':O': 'Surprised',
    ':-o': 'Surprised',
    ':o': 'Surprised',
    ':-0': 'Surprised',
    '8-0': 'Surprised',
    '>:O': 'Surprised',
    '(￣□￣;)': 'Surprised',
    '°o°': 'Surprised',
    '°O°': 'Surprised',
    'o_0': 'Surprised',
    'o.O': 'Surprised',
    '(o.o)': 'Surprised',
    'oO': 'Surprised',
    ':-*': 'Kiss',
    ':*': 'Kiss',
    ':×': 'Kiss',
    ';-)': 'Wink',
    ';)': 'Wink',
    '*-)': 'Wink',
    '*)': 'Wink',
    ';-]': 'Wink',
    ';]': 'Wink',
    ';^)': 'Wink',
    ':-,': 'Wink',
    ';D': 'Wink',
    ':-P': 'Tongue',
    ':P': 'Tongue',
    'X-P': 'Tongue',
    'XP': 'Tongue',
    'x-p': 'Tongue',
    'xp': 'Tongue',
    ':-p': 'Tongue',
    ':p': 'Tongue',
    ':-Þ': 'Tongue',
    ':Þ': 'Tongue',
    ':-þ': 'Tongue',
    ':þ': 'Tongue',
    ':-b': 'Tongue',
    ':b': 'Tongue',
    'd:': 'Tongue',
    '=p': 'Tongue',
    '>:P': 'Tongue',
    ':-/': 'Skeptical',
    ':/': 'Skeptical',
    ':-.': 'Skeptical',
    '>:\\': 'Skeptical',
    '>:/': 'Skeptical',
    ':\\': 'Skeptical',
    '=/': 'Skeptical',
    '=\\': 'Skeptical',
    ':L': 'Skeptical',
    '=L': 'Skeptical',
    ':S': 'Skeptical',
    ':-|': 'Neutral',
    ':|': 'Neutral',
    ':$': 'Embarrassed',
    '://)': 'Embarrassed',
    '://3': 'Embarrassed',
    '(^^ゞ': 'Embarrassed',
    '(^_^;)': 'Embarrassed',
    '(-_-;)': 'Embarrassed',
    '(~_~;)': 'Embarrassed',
    '(・.・;)': 'Embarrassed',
    '(・_・;)': 'Embarrassed',
    '(・・;)^^': 'Embarrassed',
    ';^_^;': 'Embarrassed',
    '(#^.^#)': 'Embarrassed',
    '(^^;)': 'Embarrassed',
    ':-X': 'Sealed Lips',
    ':X': 'Sealed Lips',
    ':-#': 'Sealed Lips',
    ':#': 'Sealed Lips',
    ':-&': 'Sealed Lips',
    ':&': 'Sealed Lips',
    'O:-)': 'Angel',
    'O:)': 'Angel',
    '0:-3': 'Angel',
    '0:3': 'Angel',
    '0:-)': 'Angel',
    '0:)': 'Angel',
    '0;^)': 'Angel',
    '>:-)': 'Devil',
    '>:)': 'Devil',
    '}:-)': 'Devil',
    '}:)': 'Devil',
    '3:-)': 'Devil',
    '3:)': 'Devil',
    '>;)': 'Devil',
    '>:3': 'Devil',
    '>;3': 'Devil',
    ':‑J': 'Tongue in Cheek',
    '#‑)': 'Partied All Night',
    '|;-)': 'Cool',
    '|-O': 'Bored',
    '%-)': 'Confused',
    '%)': 'Confused',
    ':-###..': 'Being Sick',
    ':###.': 'Being Sick',
    '<:-|': 'Dump',
    '(>_<)': 'Troubled',
    '(>_<)>': 'Troubled',
    '((+_+))': 'Troubled',
    '(+o+)': 'Troubled',
    '(°°)': 'Shocked',
    '(°-°)': 'Shocked',
    '(°.°)': 'Shocked',
    '(°_°)': 'Shocked',
    '(°_°>)': 'Shocked',
    '(°レ°)': 'Shocked',
    '^_^': 'Happy',
    '(°o°)': 'Shocked',
    '(^_^)/': 'Wave',
    '(^O^)／': 'Wave',
    '(^o^)／': 'Wave',
    '(^^)/': 'Wave',
    '(-∇-)/': 'Wave',
    '(/-ヮ-)/': 'Wave',
    '(^o^)丿': 'Wave',
    '∩(·ω·)∩': 'Wagging',
    '(·ω·)': 'Wagging',
    '^ω^': 'Wagging',
    '"(-""-)"': 'Annoyed',
    '(ーー゛)': 'Annoyed',
    '(^_^メ)': 'Annoyed',
    '(-_-メ)': 'Annoyed',
    '(~_~メ)': 'Annoyed',
    '(－－〆)': 'Annoyed',
    '(・へ・)': 'Annoyed',
    '(｀´)': 'Annoyed',
    '<`～´>': 'Annoyed',
    '<`ヘ´>': 'Annoyed',
    '(ーー;)': 'Annoyed'
}

class CustomTokenizer:
    """
    A custom tokenizer that handles emoticons, URLs, hashtags, and mentions.
    Similar to NLTK's TweetTokenizer but without the dependency.
    """
    
    def __init__(self, preserve_case: bool = True):
        """
        Initialize the tokenizer.
        
        Args:
            preserve_case (bool): Whether to preserve the case of tokens
        """
        self.preserve_case = preserve_case
        
        # Compile regex patterns
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.mention_pattern = re.compile(r'@\w+')
        self.emoticon_pattern = re.compile(r'|'.join(re.escape(emoticon) for emoticon in EMOTICON_DICT.keys()))
        self.word_pattern = re.compile(r'\w+')
        self.punctuation_pattern = re.compile(r'[^\w\s]')
        
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text.
        
        Args:
            text (str): The input text to tokenize
            
        Returns:
            List[str]: List of tokens
        """
        if not isinstance(text, str):
            raise TypeError("Input must be a string")
            
        # Find all matches for each pattern
        tokens = []
        pos = 0
        
        while pos < len(text):
            # Try to match each pattern in order of priority
            match = None
            
            # Try URL first
            url_match = self.url_pattern.match(text, pos)
            if url_match:
                match = url_match
                
            # Then try emoticons
            if not match:
                emoticon_match = self.emoticon_pattern.match(text, pos)
                if emoticon_match:
                    match = emoticon_match
                    
            # Then try hashtags
            if not match:
                hashtag_match = self.hashtag_pattern.match(text, pos)
                if hashtag_match:
                    match = hashtag_match
                    
            # Then try mentions
            if not match:
                mention_match = self.mention_pattern.match(text, pos)
                if mention_match:
                    match = mention_match
                    
            # Then try words
            if not match:
                word_match = self.word_pattern.match(text, pos)
                if word_match:
                    match = word_match
                    
            # Finally try punctuation
            if not match:
                punct_match = self.punctuation_pattern.match(text, pos)
                if punct_match:
                    match = punct_match
                    
            if match:
                token = match.group(0)
                if not self.preserve_case and token not in EMOTICON_DICT:
                    token = token.lower()
                tokens.append(token)
                pos = match.end()
            else:
                # Skip whitespace
                pos += 1
                
        return tokens

def emoticon_fix(input_string: str) -> str:
    """
    Transform emoticons in a string to their corresponding meanings.
    
    Args:
        input_string (str): The input string containing emoticons
        
    Returns:
        str: The string with emoticons replaced by their meanings
        
    Example:
        >>> emoticon_fix('Hello :) World :D')
        'Hello Smile World Laugh'
    """
    if not isinstance(input_string, str):
        raise TypeError("Input must be a string")
        
    tknzr = CustomTokenizer()
    tokens = tknzr.tokenize(input_string)
    
    result = []
    for i, token in enumerate(tokens):
        # Handle spaces before current token
        if i > 0:
            prev_token = tokens[i-1]
            
            # Add space after punctuation (except closing punctuation)
            if prev_token in '.,!?;:' and token not in '.,!?;:':
                result.append(' ')
            # Add space between words/emoticons
            elif not prev_token.strip() or (prev_token not in '.,!?;:' and token not in '.,!?;:'):
                result.append(' ')
            
        # Add the token (either emoticon meaning or original token)
        if token in EMOTICON_DICT:
            result.append(EMOTICON_DICT[token])
        else:
            result.append(token)
            
    return ''.join(result)