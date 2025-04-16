#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A Python package to transform emoticons in text to their corresponding meanings.
This is useful for NLP preprocessing where emoticons need to be preserved as meaningful text.
"""

import re
from typing import Dict, List, Union, Optional

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
    '(ーー;)': 'Annoyed',
    '(｡♥‿♥｡)': 'Love',
    '(✿◠‿◠)': 'Happy',
    '(◕‿◕✿)': 'Sweet',
    '(づ｡◕‿‿◕｡)づ': 'Hug',
    '(ﾉ◕ヮ◕)ﾉ*:･ﾟ✧': 'Sparkle',
    '(╯°□°）╯︵ ┻━┻': 'Table Flip',
    '┬─┬ノ( º _ ºノ)': 'Put Table Back',
    '(¬_¬)': 'Disapproval',
    '(；⌣̀_⌣́)': 'Nervous',
    '(╥﹏╥)': 'Sobbing',
    '(｡•́︿•̀｡)': 'Sad',
    '(◡‿◡✿)': 'Peaceful',
    '(✖╭╮✖)': 'Dead',
    '(◕д◕)': 'Amazed',
    '(｡◝‿◜｡)': 'Content',
    '(っ˘ω˘ς)': 'Sleepy',
    '⊂((・▽・))⊃': 'Big Hug',
    '(◍•ᴗ•◍)': 'Joyful',
    '(｀∀´)Ψ': 'Evil',
    '(ノ°益°)ノ': 'Rage',
    '(๑•́ ₃ •̀๑)': 'Pouty',
    '(´･_･`)': 'Worried',
    '(◠‿◠✿)': 'Flower Smile',
    '(✿ヘᴥヘ)': 'Cute Animal',
    'ʕ•ᴥ•ʔ': 'Bear',
    '(=^･ω･^=)': 'Cat',
    '(◕ᴥ◕ʋ)': 'Dog',
    '(｡♥‿♥｡)': 'Heart Eyes',
    '(⊙_⊙)': 'Wide Eyes',
    '(✧ω✧)': 'Starry Eyes',
    '(╬ Ò﹏Ó)': 'Very Angry',
    '(＾▽＾)': 'Big Smile',
    '(〜￣△￣)〜': 'Dancing',
    '(･ω･)つ⊂(･ω･)': 'Hugging',
    '(◕‿◕)♡': 'Love',
    '(｡•̀ᴗ-)✧': 'Winking',
    '( ˘▽˘)っ♨': 'Having Tea',
    '(っ˘ڡ˘ς)': 'Yummy',
    '╮(╯▽╰)╭': 'Shrug',
    '(๑•̀ㅂ•́)و✧': 'Determined',
    '(ㆆ _ ㆆ)': 'Suspicious',
    '(｡ŏ﹏ŏ)': 'Worried',
    '⊙﹏⊙': 'Confused',
    '(●´ω｀●)': 'Bashful',
    '(´｡• ᵕ •｡`)': 'Innocent',
    '( ･ω･)ﾉ': 'Hello',
    '(｡ì _ í｡)': 'Skeptical',
    '(︶｡︶✽)': 'Peaceful Sleep',
    '(╯✧▽✧)╯': 'Excited Jump',
    'o(≧▽≦)o': 'Very Happy',
    '(＃￣ω￣)': 'Dissatisfied',
    '(◔◡◔)': 'Cute Smile',
    '(；一_一)': 'Unimpressed',
    '(¬‿¬)': 'Smirking',
    '(~˘▾˘)~': 'Dancing Happy',
    '(｀皿´)': 'Angry Face',
    '(；◔ิз◔ิ)': 'Confused Kiss',
    '┏(＾0＾)┛': 'Dancing Joy',
    '┗(＾0＾)┓': 'Dancing Joy',
    '(｡♥‿♥｡)': 'In Love',
    '(ᵔᴥᵔ)': 'Happy Puppy',
    '(◕ᴗ◕✿)': 'Sweet Smile',
    '(｡•̀‿-)✧': 'Playful Wink',
    '(ﾉ◕ヮ◕)ﾉ*:･ﾟ✧': 'Throwing Sparkles',
    '(´∩｡• ᵕ •｡∩`)': 'Shy Happy',
    '(◍•ᴗ•◍)❤': 'Love Heart',
    '(⁄ ⁄•⁄ω⁄•⁄ ⁄)': 'Embarrassed Blush',
    '(╬☉д⊙)⊰⊹ฺ': 'Shocked',
    '(∩╹□╹∩)': 'Surprised Joy',
    '(⊃｡•́‿•̀｡)⊃': 'Want Hug',
    '(◞≼◉ื≽◟ ;益;◞≼◉ื≽◟)': 'Super Angry',
    '(≖͞_≖̥)': 'Side Eye',
    '(◕દ◕)': 'Cute Plead',
    '(｡ﾉω＼｡)': 'Shy Embarrassed',
    '(´,,•ω•,,)♡': 'Loving Cute',
    '(⊙ω⊙)': 'Wide Eyed',
    '(◕‿◕✿)': 'Flower Happy',
    '(｡•̀ᴗ-)✧': 'Wink Star',
    '(●´□`)♡': 'Love Struck',
    '(⁎˃ᆺ˂)': 'Cute Animal',
    '(๑˃ᴗ˂)ﻭ': 'Cheering',
    '(｀∀´)Ψ': 'Mischievous',
    '(´-ω-`)': 'Tired',
    '(◡‿◡✿)': 'Gentle Smile',
    '₍ᐢ•ﻌ•ᐢ₎': 'Hamster',
    '(´･ᴗ･ ` )': 'Serene',
    '(ノ•̀ o •́ )ノ': 'Excited Wave',
    '( ˙▿˙ )': 'Simple Smile',
    '(◕▿◕✿)': 'Happy Flower',
    '(｡•̀ᴗ-)': 'Gentle Wink',
    '(｡◕‿◕｡)': 'Sweet Eyes'
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

def remove_emoticons(input_string: str) -> str:
    """
    Remove all emoticons from the input string.
    
    Args:
        input_string (str): The input string containing emoticons
        
    Returns:
        str: The string with all emoticons removed
        
    Example:
        >>> remove_emoticons('Hello :) World :D')
        'Hello World'
    """
    if not isinstance(input_string, str):
        raise TypeError("Input must be a string")
        
    tknzr = CustomTokenizer()
    tokens = tknzr.tokenize(input_string)
    
    result = []
    last_added_idx = -1  # Track the index of the last token we added
    
    for i, token in enumerate(tokens):
        # Skip emoticons
        if token in EMOTICON_DICT:
            continue
            
        # Handle spacing
        if result:  # If we've already added something
            # Check if we need to add space
            if last_added_idx >= 0:  # If we have added a token before
                last_token = tokens[last_added_idx]
                
                # Add space after punctuation (except closing punctuation)
                if last_token in '.,!?;:' and token not in '.,!?;:':
                    result.append(' ')
                # Add space between words
                elif not last_token.strip() or (last_token not in '.,!?;:' and token not in '.,!?;:'):
                    result.append(' ')
        
        # Add the non-emoticon token
        result.append(token)
        last_added_idx = i  # Update last added token index
            
    return ''.join(result)

def replace_emoticons(input_string: str, tag_format: str = "__EMO_{tag}__") -> str:
    """
    Replace emoticons with customizable NER-friendly tags.
    
    Args:
        input_string (str): The input string containing emoticons
        tag_format (str): Format string for the replacement tag. 
                         Use {tag} as a placeholder for the emoticon meaning.
        
    Returns:
        str: The string with emoticons replaced by formatted tags
        
    Example:
        >>> replace_emoticons('Hello :) World :D', tag_format="__EMO_{tag}__")
        'Hello __EMO_Smile__ World __EMO_Laugh__'
        
        >>> replace_emoticons('Hello :) World :D', tag_format="<EMO:{tag}>")
        'Hello <EMO:Smile> World <EMO:Laugh>'
    """
    if not isinstance(input_string, str):
        raise TypeError("Input must be a string")
        
    if "{tag}" not in tag_format:
        raise ValueError("tag_format must contain the {tag} placeholder")
        
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
            
        # Add the token (either formatted emoticon tag or original token)
        if token in EMOTICON_DICT:
            meaning = EMOTICON_DICT[token]
            formatted_tag = tag_format.format(tag=meaning)
            result.append(formatted_tag)
        else:
            result.append(token)
            
    return ''.join(result)