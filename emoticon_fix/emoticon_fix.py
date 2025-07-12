#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A Python package to transform emoticons in text to their corresponding meanings.
This is useful for NLP preprocessing where emoticons need to be preserved as meaningful text.
"""

import re
from typing import Dict, List

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

# Pre-compiled regex patterns for optimal performance
_EMOTICON_PATTERN = '|'.join(re.escape(emoticon) for emoticon in sorted(EMOTICON_DICT.keys(), key=len, reverse=True))
_TOKEN_PATTERN = re.compile(
    f'({_EMOTICON_PATTERN})|'  # Emoticons (highest priority)
    r'(https?://\S+|www\.\S+)|'  # URLs
    r'(#\w+)|'  # Hashtags
    r'(@\w+)|'  # Mentions
    r'(\w+)|'  # Words
    r'([^\w\s])'  # Punctuation
)

def _tokenize(text: str) -> List[str]:
    """Fast tokenization using pre-compiled regex pattern."""
    return [match for match in _TOKEN_PATTERN.findall(text) if any(match)]

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

def emoticon_fix(input_string: str) -> str:
    """Transform emoticons in a string to their corresponding meanings."""
    if not isinstance(input_string, str):
        raise TypeError("Input must be a string")
    
    tokens = [match[0] or match[1] or match[2] or match[3] or match[4] or match[5] 
              for match in _TOKEN_PATTERN.finditer(input_string)]
    
    return _rebuild_with_spacing(tokens, lambda t: EMOTICON_DICT.get(t, t))

def remove_emoticons(input_string: str) -> str:
    """Remove all emoticons from the input string."""
    if not isinstance(input_string, str):
        raise TypeError("Input must be a string")
    
    tokens = [match[0] or match[1] or match[2] or match[3] or match[4] or match[5] 
              for match in _TOKEN_PATTERN.finditer(input_string)]
    filtered_tokens = [t for t in tokens if t not in EMOTICON_DICT]
    
    return _rebuild_with_spacing(filtered_tokens)

def replace_emoticons(input_string: str, tag_format: str = "__EMO_{tag}__") -> str:
    """Replace emoticons with customizable NER-friendly tags."""
    if not isinstance(input_string, str):
        raise TypeError("Input must be a string")
    
    if "{tag}" not in tag_format:
        raise ValueError("tag_format must contain the {tag} placeholder")
    
    tokens = [match[0] or match[1] or match[2] or match[3] or match[4] or match[5] 
              for match in _TOKEN_PATTERN.finditer(input_string)]
    
    def transform(token):
        if token in EMOTICON_DICT:
            return tag_format.format(tag=EMOTICON_DICT[token])
        return token
    
    return _rebuild_with_spacing(tokens, transform)