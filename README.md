# emoticon_fix
A Python package to transform emoticon in a string to text, e.g., :) => Smile.

## What are emoticons?
An emoticon short for "emotion icon", also known simply as an emote, is a pictorial representation of a facial expression using characters—usually punctuation marks, numbers, and letters—to express a person's feelings or mood, or as a time-saving method. The first ASCII emoticons, :-) and :-(, were written by Scott Fahlman in 1982, but emoticons actually originated on the PLATO IV computer system in 1972.

## Why do we need to transform emoticon to text?
When we preprocess the text for the NLP model, if we just remove all the punctuations in the text, emoticons will leave some meaningless letters. For example, ":D" (laugh) will become just "D". Obviously, these meaningless letters will affect the performance of the NLP model.

## Installation
```
pip install emoticon-fix
```

## Example
```python
from emoticon_fix import emoticon_fix

text = 'test :) test :D test'
emoticon_fix.emoticon_fix(text)
```
The output will be:
```
'test Smile test Laugh test'
```

## PyPI Link
https://pypi.org/project/emoticon-fix/
