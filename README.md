# emoticon_fix

[![PyPI version](https://img.shields.io/pypi/v/emoticon-fix.svg)](https://pypi.org/project/emoticon-fix/)
[![Python Versions](https://img.shields.io/pypi/pyversions/emoticon-fix.svg)](https://pypi.org/project/emoticon-fix/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight and efficient library for transforming emoticons into their semantic meanings. This is particularly useful for NLP preprocessing where emoticons need to be preserved as meaningful text.

## Table of Contents

- [What are emoticons?](#what-are-emoticons)
- [What are kaomoji?](#what-are-kaomoji)
- [Why transform emoticons to text?](#why-transform-emoticons-to-text)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [Testing](#testing)
- [License](#license)

## What are emoticons?

An emoticon (short for "emotion icon") is a pictorial representation of a facial expression using characters—usually punctuation marks, numbers, and letters—to express a person's feelings or mood. The first ASCII emoticons, `:-)` and `:-(`, were written by Scott Fahlman in 1982, but emoticons actually originated on the PLATO IV computer system in 1972.

## What are kaomoji?

Kaomoji (顔文字) are Japanese emoticons that are read horizontally and are more elaborate than traditional Western emoticons. They often use Unicode characters to create more complex expressions and can represent a wider range of emotions and actions. For example, `(｡♥‿♥｡)` represents being in love, and `(ノ°益°)ノ` shows rage. Unlike Western emoticons that you read by tilting your head sideways, kaomoji are meant to be viewed straight on.

emoticon_fix supports a wide variety of kaomoji, making it particularly useful for processing text from Asian social media or any platform where kaomoji are commonly used.

## Why transform emoticons to text?

When preprocessing text for NLP models, simply removing punctuation can leave emoticons and kaomoji as meaningless characters. For example, `:D` (laugh) would become just `D`, and `(｡♥‿♥｡)` (in love) would be completely lost. This can negatively impact model performance. By transforming emoticons and kaomoji to their textual meanings, we preserve the emotional context in a format that's more meaningful for NLP tasks.

## Installation

```bash
pip install emoticon-fix
```

## Usage

```python
from emoticon_fix import emoticon_fix, remove_emoticons, replace_emoticons

# Basic usage - transform emoticons to their meanings
text = 'Hello :) World :D'
result = emoticon_fix(text)
print(result)  # Output: 'Hello Smile World Laugh'

# Remove emoticons completely
stripped_text = remove_emoticons(text)
print(stripped_text)  # Output: 'Hello World'

# Replace with NER-friendly tags (customizable format)
ner_text = replace_emoticons(text, tag_format="__EMO_{tag}__")
print(ner_text)  # Output: 'Hello __EMO_Smile__ World __EMO_Laugh__'

# Works with multiple emoticons
text = 'I am :-) but sometimes :-( and occasionally :-D'
result = emoticon_fix(text)
print(result)  # Output: 'I am Smile but sometimes Sad and occasionally Laugh'
```

## Examples

### Basic Example
```python
from emoticon_fix import emoticon_fix

text = 'test :) test :D test'
result = emoticon_fix(text)
print(result)  # Output: 'test Smile test Laugh test'
```

### Complex Example with Kaomoji
```python
from emoticon_fix import emoticon_fix

text = 'Feeling (｡♥‿♥｡) today! When things go wrong ┗(＾0＾)┓ keep dancing!'
result = emoticon_fix(text)
print(result)  # Output: 'Feeling In Love today! When things go wrong Dancing Joy keep dancing!'
```

### Mixed Emoticons Example
```python
from emoticon_fix import emoticon_fix

text = 'Western :) meets Eastern (◕‿◕✿) style!'
result = emoticon_fix(text)
print(result)  # Output: 'Western Smile meets Eastern Sweet Smile style!'
```

### Removing Emoticons Example
```python
from emoticon_fix import remove_emoticons

text = 'This message :D contains some (｡♥‿♥｡) emoticons that need to be removed!'
result = remove_emoticons(text)
print(result)  # Output: 'This message contains some emoticons that need to be removed!'
```

### NER-Friendly Tagging Example
```python
from emoticon_fix import replace_emoticons

# Default format: __EMO_{tag}__
text = 'Happy customers :) are returning customers!'
result = replace_emoticons(text)
print(result)  # Output: 'Happy customers __EMO_Smile__ are returning customers!'

# Custom format
text = 'User feedback: Product was great :D but shipping was slow :('
result = replace_emoticons(text, tag_format="<EMOTION type='{tag}'>")
print(result)  # Output: 'User feedback: Product was great <EMOTION type='Laugh'> but shipping was slow <EMOTION type='Sad'>'
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Testing

The package includes a test suite. To run the tests:

```bash
pip install -e ".[dev]"
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.