# emoticon_fix

[![PyPI version](https://badge.fury.io/py/emoticon-fix.svg)](https://badge.fury.io/py/emoticon-fix)
[![Python Versions](https://img.shields.io/pypi/pyversions/emoticon-fix.svg)](https://pypi.org/project/emoticon-fix/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Python package to transform emoticons in text to their corresponding meanings, e.g., `:)` => `Smile`. This is particularly useful for NLP preprocessing where emoticons need to be preserved as meaningful text.

## Table of Contents

- [What are emoticons?](#what-are-emoticons)
- [Why transform emoticons to text?](#why-transform-emoticons-to-text)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [Testing](#testing)
- [License](#license)

## What are emoticons?

An emoticon (short for "emotion icon") is a pictorial representation of a facial expression using characters—usually punctuation marks, numbers, and letters—to express a person's feelings or mood. The first ASCII emoticons, `:-)` and `:-(`, were written by Scott Fahlman in 1982, but emoticons actually originated on the PLATO IV computer system in 1972.

## Why transform emoticons to text?

When preprocessing text for NLP models, simply removing punctuation can leave emoticons as meaningless characters. For example, `":D"` (laugh) would become just `"D"`. This can negatively impact model performance. By transforming emoticons to their textual meanings, we preserve the emotional context in a format that's more meaningful for NLP tasks.

## Installation

```bash
pip install emoticon-fix
```

## Usage

```python
from emoticon_fix import emoticon_fix

# Basic usage
text = 'Hello :) World :D'
result = emoticon_fix(text)
print(result)  # Output: 'Hello Smile World Laugh'

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

### Complex Example
```python
from emoticon_fix import emoticon_fix

text = 'Feeling :-) today! But yesterday was :-( and tomorrow might be :-D'
result = emoticon_fix(text)
print(result)  # Output: 'Feeling Smile today! But yesterday was Sad and tomorrow might be Laugh'
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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## PyPI Link

https://pypi.org/project/emoticon-fix/
