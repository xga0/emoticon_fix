# Emoticon Fix

[![PyPI version](https://img.shields.io/pypi/v/emoticon-fix.svg)](https://pypi.org/project/emoticon-fix/)
[![Python Versions](https://img.shields.io/pypi/pyversions/emoticon-fix.svg)](https://pypi.org/project/emoticon-fix/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight and efficient library for transforming emoticons into their semantic meanings. Perfect for NLP preprocessing where emoticons need to be preserved as meaningful text.

## Quick Start

### Installation
```bash
pip install emoticon-fix
```

### Basic Usage
```python
from emoticon_fix import emoticon_fix

# Transform emoticons to meaningful text
text = 'Hello :) World :D'
result = emoticon_fix(text)
print(result)  # Output: 'Hello Smile World Laugh'

# Works with kaomoji too!
text = 'Feeling (｡♥‿♥｡) today!'
result = emoticon_fix(text)
print(result)  # Output: 'Feeling In Love today!'
```

## Features

- **Core Functionality**
  - Transform emoticons to semantic meanings (`:)` → `Smile`)
  - Support for Western emoticons and Japanese kaomoji
  - Remove or replace emoticons with custom tags

- **Advanced Analytics**
  - Sentiment analysis and scoring
  - Emotion profiling and comparison
  - Statistics and trend analysis
  - Data export (JSON/CSV)

- **Text Processing Pipeline**
  - Configurable, reusable workflows
  - Built-in preprocessing steps
  - Caching and batch processing
  - Performance metrics and metadata

## More Options

```python
from emoticon_fix import remove_emoticons, replace_emoticons

# Remove emoticons completely
clean_text = remove_emoticons('Hello :) World :D')
print(clean_text)  # Output: 'Hello World'

# Replace with NER-friendly tags
tagged_text = replace_emoticons('Hello :) World :D', tag_format="__EMO_{tag}__")
print(tagged_text)  # Output: 'Hello __EMO_Smile__ World __EMO_Laugh__'
```

## Documentation

**[Complete Documentation](README_FULL.md)** - For comprehensive guides, advanced features, and detailed examples, see our full documentation.

### Key Features Covered:
- **Sentiment Analysis** - Advanced emotion detection and scoring
- **Analytics & Statistics** - Comprehensive emoticon usage analysis  
- **Text Processing Pipeline** - Configurable, reusable workflows
- **Data Export** - JSON/CSV export capabilities
- **Real-world Examples** - Social media analysis, customer feedback, and more

## Contributing

Contributions are welcome! Please see the Contributing section in [README_FULL.md](README_FULL.md#contributing) for detailed guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.