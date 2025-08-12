# Emoticon Fix

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
- [Sentiment Analysis](#sentiment-analysis)
- [Analytics & Statistics](#analytics--statistics)
- [Data Export](#data-export)
- [Text Preprocessing Pipeline](#text-preprocessing-pipeline)
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

### New: Sentiment Analysis

```python
from emoticon_fix import analyze_sentiment, get_sentiment_score, classify_sentiment

# Analyze sentiment of emoticons in text
text = "Having a great day :) :D!"
analysis = analyze_sentiment(text)
print(f"Sentiment: {analysis.classification}")  # "Very Positive"
print(f"Score: {analysis.average_score:.3f}")   # "0.800"

# Get just the sentiment score (-1.0 to 1.0)
score = get_sentiment_score("Happy :) but sad :(")
print(score)  # 0.05 (slightly positive)

# Get sentiment classification
classification = classify_sentiment("Love this (｡♥‿♥｡) so much!")
print(classification)  # "Very Positive"
```

### New: Analytics & Statistics

The analytics extension provides comprehensive emoticon usage analysis:

```python
from emoticon_fix import (
    get_emoticon_statistics, 
    create_emotion_profile, 
    compare_emotion_profiles,
    get_emoticon_trends
)

# Get detailed statistics about emoticon usage
text = "Happy :) very :) extremely :D and sometimes sad :("
stats = get_emoticon_statistics(text)

print(f"Total emoticons: {stats.total_emoticons}")        # 4
print(f"Unique emoticons: {stats.unique_emoticons}")      # 3
print(f"Dominant emotion: {stats.dominant_emotion}")      # "Smile"
print(f"Average sentiment: {stats.average_sentiment:.3f}") # 0.525
print(f"Emoticon density: {stats.get_emoticon_density():.1f}%") # per 100 chars

# Get top emoticons and emotions
print("Top emoticons:", stats.get_top_emoticons(3))
print("Top emotions:", stats.get_top_emotions(3))
```

### Emotion Profiling

Create comprehensive emotion profiles for users or text collections:

```python
# Create emotion profile from multiple texts
texts = [
    "Having a great day :) :D",
    "Feeling sad today :(",
    "Mixed emotions :) but also :/ sometimes",
    "Super excited! :D :D (｡♥‿♥｡)"
]

profile = create_emotion_profile(texts, "User Profile")

print(f"Profile: {profile.name}")
print(f"Texts analyzed: {profile.texts_analyzed}")
print(f"Total emoticons: {profile.total_emoticons}")
print(f"Overall sentiment: {profile.get_overall_sentiment():.3f}")
print(f"Emotion diversity: {profile.get_emotion_diversity():.3f}")
print(f"Sentiment consistency: {profile.get_sentiment_consistency():.3f}")

# Get dominant emotions across all texts
dominant_emotions = profile.get_dominant_emotions(5)
print("Dominant emotions:", dominant_emotions)
```

### Profile Comparison

Compare emotion patterns between different users or text collections:

```python
# Create multiple profiles
happy_user = create_emotion_profile([
    "Great day :D", "So happy :)", "Love this! (｡♥‿♥｡)"
], "Happy User")

sad_user = create_emotion_profile([
    "Feeling down :(", "Bad day :(", "Not good :("
], "Sad User")

mixed_user = create_emotion_profile([
    "Happy :) but worried :(", "Good :) and bad :(", "Mixed feelings :/ :)"
], "Mixed User")

# Compare profiles
comparison = compare_emotion_profiles([happy_user, sad_user, mixed_user])

print(f"Profiles compared: {comparison['profiles_compared']}")
print("Sentiment range:", comparison['overall_comparison']['sentiment_range'])
print("Diversity range:", comparison['overall_comparison']['diversity_range'])

# Individual profile summaries
for profile in comparison['profile_summaries']:
    print(f"{profile['name']}: sentiment={profile['overall_sentiment']:.3f}")
```

### Trend Analysis

Analyze emoticon trends across multiple texts or time periods:

```python
# Analyze trends across multiple texts
texts = [
    "Day 1: Excited to start :D",
    "Day 2: Going well :)",
    "Day 3: Some challenges :/",
    "Day 4: Feeling better :)",
    "Day 5: Great finish :D :D"
]

labels = [f"Day {i+1}" for i in range(len(texts))]
trends = get_emoticon_trends(texts, labels)

print(f"Total texts analyzed: {trends['total_texts']}")
print("Sentiment trend:", trends['trend_summary']['sentiment_trend'])
print("Average sentiment:", trends['trend_summary']['average_sentiment_across_texts'])

# Most common emotions across all texts
print("Most common emotions:", trends['trend_summary']['most_common_emotions'])
```

## Sentiment Analysis

The sentiment analysis extension provides powerful emotion detection capabilities:

### Features

- **Sentiment Scoring**: Get numerical sentiment scores (-1.0 to 1.0)
- **Classification**: Automatic categorization (Very Positive, Positive, Neutral, Negative, Very Negative)
- **Emotion Extraction**: Extract individual emoticons with their emotions and scores
- **Batch Processing**: Analyze multiple texts efficiently
- **Detailed Analysis**: Get comprehensive sentiment reports

### Advanced Usage

```python
from emoticon_fix import analyze_sentiment, extract_emotions, batch_analyze

# Detailed sentiment analysis
text = "Mixed feelings :) but also :( about this"
analysis = analyze_sentiment(text)
print(analysis.summary())

# Extract individual emotions
emotions = extract_emotions("Happy :) but worried :(")
for emoticon, emotion, score in emotions:
    print(f"'{emoticon}' → {emotion} (score: {score:.3f})")

# Batch processing
texts = ["Happy :)", "Sad :(", "Excited :D"]
results = batch_analyze(texts)
```

### Sentiment Scoring System

- **Very Positive (0.8-1.0)**: Love, Very Happy, Excited, Dancing Joy
- **Positive (0.3-0.7)**: Smile, Happy, Wink, Hug, Kiss
- **Neutral (0.0-0.2)**: Neutral, Tongue, Surprised, Confused
- **Negative (-0.2 to -0.7)**: Sad, Crying, Worried, Annoyed
- **Very Negative (-0.8 to -1.0)**: Angry, Rage, Table Flip

## Analytics & Statistics

The analytics extension provides comprehensive emoticon usage analysis capabilities:

### EmoticonStats Features

- **Frequency Analysis**: Count emoticon and emotion occurrences
- **Sentiment Distribution**: Categorize emoticons by sentiment
- **Density Calculation**: Emoticons per 100 characters
- **Position Tracking**: Track emoticon positions in text
- **Top Rankings**: Get most frequent emoticons and emotions

### EmoticonProfile Features

- **Multi-text Analysis**: Aggregate statistics across multiple texts
- **Emotion Diversity**: Measure variety of emotions used
- **Sentiment Consistency**: Measure emotional stability over time
- **Comparative Metrics**: Compare different users or periods

### Advanced Analytics

```python
from emoticon_fix import get_emoticon_statistics, EmoticonProfile

# Detailed emoticon statistics
text = "Super happy :D today! Great mood :) and excited (｡♥‿♥｡) for later!"
stats = get_emoticon_statistics(text)

# Access detailed information
print(f"Emoticon positions: {stats.emoticon_positions}")
print(f"Sentiment distribution: {stats.sentiment_distribution}")
print(f"Top 3 emoticons: {stats.get_top_emoticons(3)}")
print(f"Analysis timestamp: {stats.analysis_timestamp}")

# Create custom profile
profile = EmoticonProfile("Custom Analysis")
profile.add_text("First text :)", "text_1")
profile.add_text("Second text :(", "text_2")
profile.add_text("Third text :D", "text_3")

print(f"Emotion diversity: {profile.get_emotion_diversity():.3f}")
print(f"Sentiment consistency: {profile.get_sentiment_consistency():.3f}")
```

## Data Export

Export analysis results for further processing or visualization:

```python
from emoticon_fix import export_analysis, get_emoticon_statistics, create_emotion_profile

# Export statistics to JSON
text = "Happy :) day with multiple :D emoticons!"
stats = get_emoticon_statistics(text)

# Export to JSON (default format)
json_file = export_analysis(stats, format="json", filename="emoticon_stats.json")
print(f"Exported to: {json_file}")

# Export to CSV
csv_file = export_analysis(stats, format="csv", filename="emoticon_stats.csv")
print(f"Exported to: {csv_file}")

# Export emotion profile
texts = ["Happy :)", "Sad :(", "Excited :D"]
profile = create_emotion_profile(texts, "Sample Profile")
profile_file = export_analysis(profile, format="json", filename="emotion_profile.json")

# Auto-generate filename with timestamp
auto_file = export_analysis(stats)  # Creates: emoticon_analysis_YYYYMMDD_HHMMSS.json
```

### Export Formats

**JSON Export**: Complete data structure with all metrics and metadata
```json
{
  "total_emoticons": 3,
  "unique_emoticons": 2,
  "emoticon_density": 12.5,
  "emoticon_frequency": {":)": 2, ":D": 1},
  "emotion_frequency": {"Smile": 2, "Laugh": 1},
  "sentiment_distribution": {"positive": 3, "negative": 0, "neutral": 0},
  "average_sentiment": 0.8,
  "dominant_emotion": "Smile",
  "analysis_timestamp": "2024-01-15T10:30:00"
}
```

**CSV Export**: Structured tabular format for spreadsheet analysis
- Emoticon statistics with frequencies
- Emotion breakdowns
- Sentiment distributions
- Compatible with Excel, Google Sheets, etc.

## Text Preprocessing Pipeline

The text preprocessing pipeline feature allows you to chain multiple emoticon processing operations together in a configurable, reusable workflow. This is particularly useful for NLP workflows where you need consistent text preprocessing across multiple texts or datasets.

### Basic Pipeline Usage

```python
from emoticon_fix import TextPreprocessingPipeline

# Create a custom pipeline
pipeline = (TextPreprocessingPipeline("MyPipeline")
           .add_text_cleaning(normalize_whitespace=True, remove_extra_punctuation=True)
           .add_emoticon_fix()
           .add_sentiment_analysis())

# Process text
text = "Hello   world!!!   :)   Great   day   :D"
result = pipeline.process(text)
print(result)  # Output: "Hello world! Smile Great day Laugh"

# Process with metadata
result, metadata = pipeline.process(text, collect_metadata=True)
print(f"Processed: {result}")
print(f"Pipeline: {metadata['pipeline']}")
print(f"Steps executed: {len(metadata['steps'])}")
```

### Pipeline Steps

The pipeline supports various built-in steps:

#### Text Cleaning Step
```python
from emoticon_fix import TextPreprocessingPipeline

pipeline = TextPreprocessingPipeline("Cleaning")
pipeline.add_text_cleaning(
    normalize_whitespace=True,     # Replace multiple spaces with single space
    remove_extra_punctuation=True  # Replace repeated punctuation with single
)

text = "Text   with!!!   extra   spaces???"
result = pipeline.process(text)  # "Text with! extra spaces?"
```

#### Emoticon Processing Steps
```python
from emoticon_fix import TextPreprocessingPipeline

# Fix emoticons (convert to text)
pipeline = TextPreprocessingPipeline("Fix").add_emoticon_fix()

# Remove emoticons completely
pipeline = TextPreprocessingPipeline("Remove").add_remove_emoticons()

# Replace emoticons with NER tags
pipeline = TextPreprocessingPipeline("Tag").add_replace_emoticons("__EMO_{tag}__")

# Sentiment analysis (doesn't modify text, stores results in context)
pipeline = TextPreprocessingPipeline("Analysis").add_sentiment_analysis()
```

#### Custom Steps
```python
from emoticon_fix import TextPreprocessingPipeline

def uppercase_transform(text, prefix=""):
    return prefix + text.upper()

pipeline = (TextPreprocessingPipeline("Custom")
           .add_custom_step("uppercase", uppercase_transform, prefix=">>> "))

result = pipeline.process("hello world")  # ">>> HELLO WORLD"
```

### Prebuilt Pipelines

Use prebuilt pipelines for common use cases:

#### Standard Pipeline
```python
from emoticon_fix import create_standard_pipeline

# Creates: text_cleaning -> emoticon_fix -> sentiment_analysis
pipeline = create_standard_pipeline("StandardProcessing")

text = "Great   day!!!  :D   with   friends  :)"
result = pipeline.process(text)
print(result)  # "Great day! Laugh with friends Smile"
```

#### NER Pipeline
```python
from emoticon_fix import create_ner_pipeline

# Creates: text_cleaning -> replace_emoticons -> sentiment_analysis
# Optimized for Named Entity Recognition tasks
pipeline = create_ner_pipeline("__EMOTION_{tag}__", "NERProcessing")

text = "Happy  :)  customer  feedback"
result = pipeline.process(text)
print(result)  # "Happy __EMOTION_Smile__ customer feedback"
```

#### Analysis Pipeline
```python
from emoticon_fix import create_analysis_pipeline

# Creates: sentiment_analysis (with metadata enabled)
# For analysis without text modification
pipeline = create_analysis_pipeline("AnalysisOnly")

text = "Customer feedback: Great product :D but slow shipping :("
result, metadata = pipeline.process(text)

print(f"Text unchanged: {result}")
print(f"Sentiment: {metadata['context']['sentiment_analysis_result'].classification}")
```

### Advanced Pipeline Features

#### Caching
```python
from emoticon_fix import TextPreprocessingPipeline

pipeline = (TextPreprocessingPipeline("CachedPipeline")
           .add_emoticon_fix()
           .enable_cache())

# First processing (computed)
result1 = pipeline.process("Happy :)")

# Second processing (cached)
result2 = pipeline.process("Happy :)")  # Same result, faster

# Clear cache when needed
pipeline.clear_cache()
```

#### Batch Processing
```python
from emoticon_fix import create_standard_pipeline

pipeline = create_standard_pipeline()

texts = [
    "Happy day :)",
    "Sad news :(",
    "Excited for tomorrow :D"
]

results = pipeline.process_batch(texts)
for result in results:
    print(result)
```

#### Step Management
```python
from emoticon_fix import TextPreprocessingPipeline

pipeline = (TextPreprocessingPipeline("Manageable")
           .add_text_cleaning()
           .add_emoticon_fix()
           .add_sentiment_analysis())

# Get step names
print(pipeline.get_step_names())  # ['text_cleaning', 'emoticon_fix', 'sentiment_analysis']

# Remove a step
pipeline.remove_step("sentiment_analysis")

# Get specific step
cleaning_step = pipeline.get_step("text_cleaning")

# Clone pipeline
new_pipeline = pipeline.clone()
```

#### Metadata Collection
```python
from emoticon_fix import TextPreprocessingPipeline

pipeline = (TextPreprocessingPipeline("MetadataPipeline")
           .add_emoticon_fix()
           .add_sentiment_analysis()
           .enable_metadata_collection())  # Always collect metadata

result, metadata = pipeline.process("Happy :D day!")

print(f"Pipeline: {metadata['pipeline']}")
print(f"Total processing time: {sum(step['processing_time_ms'] for step in metadata['steps']):.2f}ms")

for step in metadata['steps']:
    print(f"Step {step['step']}: {step['processing_time_ms']:.2f}ms")
```

### Pipeline Serialization

```python
from emoticon_fix import TextPreprocessingPipeline

pipeline = (TextPreprocessingPipeline("SaveablePipeline")
           .add_text_cleaning()
           .add_emoticon_fix()
           .enable_cache())

# Export pipeline configuration
config = pipeline.to_dict()
print(config)
# {
#   "name": "SaveablePipeline",
#   "steps": [
#     {"name": "text_cleaning", "type": "TextCleaningStep"},
#     {"name": "emoticon_fix", "type": "EmoticonFixStep"}
#   ],
#   "step_count": 2,
#   "caching_enabled": True,
#   "metadata_enabled": False,
#   "creation_timestamp": "2024-01-15T10:30:00"
# }
```

### Use Case Examples

#### Social Media Content Processing
```python
from emoticon_fix import create_standard_pipeline

# Pipeline for social media posts
social_pipeline = create_standard_pipeline("SocialMedia")

posts = [
    "Just had the best coffee :D #coffeelover",
    "Traffic is terrible today :( #commute", 
    "Weekend vibes!!! :) :) Can't wait!!!"
]

processed_posts = social_pipeline.process_batch(posts)
for original, processed in zip(posts, processed_posts):
    print(f"Original: {original}")
    print(f"Processed: {processed}")
    print()
```

#### Customer Feedback Analysis
```python
from emoticon_fix import TextPreprocessingPipeline

# Pipeline for customer feedback
feedback_pipeline = (TextPreprocessingPipeline("CustomerFeedback")
                    .add_text_cleaning(normalize_whitespace=True, remove_extra_punctuation=True)
                    .add_replace_emoticons("__SENTIMENT_{tag}__")
                    .add_sentiment_analysis()
                    .enable_metadata_collection())

feedback = "Product quality is amazing!!! :D But shipping was slow... :("
result, metadata = feedback_pipeline.process(feedback)

print(f"Processed feedback: {result}")
sentiment_analysis = metadata['context']['sentiment_analysis_result']
print(f"Overall sentiment: {sentiment_analysis.classification}")
print(f"Sentiment score: {sentiment_analysis.average_score:.3f}")
```

#### Research Data Preprocessing
```python
from emoticon_fix import TextPreprocessingPipeline

# Pipeline for research data
research_pipeline = (TextPreprocessingPipeline("Research")
                    .add_text_cleaning(normalize_whitespace=True)
                    .add_emoticon_fix()
                    .add_sentiment_analysis()
                    .enable_cache()  # Cache for repeated analysis
                    .enable_metadata_collection())

# Process large dataset
research_texts = ["Text with :) emotions" for _ in range(1000)]
results = research_pipeline.process_batch(research_texts)

print(f"Processed {len(results)} texts with caching enabled")
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

### Social Media Analysis Example
```python
from emoticon_fix import create_emotion_profile, compare_emotion_profiles, export_analysis

# Analyze social media posts from different users
user1_posts = [
    "Amazing product! :D Love it!",
    "Great customer service :)",
    "Highly recommended! (｡♥‿♥｡)"
]

user2_posts = [
    "Product was okay :/",
    "Shipping was slow :(",
    "Could be better... :/"
]

user3_posts = [
    "Mixed experience :) good product but :( bad delivery",
    "Happy with purchase :) but upset about delay :(",
    "Overall satisfied :) despite issues :/"
]

# Create emotion profiles
user1_profile = create_emotion_profile(user1_posts, "Satisfied Customer")
user2_profile = create_emotion_profile(user2_posts, "Dissatisfied Customer")
user3_profile = create_emotion_profile(user3_posts, "Mixed Customer")

# Compare profiles
comparison = compare_emotion_profiles([user1_profile, user2_profile, user3_profile])

# Export results
export_analysis(comparison, format="json", filename="customer_sentiment_analysis.json")

print("Customer sentiment analysis completed!")
print(f"Satisfied customer sentiment: {user1_profile.get_overall_sentiment():.3f}")
print(f"Dissatisfied customer sentiment: {user2_profile.get_overall_sentiment():.3f}")
print(f"Mixed customer sentiment: {user3_profile.get_overall_sentiment():.3f}")
```

### Time Series Analysis Example
```python
from emoticon_fix import get_emoticon_trends, export_analysis

# Analyze emotional progression over time
weekly_posts = [
    "Week 1: Starting new job :) excited!",
    "Week 2: Learning lots :D challenging but fun!",
    "Week 3: Feeling overwhelmed :( too much work",
    "Week 4: Getting better :) finding my rhythm",
    "Week 5: Confident now :D loving the work!",
    "Week 6: Stress again :( big project deadline",
    "Week 7: Relief! :D Project completed successfully!",
    "Week 8: Balanced now :) happy with progress"
]

week_labels = [f"Week {i+1}" for i in range(len(weekly_posts))]
trends = get_emoticon_trends(weekly_posts, week_labels)

# Export trend analysis
export_analysis(trends, format="json", filename="emotional_journey.json")

print("Emotional journey analysis:")
sentiment_trend = trends['trend_summary']['sentiment_trend']
for i, sentiment in enumerate(sentiment_trend):
    print(f"Week {i+1}: {sentiment:.3f}")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Testing

The package includes a comprehensive test suite. To run the tests:

```bash
pip install -e ".[dev]"
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.