# Quick Usage Guide

This guide shows you how to use the Bias Checker quickly and easily.

## 🚀 Quick Start

### 1. Install and Setup
```bash
# Activate virtual environment
venv\Scripts\activate

# Install dependencies (if not done already)
pip install -r requirements.txt

# Train the model (first time only)
cd training
python train_model_simple.py
cd ..
```

### 2. Basic Usage

**Analyze text with source URL:**
```bash
python main.py --text "Government announces new climate policy to reduce emissions by 50% by 2030." --source "https://nytimes.com/article"
```

**Analyze a URL directly:**
```bash
python main.py --url "https://example.com/article"
```

**Get detailed results:**
```bash
python main.py --text "Your article text here" --source "https://foxnews.com/article" --verbose
```

## 📝 Common Examples

### Example 1: Analyze NYTimes Article
```bash
python main.py --text "The Biden administration today announced sweeping new regulations..." --source "https://nytimes.com/politics/article"
```

**Expected Output:**
```
=== Bias Analysis Results ===
Predicted Bias: Lean Left
Confidence: 73.6%
```

### Example 2: Analyze Fox News Article
```bash
python main.py --text "Senator Johnson criticized the administration's spending policies..." --source "https://foxnews.com/politics/article"
```

**Expected Output:**
```
=== Bias Analysis Results ===
Predicted Bias: Lean Right
Confidence: 82.1%
```

### Example 3: Analyze Reuters Article
```bash
python main.py --text "Economic data shows mixed signals for the upcoming quarter..." --source "https://reuters.com/business/article"
```

**Expected Output:**
```
=== Bias Analysis Results ===
Predicted Bias: Center
Confidence: 91.5%
```

### Example 4: Analyze Far Left Source
```bash
python main.py --text "Revolutionary socialist movement calls for complete economic system overhaul..." --source "https://jacobinmag.com/article"
```

**Expected Output:**
```
=== Bias Analysis Results ===
Predicted Bias: Far Left
Confidence: 85.2%
```

### Example 5: Analyze Far Right Source
```bash
python main.py --text "Ultra-conservative group calls for complete government shutdown..." --source "https://breitbart.com/article"
```

**Expected Output:**
```
=== Bias Analysis Results ===
Predicted Bias: Far Right
Confidence: 78.9%
```

## 🔧 Python API Usage

### Simple Text Analysis
```python
from src.core import SimpleBiasClassifier

# Initialize classifier
classifier = SimpleBiasClassifier()

# Analyze text
result = classifier.predict_bias(
    text="Your article text here",
    source_url="https://nytimes.com/article"
)

print(f"Bias: {result['bias_label']}")
print(f"Confidence: {result['confidence']}")
```

### URL Analysis
```python
# Analyze URL directly
result = classifier.predict_bias_from_url("https://example.com/article")

if 'error' not in result:
    print(f"Bias: {result['bias_label']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Text Preview: {result['extracted_text'][:100]}...")
else:
    print(f"Error: {result['error']}")
```

### Detailed Analysis
```python
result = classifier.predict_bias(text, source_url)

# Get all details
print(f"Predicted Bias: {result['bias_label']}")
print(f"Confidence: {result['confidence']}")
print(f"Source Bias: {result['source_bias_info']['consensus_bias']}")
print(f"Reliability Score: {result['source_bias_info']['reliability_score']}")

# Confidence for each bias class
for bias, conf in result['confidence_by_class'].items():
    print(f"{bias}: {conf:.1%}")
```

## 🎯 Use Cases

### 1. News Article Analysis
```bash
# Analyze a news article you're reading
python main.py --url "https://news-website.com/article" --verbose
```

### 2. Social Media Content
```bash
# Analyze text from social media
python main.py --text "Your social media post text here" --verbose
```

### 3. Research Papers
```bash
# Analyze academic or research content
python main.py --text "Research paper abstract or content" --verbose
```

### 4. Batch Analysis
```python
# Analyze multiple articles
articles = [
    ("Article 1 text", "https://source1.com"),
    ("Article 2 text", "https://source2.com"),
    ("Article 3 text", "https://source3.com")
]

classifier = SimpleBiasClassifier()
for text, url in articles:
    result = classifier.predict_bias(text, url)
    print(f"URL: {url} -> Bias: {result['bias_label']} ({result['confidence']:.1%})")
```

## 🔍 Understanding Results

### Bias Labels
- **Far Left**: Radical/extreme liberal bias
- **Lean Left**: Liberal/progressive bias
- **Center**: Neutral/balanced bias  
- **Lean Right**: Conservative bias
- **Far Right**: Radical/extreme conservative bias

### Confidence Levels
- **90%+**: Very confident prediction
- **70-89%**: Confident prediction
- **50-69%**: Moderate confidence
- **<50%**: Low confidence

### Source Bias Information
- **Consensus Bias**: Agreement between AllSides and Ad Fontes
- **Reliability Score**: Factual accuracy rating (0-64)
- **Agreement Level**: How much the sources agree (High/Medium/Low)

## 🛠️ Troubleshooting

### Common Issues

**"Model not found" error:**
```bash
# Run training first
cd training
python train_model_simple.py
cd ..
```

**Import errors:**
```bash
# Make sure virtual environment is activated
venv\Scripts\activate
```

**URL scraping fails:**
```bash
# Try with just text analysis
python main.py --text "Your text here" --source "https://example.com"
```

### Getting Help
```bash
# See all available options
python main.py --help
```

## 📊 Sample Output

```
=== Bias Analysis Results ===
Predicted Bias: Lean Left
Confidence: 73.6%

Detailed Results:
Text Length: 245 characters
Source Domain: nytimes.com
Source Bias: Lean Left
Reliability Score: 41.05
Has Source Bias Data: True

Confidence by Class:
  Far Left: 5.2%
  Lean Left: 73.6%
  Center: 15.8%
  Lean Right: 4.1%
  Far Right: 1.3%
```

## 🎉 That's It!

You're now ready to analyze bias in any text or URL. The system combines:
- **Content analysis** (what the text says)
- **Source bias** (who's saying it)
- **Reliability scoring** (how factual it is)

For more advanced features, check the main README.md file! 