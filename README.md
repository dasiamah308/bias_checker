# Bias Checker

A comprehensive bias detection and fact-checking system that analyzes political bias in news articles using both content analysis and source bias information from AllSides and Ad Fontes Media.

## Features

- **Content Analysis**: Analyzes article text for bias indicators using machine learning
- **Source Bias Integration**: Incorporates bias ratings from AllSides and Ad Fontes Media
- **Reliability Scoring**: Uses factual accuracy ratings from Ad Fontes
- **Web Scraping**: Extracts text content from URLs for analysis
- **Fact Checking**: Integrates with Google Fact Check API
- **Confidence Scoring**: Provides prediction confidence for each bias class

## Project Structure

```
bias_checker/
├── src/                          # Source code
│   ├── core/                     # Core functionality
│   │   ├── classifier_simple.py  # Enhanced bias classifier
│   │   ├── scrapper.py          # Web scraping utilities
│   │   ├── fact_checker.py      # Google Fact Check integration
│   │   └── classifier.py        # Basic classifier
│   ├── utils/                    # Utility functions
│   │   └── bias_utils.py        # Bias feature extraction and normalization
│   └── data/                     # Data sources
│       ├── allsides_bias_lookup.py    # AllSides bias ratings
│       └── adfontes_bias_lookup.py    # Ad Fontes bias ratings
├── training/                     # Training scripts
│   ├── train_model_simple.py    # Simple training pipeline
│   └── train_model_enhanced.py  # Advanced training pipeline
├── tests/                        # Test files
│   ├── test_bias_utils.py       # Bias utilities tests
│   ├── test_scrapper.py         # Scraper tests
│   └── test_fact_checker.py     # Fact checker tests
├── models/                       # Trained models
├── main.py                      # Command-line interface
├── app.py                       # Web application (FastAPI)
└── requirements.txt             # Python dependencies
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd bias_checker
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file with your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage

### Command Line Interface

**Analyze text content**:
```bash
python main.py --text "Government announces new climate policy" --source "https://nytimes.com/article"
```

**Analyze URL directly**:
```bash
python main.py --url "https://example.com/article"
```

**Verbose output**:
```bash
python main.py --text "Article text" --source "https://foxnews.com/article" --verbose
```

### Python API

```python
from src.core import SimpleBiasClassifier

# Initialize classifier
classifier = SimpleBiasClassifier()

# Analyze text with source URL
result = classifier.predict_bias(
    text="Government announces new climate policy...",
    source_url="https://nytimes.com/article"
)

print(f"Bias: {result['bias_label']}")
print(f"Confidence: {result['confidence']}")
print(f"Source Bias: {result['source_bias_info']['consensus_bias']}")
```

### Web Application

Start the FastAPI web server:
```bash
python app.py
```

Visit `http://localhost:8000` for the web interface.

## Training

### Train the Model

1. **Prepare your dataset** (optional):
   Create a `labeled_articles.csv` file with columns:
   - `text`: Article content
   - `label`: Bias label (Far Left, Lean Left, Center, Lean Right, Far Right)
   - `source_url`: Source URL for bias lookup

2. **Run training**:
   ```bash
   cd training
   python train_model_simple.py
   ```

The training script will:
- Extract bias features from source URLs
- Combine text and bias features
- Train a logistic regression model
- Save the model and components to `models/`

## Testing

Run all tests:
```bash
python -m unittest discover -s tests
```

Run specific test files:
```bash
python -m unittest tests.test_bias_utils -v
python -m unittest tests.test_scrapper -v
python -m unittest tests.test_fact_checker -v
```

## Bias Sources

### AllSides Media
- **Scale**: -6 to +6
- **Categories**: Left, Lean Left, Center, Lean Right, Right, Far Right
- **Source**: https://www.allsides.com/

### Ad Fontes Media
- **Scale**: -42 to +42
- **Categories**: Skews Left, Strong Left, Middle, Skews Right, Strong Right
- **Reliability**: 0-64 scale for factual accuracy
- **Source**: https://adfontesmedia.com/

## Model Features

The enhanced model uses:

**Text Features**:
- TF-IDF vectorization (5000 features)
- N-gram analysis (1-2 grams)
- Stop word removal

**Bias Features**:
- AllSides bias score (normalized)
- Ad Fontes bias score
- Reliability score
- Consensus bias determination
- Agreement level between sources

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **AllSides Media** for bias ratings
- **Ad Fontes Media** for bias and reliability ratings
- **Google Fact Check API** for fact-checking capabilities