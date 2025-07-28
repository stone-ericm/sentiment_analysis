# Sentiment Analysis Project

This project performs sentiment analysis on datasets containing datetimes, products, and English language quotes.

## Features

- **Multiple Sentiment Analysis Methods**: TextBlob, VADER, and Transformers-based models
- **Data Processing**: Handles datetime parsing, text preprocessing, and data cleaning
- **Visualization**: Interactive charts and plots for sentiment trends
- **Product Analysis**: Sentiment analysis grouped by products
- **Time Series Analysis**: Sentiment trends over time

## Project Structure

```
semantic_analysis/
├── data/                   # Data files
│   ├── sample_data.csv     # Sample dataset
│   └── processed/          # Processed data outputs
├── src/                    # Source code
│   ├── data_processor.py   # Data loading and preprocessing
│   ├── sentiment_analyzer.py # Sentiment analysis implementations
│   ├── visualizer.py       # Visualization functions
│   └── main.py            # Main execution script
├── notebooks/              # Jupyter notebooks for exploration
├── results/                # Analysis results and outputs
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download NLTK data** (run once):
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
   ```

3. **Download spaCy model** (optional, for advanced text processing):
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

### Basic Usage

```python
from src.main import run_sentiment_analysis

# Run analysis on your dataset
results = run_sentiment_analysis('data/your_data.csv')
```

### Data Format

Your CSV file should have the following columns:
- `datetime`: Date and time of the quote (format: YYYY-MM-DD HH:MM:SS)
- `product`: Product name or identifier
- `quote`: English language text to analyze

Example:
```csv
datetime,product,quote
2024-01-15 10:30:00,Product A,"This product is amazing and works perfectly!"
2024-01-15 11:45:00,Product B,"I'm not satisfied with the quality."
```

## Sentiment Analysis Methods

1. **TextBlob**: Rule-based sentiment analysis with polarity and subjectivity scores
2. **VADER**: Lexicon-based sentiment analysis optimized for social media text
3. **Transformers**: Advanced deep learning models (BERT, RoBERTa) for more accurate analysis

## Output

The analysis generates:
- Sentiment scores for each quote
- Product-wise sentiment summaries
- Time-based sentiment trends
- Interactive visualizations
- Detailed reports in the `results/` directory

## Customization

You can customize the analysis by:
- Modifying preprocessing steps in `data_processor.py`
- Adding new sentiment analysis methods in `sentiment_analyzer.py`
- Creating custom visualizations in `visualizer.py`
- Adjusting parameters in `main.py` 