# Sentiment Analysis

Compare TextBlob, VADER, and Transformer-based models on the same dataset to see how different sentiment analysis methods interpret product reviews over time.

## What It Does

Given a CSV with datetimes, product identifiers, and English-language text, this project runs three independent sentiment analysis methods against every record and produces side-by-side comparisons. It generates per-product summaries, time-series trend lines, word clouds, and an interactive HTML dashboard so you can evaluate which method best fits your data.

## Methods

**TextBlob** -- A rule-based approach that returns polarity (negative to positive) and subjectivity scores. Fast and simple, but treats every domain the same way.

**VADER** -- A lexicon and rule-based tool built specifically for short, informal text. It handles punctuation emphasis, capitalization, and degree modifiers (e.g., "very good" vs. "good"). Generally the best choice for social media and review data.

**Transformers (BERT/RoBERTa)** -- Pre-trained deep learning models fine-tuned on sentiment tasks. Slower and more resource-intensive, but captures context and nuance that lexicon methods miss.

## Output

All results are written to the `results/` directory:

- `interactive_dashboard_vader.html` -- Plotly dashboard with filterable charts
- `sentiment_distribution_vader.png` -- Histogram of sentiment score distribution
- `sentiment_by_product_vader.png` -- Per-product sentiment comparison
- `sentiment_timeline_vader.png` -- Sentiment trends over time
- `wordcloud_positive.png` / `wordcloud_negative.png` -- Most frequent terms by polarity
- `sentiment_analysis_report_*.txt` -- Full text report
- `sentiment_analysis_results_*.csv` -- Raw scored data
- `*_summary.json` -- Aggregated statistics

## Getting Started

```bash
git clone https://github.com/stone-ericm/sentiment_analysis.git
cd sentiment_analysis
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download NLTK data (once):

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

Run the analysis:

```bash
python src/main.py data/sample_data.csv
python src/main.py data/sample_data.csv --methods vader textblob
python src/main.py data/sample_data.csv --methods vader --viz-method vader
```

Input CSV format:

```
datetime,product,quote
2024-01-15 10:30:00,Widget A,"This product is amazing and works perfectly!"
```

Or explore interactively with `jupyter lab` and open `notebooks/sentiment_analysis_exploration.ipynb`.

## Project Structure

```
sentiment_analysis/
  src/
    main.py                 Entry point and CLI
    data_processor.py       Loading, cleaning, datetime parsing
    sentiment_analyzer.py   TextBlob, VADER, and Transformer wrappers
    visualizer.py           Charts, dashboards, word clouds
  notebooks/
    sentiment_analysis_exploration.ipynb
  data/
    sample_data.csv         Example dataset
  results/                  Generated reports and visualizations
  requirements.txt
  setup.py                  Automated environment setup script
```
