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

## Quick Setup (New Device)

### 1. Clone the Repository

```bash
git clone https://github.com/stone-ericm/sentiment_analysis.git
cd sentiment_analysis
```

### 2. Set Up Python Environment

**Option A: Using Virtual Environment (Recommended)**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Option B: Using Conda**
```bash
# Create conda environment
conda create -n sentiment_analysis python=3.9
conda activate sentiment_analysis

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Required Data

```bash
# Download NLTK data (run once)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"

# Download spaCy model (optional, for advanced text processing)
python -m spacy download en_core_web_sm
```

### 4. Prepare Your Data

Create a CSV file in the `data/` directory with your data:

```csv
datetime,product,quote
2024-01-15 10:30:00,Product A,"This product is amazing and works perfectly!"
2024-01-15 11:45:00,Product B,"I'm not satisfied with the quality."
```

Required columns:
- `datetime`: Date and time (format: YYYY-MM-DD HH:MM:SS)
- `product`: Product name or identifier
- `quote`: English language text to analyze

### 5. Run the Analysis

```bash
# Run with sample data
python src/main.py

# Run with your own data
python src/main.py --input data/your_data.csv
```

## Usage

### Basic Usage

```python
from src.main import run_sentiment_analysis

# Run analysis on your dataset
results = run_sentiment_analysis('data/your_data.csv')
```

### Command Line Options

```bash
# Run with default settings
python src/main.py

# Specify input file
python src/main.py --input data/my_data.csv

# Specify output directory
python src/main.py --output results/my_analysis/

# Run with specific sentiment method
python src/main.py --method vader
```

### Jupyter Notebooks

For interactive exploration:
```bash
# Start Jupyter
jupyter lab

# Or for classic Jupyter
jupyter notebook
```

Then open `notebooks/sentiment_analysis_exploration.ipynb`

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

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Make sure you're in the virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

**2. NLTK Data Not Found**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

**3. Memory Issues with Large Datasets**
- Use smaller chunks of data
- Try different sentiment methods (VADER is faster than Transformers)

**4. Missing Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended for large datasets)
- Internet connection for downloading models (first run only)

## Customization

You can customize the analysis by:
- Modifying preprocessing steps in `data_processor.py`
- Adding new sentiment analysis methods in `sentiment_analyzer.py`
- Creating custom visualizations in `visualizer.py`
- Adjusting parameters in `main.py`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License. 