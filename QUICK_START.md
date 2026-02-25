# Quick Start Guide

## 🚀 One-Command Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/stone-ericm/sentiment_analysis.git
cd sentiment_analysis

# Run the automated setup
python setup.py
```

## 📋 Manual Setup

If the automated setup doesn't work, follow these steps:

### 1. Clone Repository
```bash
git clone https://github.com/stone-ericm/sentiment_analysis.git
cd sentiment_analysis
```

### 2. Create Virtual Environment
```bash
# macOS/Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Required Data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

### 5. Run Analysis
```bash
# Run with sample data
python src/main.py data/sample_data.csv

# Run with your own data
python src/main.py data/your_data.csv
```

## 📊 Data Format

Your CSV file needs these columns:
```csv
datetime,product,quote
2024-01-15 10:30:00,Product A,"This product is amazing!"
2024-01-15 11:45:00,Product B,"I'm not satisfied."
```

## 🔧 Troubleshooting

**Import Errors**: Make sure virtual environment is activated
**NLTK Errors**: Run the NLTK download command above
**Memory Issues**: Use smaller datasets or VADER method

## 📚 Next Steps

- Read `README.md` for detailed documentation
- Explore `notebooks/` for interactive analysis
- Check `results/` for generated outputs

## 🆘 Need Help?

- Check the troubleshooting section in `README.md`
- Open an issue on GitHub
- Review the code in `src/` directory 