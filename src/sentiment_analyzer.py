"""
Sentiment analysis module with multiple analysis methods.
Includes TextBlob, VADER, and Transformers-based sentiment analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

# Sentiment analysis libraries
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not available. Install with: pip install textblob")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logging.warning("VADER not available. Install with: pip install vaderSentiment")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Install with: pip install transformers torch")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Main sentiment analyzer class with multiple analysis methods."""
    
    def __init__(self, methods: Optional[List[str]] = None):
        """
        Initialize the sentiment analyzer.
        
        Args:
            methods (Optional[List[str]]): List of methods to use. 
                Options: ['textblob', 'vader', 'transformers']
        """
        self.methods = methods or ['textblob', 'vader']
        self.analyzers = {}
        self.results = {}
        
        # Initialize analyzers
        self._initialize_analyzers()
    
    def _initialize_analyzers(self):
        """Initialize sentiment analysis models."""
        logger.info("Initializing sentiment analyzers...")
        
        # TextBlob
        if 'textblob' in self.methods and TEXTBLOB_AVAILABLE:
            self.analyzers['textblob'] = TextBlobAnalyzer()
            logger.info("TextBlob analyzer initialized")
        
        # VADER
        if 'vader' in self.methods and VADER_AVAILABLE:
            self.analyzers['vader'] = VADERAnalyzer()
            logger.info("VADER analyzer initialized")
        
        # Transformers
        if 'transformers' in self.methods and TRANSFORMERS_AVAILABLE:
            try:
                self.analyzers['transformers'] = TransformersAnalyzer()
                logger.info("Transformers analyzer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Transformers analyzer: {e}")
                self.methods.remove('transformers')
        
        if not self.analyzers:
            raise ValueError("No sentiment analyzers could be initialized. Check your installations.")
    
    def analyze_text(self, text: str) -> Dict[str, Dict]:
        """
        Analyze sentiment of a single text using all available methods.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, Dict]: Results from each method
        """
        results = {}
        
        for method_name, analyzer in self.analyzers.items():
            try:
                results[method_name] = analyzer.analyze(text)
            except Exception as e:
                logger.error(f"Error in {method_name} analysis: {e}")
                results[method_name] = {'error': str(e)}
        
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'quote_cleaned') -> pd.DataFrame:
        """
        Analyze sentiment for all texts in a dataframe.
        
        Args:
            df (pd.DataFrame): DataFrame containing texts
            text_column (str): Name of the column containing text to analyze
            
        Returns:
            pd.DataFrame: Original dataframe with sentiment scores added
        """
        logger.info(f"Starting sentiment analysis on {len(df)} texts...")
        
        # Create a copy to avoid modifying original data
        result_df = df.copy()
        
        # Analyze each text
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing sentiments"):
            text = row[text_column]
            
            if pd.isna(text) or text == '':
                continue
            
            # Get sentiment analysis results
            sentiment_results = self.analyze_text(text)
            
            # Add results to dataframe
            for method_name, results in sentiment_results.items():
                if 'error' not in results:
                    for key, value in results.items():
                        column_name = f"{method_name}_{key}"
                        result_df.at[idx, column_name] = value
        
        # Fill NaN values with 0 for numeric columns
        sentiment_columns = [col for col in result_df.columns if any(method in col for method in self.methods)]
        for col in sentiment_columns:
            if result_df[col].dtype in ['float64', 'int64']:
                result_df[col] = result_df[col].fillna(0)
        
        self.results = result_df
        logger.info("Sentiment analysis completed")
        
        return result_df
    
    def get_sentiment_summary(self) -> Dict:
        """
        Get summary statistics for sentiment analysis results.
        
        Returns:
            Dict: Summary statistics
        """
        if self.results is None or len(self.results) == 0:
            raise ValueError("No analysis results available. Run analyze_dataframe first.")
        
        summary = {}
        
        for method in self.methods:
            if method not in self.analyzers:
                continue
            
            method_summary = {}
            
            # Get sentiment score columns for this method
            sentiment_columns = [col for col in self.results.columns if col.startswith(f"{method}_")]
            
            for col in sentiment_columns:
                if self.results[col].dtype in ['float64', 'int64']:
                    method_summary[col] = {
                        'mean': self.results[col].mean(),
                        'std': self.results[col].std(),
                        'min': self.results[col].min(),
                        'max': self.results[col].max(),
                        'median': self.results[col].median()
                    }
            
            summary[method] = method_summary
        
        return summary
    
    def classify_sentiment(self, score: float, method: str = 'vader') -> str:
        """
        Classify sentiment based on score.
        
        Args:
            score (float): Sentiment score
            method (str): Method used for analysis
            
        Returns:
            str: Sentiment classification
        """
        if method == 'vader':
            if score >= 0.05:
                return 'positive'
            elif score <= -0.05:
                return 'negative'
            else:
                return 'neutral'
        elif method == 'textblob':
            if score > 0:
                return 'positive'
            elif score < 0:
                return 'negative'
            else:
                return 'neutral'
        else:
            # Default classification
            if score > 0.1:
                return 'positive'
            elif score < -0.1:
                return 'negative'
            else:
                return 'neutral'


class TextBlobAnalyzer:
    """TextBlob-based sentiment analyzer."""
    
    def __init__(self):
        """Initialize TextBlob analyzer."""
        pass
    
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Polarity and subjectivity scores
        """
        blob = TextBlob(text)
        
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }


class VADERAnalyzer:
    """VADER-based sentiment analyzer."""
    
    def __init__(self):
        """Initialize VADER analyzer."""
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Compound, positive, negative, and neutral scores
        """
        scores = self.analyzer.polarity_scores(text)
        
        return {
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        }


class TransformersAnalyzer:
    """Transformers-based sentiment analyzer using pre-trained models."""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initialize Transformers analyzer.
        
        Args:
            model_name (str): Name of the pre-trained model to use
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        
        # Map model labels to sentiment scores
        self.label_mapping = {
            'LABEL_0': 'negative',
            'LABEL_1': 'neutral', 
            'LABEL_2': 'positive'
        }
    
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using Transformers.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Sentiment scores
        """
        # Truncate text if too long (most models have token limits)
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
        
        result = self.pipeline(text)[0]
        
        # Convert label to sentiment score
        label = result['label']
        score = result['score']
        
        # Map to sentiment scores
        if label == 'LABEL_0':  # Negative
            sentiment_score = -score
        elif label == 'LABEL_2':  # Positive
            sentiment_score = score
        else:  # Neutral
            sentiment_score = 0
        
        return {
            'sentiment_score': sentiment_score,
            'confidence': score,
            'label': self.label_mapping.get(label, label)
        }


def run_sentiment_analysis(data: pd.DataFrame, methods: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to run sentiment analysis on a dataframe.
    
    Args:
        data (pd.DataFrame): DataFrame with text data
        methods (Optional[List[str]]): Sentiment analysis methods to use
        
    Returns:
        Tuple[pd.DataFrame, Dict]: Results dataframe and summary
    """
    analyzer = SentimentAnalyzer(methods=methods)
    results = analyzer.analyze_dataframe(data)
    summary = analyzer.get_sentiment_summary()
    
    return results, summary 