"""
Data processing module for sentiment analysis project.
Handles data loading, cleaning, and preprocessing.
"""

import pandas as pd
import numpy as np
import re
import string
from datetime import datetime
from typing import Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data loading, cleaning, and preprocessing for sentiment analysis."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.data = None
        self.processed_data = None
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            logger.info(f"Loading data from {file_path}")
            self.data = pd.read_csv(file_path)
            logger.info(f"Loaded {len(self.data)} records")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the data has the required columns.
        
        Args:
            data (pd.DataFrame): Data to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        required_columns = ['datetime', 'product', 'quote']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        logger.info("Data validation passed")
        return True
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text for sentiment analysis.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string if not already
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation (optional - can be kept for some sentiment analyzers)
        # text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def parse_datetime(self, datetime_str: str) -> Optional[datetime]:
        """
        Parse datetime string to datetime object.
        
        Args:
            datetime_str (str): Datetime string
            
        Returns:
            Optional[datetime]: Parsed datetime or None if failed
        """
        try:
            return pd.to_datetime(datetime_str)
        except:
            logger.warning(f"Could not parse datetime: {datetime_str}")
            return None
    
    def preprocess_data(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Preprocess the data for sentiment analysis.
        
        Args:
            data (Optional[pd.DataFrame]): Data to preprocess. If None, uses self.data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        if data is None:
            data = self.data
        
        if data is None:
            raise ValueError("No data to preprocess. Load data first.")
        
        logger.info("Starting data preprocessing...")
        
        # Create a copy to avoid modifying original data
        processed_data = data.copy()
        
        # Parse datetime
        processed_data['datetime_parsed'] = processed_data['datetime'].apply(self.parse_datetime)
        
        # Clean text
        processed_data['quote_cleaned'] = processed_data['quote'].apply(self.clean_text)
        
        # Remove rows with empty quotes after cleaning
        initial_count = len(processed_data)
        processed_data = processed_data[processed_data['quote_cleaned'].str.len() > 0]
        final_count = len(processed_data)
        
        if initial_count != final_count:
            logger.info(f"Removed {initial_count - final_count} rows with empty quotes after cleaning")
        
        # Add text length features
        processed_data['quote_length'] = processed_data['quote_cleaned'].str.len()
        processed_data['word_count'] = processed_data['quote_cleaned'].str.split().str.len()
        
        # Add time-based features
        processed_data['hour'] = processed_data['datetime_parsed'].dt.hour
        processed_data['day_of_week'] = processed_data['datetime_parsed'].dt.day_name()
        processed_data['month'] = processed_data['datetime_parsed'].dt.month
        
        # Reset index
        processed_data = processed_data.reset_index(drop=True)
        
        self.processed_data = processed_data
        logger.info(f"Preprocessing complete. Final dataset has {len(processed_data)} records")
        
        return processed_data
    
    def get_data_summary(self) -> dict:
        """
        Get a summary of the processed data.
        
        Returns:
            dict: Summary statistics
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Run preprocess_data first.")
        
        summary = {
            'total_records': len(self.processed_data),
            'unique_products': self.processed_data['product'].nunique(),
            'date_range': {
                'start': self.processed_data['datetime_parsed'].min(),
                'end': self.processed_data['datetime_parsed'].max()
            },
            'text_stats': {
                'avg_length': self.processed_data['quote_length'].mean(),
                'avg_word_count': self.processed_data['word_count'].mean(),
                'min_length': self.processed_data['quote_length'].min(),
                'max_length': self.processed_data['quote_length'].max()
            },
            'products': self.processed_data['product'].value_counts().to_dict()
        }
        
        return summary
    
    def save_processed_data(self, file_path: str) -> None:
        """
        Save processed data to CSV file.
        
        Args:
            file_path (str): Path to save the processed data
        """
        if self.processed_data is None:
            raise ValueError("No processed data to save. Run preprocess_data first.")
        
        try:
            self.processed_data.to_csv(file_path, index=False)
            logger.info(f"Processed data saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise


def load_and_preprocess(file_path: str, save_processed: bool = True) -> Tuple[pd.DataFrame, dict]:
    """
    Convenience function to load and preprocess data in one step.
    
    Args:
        file_path (str): Path to the CSV file
        save_processed (bool): Whether to save processed data
        
    Returns:
        Tuple[pd.DataFrame, dict]: Processed data and summary
    """
    processor = DataProcessor()
    
    # Load data
    data = processor.load_data(file_path)
    
    # Validate data
    if not processor.validate_data(data):
        raise ValueError("Data validation failed")
    
    # Preprocess data
    processed_data = processor.preprocess_data()
    
    # Get summary
    summary = processor.get_data_summary()
    
    # Save processed data if requested
    if save_processed:
        output_path = file_path.replace('.csv', '_processed.csv')
        processor.save_processed_data(output_path)
    
    return processed_data, summary 