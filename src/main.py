"""
Main execution script for sentiment analysis project.
Orchestrates the entire pipeline from data loading to visualization.
"""

import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime
from typing import Dict, Optional, Tuple

# Import project modules
from data_processor import DataProcessor, load_and_preprocess
from sentiment_analyzer import SentimentAnalyzer
from visualizer import SentimentVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SentimentAnalysisPipeline:
    """Main pipeline class for sentiment analysis."""
    
    def __init__(self, data_path: str, results_dir: str = "results"):
        """
        Initialize the sentiment analysis pipeline.
        
        Args:
            data_path (str): Path to the input data file
            results_dir (str): Directory to save results
        """
        self.data_path = data_path
        self.results_dir = results_dir
        self.data_processor = DataProcessor()
        self.sentiment_analyzer = None
        self.visualizer = SentimentVisualizer(results_dir)
        
        # Results storage
        self.processed_data = None
        self.sentiment_results = None
        self.analysis_summary = None
        
        # Create results directory
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            logger.info(f"Created results directory: {results_dir}")
    
    def load_and_preprocess_data(self, save_processed: bool = True) -> pd.DataFrame:
        """
        Load and preprocess the data.
        
        Args:
            save_processed (bool): Whether to save processed data
            
        Returns:
            pd.DataFrame: Processed data
        """
        logger.info("Starting data loading and preprocessing...")
        
        try:
            # Load and preprocess data
            processed_data, data_summary = load_and_preprocess(
                self.data_path, save_processed=save_processed
            )
            
            self.processed_data = processed_data
            self.analysis_summary = {'data_summary': data_summary}
            
            logger.info("Data preprocessing completed successfully")
            logger.info(f"Processed {len(processed_data)} records")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            raise
    
    def run_sentiment_analysis(self, methods: Optional[list] = None, 
                             text_column: str = 'quote_cleaned') -> pd.DataFrame:
        """
        Run sentiment analysis on the processed data.
        
        Args:
            methods (Optional[list]): List of sentiment analysis methods to use
            text_column (str): Name of the column containing text to analyze
            
        Returns:
            pd.DataFrame: Data with sentiment scores added
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Run load_and_preprocess_data first.")
        
        logger.info("Starting sentiment analysis...")
        
        try:
            # Initialize sentiment analyzer
            self.sentiment_analyzer = SentimentAnalyzer(methods=methods)
            
            # Run analysis
            sentiment_results = self.sentiment_analyzer.analyze_dataframe(
                self.processed_data, text_column=text_column
            )
            
            self.sentiment_results = sentiment_results
            
            # Get sentiment summary
            sentiment_summary = self.sentiment_analyzer.get_sentiment_summary()
            self.analysis_summary['sentiment_summary'] = sentiment_summary
            
            logger.info("Sentiment analysis completed successfully")
            
            return sentiment_results
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            raise
    
    def generate_visualizations(self, method: str = 'vader') -> Dict:
        """
        Generate visualizations for the sentiment analysis results.
        
        Args:
            method (str): Sentiment analysis method to use for visualizations
            
        Returns:
            Dict: Dictionary with paths to generated visualizations
        """
        if self.sentiment_results is None:
            raise ValueError("No sentiment results available. Run run_sentiment_analysis first.")
        
        logger.info("Generating visualizations...")
        
        try:
            visualization_paths = self.visualizer.generate_all_visualizations(
                self.sentiment_results, method=method
            )
            
            self.analysis_summary['visualization_paths'] = visualization_paths
            
            logger.info("Visualization generation completed")
            
            return visualization_paths
            
        except Exception as e:
            logger.error(f"Error in visualization generation: {e}")
            raise
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save all results to files.
        
        Args:
            filename (Optional[str]): Base filename for results
            
        Returns:
            str: Path to the saved results
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_analysis_results_{timestamp}"
        
        logger.info("Saving results...")
        
        try:
            # Save processed data with sentiment scores
            if self.sentiment_results is not None:
                results_path = os.path.join(self.results_dir, f"{filename}.csv")
                self.sentiment_results.to_csv(results_path, index=False)
                logger.info(f"Saved results to {results_path}")
            
            # Save analysis summary
            summary_path = os.path.join(self.results_dir, f"{filename}_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(self.analysis_summary, f, indent=2, default=str)
            logger.info(f"Saved summary to {summary_path}")
            
            return results_path
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def generate_report(self, filename: Optional[str] = None) -> str:
        """
        Generate a comprehensive analysis report.
        
        Args:
            filename (Optional[str]): Base filename for the report
            
        Returns:
            str: Path to the generated report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_analysis_report_{timestamp}"
        
        logger.info("Generating analysis report...")
        
        try:
            report_path = os.path.join(self.results_dir, f"{filename}.txt")
            
            with open(report_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("SENTIMENT ANALYSIS REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Data source: {self.data_path}\n\n")
                
                # Data summary
                if 'data_summary' in self.analysis_summary:
                    f.write("DATA SUMMARY\n")
                    f.write("-" * 40 + "\n")
                    data_summary = self.analysis_summary['data_summary']
                    f.write(f"Total records: {data_summary['total_records']}\n")
                    f.write(f"Unique products: {data_summary['unique_products']}\n")
                    f.write(f"Date range: {data_summary['date_range']['start']} to {data_summary['date_range']['end']}\n")
                    f.write(f"Average text length: {data_summary['text_stats']['avg_length']:.1f} characters\n")
                    f.write(f"Average word count: {data_summary['text_stats']['avg_word_count']:.1f} words\n\n")
                
                # Sentiment summary
                if 'sentiment_summary' in self.analysis_summary:
                    f.write("SENTIMENT ANALYSIS SUMMARY\n")
                    f.write("-" * 40 + "\n")
                    sentiment_summary = self.analysis_summary['sentiment_summary']
                    
                    for method, stats in sentiment_summary.items():
                        f.write(f"\n{method.upper()} Analysis:\n")
                        for metric, values in stats.items():
                            f.write(f"  {metric}:\n")
                            f.write(f"    Mean: {values['mean']:.3f}\n")
                            f.write(f"    Std: {values['std']:.3f}\n")
                            f.write(f"    Min: {values['min']:.3f}\n")
                            f.write(f"    Max: {values['max']:.3f}\n")
                            f.write(f"    Median: {values['median']:.3f}\n")
                
                # Visualization paths
                if 'visualization_paths' in self.analysis_summary:
                    f.write("\nGENERATED VISUALIZATIONS\n")
                    f.write("-" * 40 + "\n")
                    for viz_type, path in self.analysis_summary['visualization_paths'].items():
                        f.write(f"{viz_type}: {path}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 80 + "\n")
            
            logger.info(f"Generated report: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
    
    def run_full_pipeline(self, methods: Optional[list] = None, 
                         visualization_method: str = 'vader',
                         save_processed: bool = True) -> Dict:
        """
        Run the complete sentiment analysis pipeline.
        
        Args:
            methods (Optional[list]): Sentiment analysis methods to use
            visualization_method (str): Method to use for visualizations
            save_processed (bool): Whether to save processed data
            
        Returns:
            Dict: Complete pipeline results
        """
        logger.info("Starting full sentiment analysis pipeline...")
        
        try:
            # Step 1: Load and preprocess data
            self.load_and_preprocess_data(save_processed=save_processed)
            
            # Step 2: Run sentiment analysis
            self.run_sentiment_analysis(methods=methods)
            
            # Step 3: Generate visualizations
            visualization_paths = self.generate_visualizations(method=visualization_method)
            
            # Step 4: Save results
            results_path = self.save_results()
            
            # Step 5: Generate report
            report_path = self.generate_report()
            
            pipeline_results = {
                'processed_data': self.processed_data,
                'sentiment_results': self.sentiment_results,
                'analysis_summary': self.analysis_summary,
                'visualization_paths': visualization_paths,
                'results_file': results_path,
                'report_file': report_path
            }
            
            logger.info("Full pipeline completed successfully!")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Error in pipeline execution: {e}")
            raise


def run_sentiment_analysis(data_path: str, methods: Optional[list] = None,
                          results_dir: str = "results") -> Dict:
    """
    Convenience function to run sentiment analysis on a dataset.
    
    Args:
        data_path (str): Path to the input data file
        methods (Optional[list]): Sentiment analysis methods to use
        results_dir (str): Directory to save results
        
    Returns:
        Dict: Complete analysis results
    """
    pipeline = SentimentAnalysisPipeline(data_path, results_dir)
    return pipeline.run_full_pipeline(methods=methods)


def main():
    """Main function to run the sentiment analysis pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sentiment Analysis Pipeline')
    parser.add_argument('data_path', help='Path to the input CSV file')
    parser.add_argument('--methods', nargs='+', default=['textblob', 'vader'],
                       choices=['textblob', 'vader', 'transformers'],
                       help='Sentiment analysis methods to use')
    parser.add_argument('--results-dir', default='results',
                       help='Directory to save results')
    parser.add_argument('--viz-method', default='vader',
                       choices=['textblob', 'vader', 'transformers'],
                       help='Method to use for visualizations')
    
    args = parser.parse_args()
    
    try:
        # Run the pipeline
        results = run_sentiment_analysis(
            data_path=args.data_path,
            methods=args.methods,
            results_dir=args.results_dir
        )
        
        print(f"\nSentiment analysis completed successfully!")
        print(f"Results saved to: {results['results_file']}")
        print(f"Report generated: {results['report_file']}")
        print(f"Visualizations: {len(results['visualization_paths'])} files generated")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main() 