"""
Visualization module for sentiment analysis results.
Provides various charts and plots for analyzing sentiment data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class SentimentVisualizer:
    """Main visualization class for sentiment analysis results."""
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize the visualizer.
        
        Args:
            results_dir (str): Directory to save visualization outputs
        """
        self.results_dir = results_dir
        self._create_results_dir()
    
    def _create_results_dir(self):
        """Create results directory if it doesn't exist."""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            logger.info(f"Created results directory: {self.results_dir}")
    
    def plot_sentiment_distribution(self, data: pd.DataFrame, method: str = 'vader', 
                                  save_plot: bool = True) -> plt.Figure:
        """
        Plot distribution of sentiment scores.
        
        Args:
            data (pd.DataFrame): Data with sentiment scores
            method (str): Sentiment analysis method used
            save_plot (bool): Whether to save the plot
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Sentiment Distribution Analysis - {method.upper()}', fontsize=16)
        
        # Get sentiment score column
        score_col = f"{method}_compound" if method == 'vader' else f"{method}_polarity"
        
        if score_col not in data.columns:
            logger.error(f"Column {score_col} not found in data")
            return fig
        
        # 1. Histogram of sentiment scores
        axes[0, 0].hist(data[score_col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Sentiment Scores')
        axes[0, 0].set_xlabel('Sentiment Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(data[score_col].mean(), color='red', linestyle='--', 
                          label=f'Mean: {data[score_col].mean():.3f}')
        axes[0, 0].legend()
        
        # 2. Box plot
        axes[0, 1].boxplot(data[score_col])
        axes[0, 1].set_title('Box Plot of Sentiment Scores')
        axes[0, 1].set_ylabel('Sentiment Score')
        
        # 3. Sentiment classification pie chart
        sentiment_labels = []
        for score in data[score_col]:
            if score >= 0.05:
                sentiment_labels.append('Positive')
            elif score <= -0.05:
                sentiment_labels.append('Negative')
            else:
                sentiment_labels.append('Neutral')
        
        sentiment_counts = pd.Series(sentiment_labels).value_counts()
        axes[1, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Sentiment Classification Distribution')
        
        # 4. Cumulative distribution
        sorted_scores = np.sort(data[score_col])
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        axes[1, 1].plot(sorted_scores, cumulative, linewidth=2)
        axes[1, 1].set_title('Cumulative Distribution of Sentiment Scores')
        axes[1, 1].set_xlabel('Sentiment Score')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.results_dir, f'sentiment_distribution_{method}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved sentiment distribution plot to {plot_path}")
        
        return fig
    
    def plot_sentiment_by_product(self, data: pd.DataFrame, method: str = 'vader',
                                 save_plot: bool = True) -> plt.Figure:
        """
        Plot sentiment analysis results grouped by product.
        
        Args:
            data (pd.DataFrame): Data with sentiment scores and product information
            method (str): Sentiment analysis method used
            save_plot (bool): Whether to save the plot
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Sentiment Analysis by Product - {method.upper()}', fontsize=16)
        
        # Get sentiment score column
        score_col = f"{method}_compound" if method == 'vader' else f"{method}_polarity"
        
        if score_col not in data.columns:
            logger.error(f"Column {score_col} not found in data")
            return fig
        
        # 1. Box plot by product
        product_data = [data[data['product'] == product][score_col] 
                       for product in data['product'].unique()]
        axes[0, 0].boxplot(product_data, labels=data['product'].unique())
        axes[0, 0].set_title('Sentiment Scores by Product')
        axes[0, 0].set_ylabel('Sentiment Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Mean sentiment by product
        product_means = data.groupby('product')[score_col].mean().sort_values()
        axes[0, 1].bar(range(len(product_means)), product_means.values, 
                      color=['red' if x < 0 else 'green' for x in product_means.values])
        axes[0, 1].set_title('Average Sentiment Score by Product')
        axes[0, 1].set_ylabel('Average Sentiment Score')
        axes[0, 1].set_xticks(range(len(product_means)))
        axes[0, 1].set_xticklabels(product_means.index, rotation=45)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 3. Sentiment classification by product
        sentiment_labels = []
        for score in data[score_col]:
            if score >= 0.05:
                sentiment_labels.append('Positive')
            elif score <= -0.05:
                sentiment_labels.append('Negative')
            else:
                sentiment_labels.append('Neutral')
        
        data_with_labels = data.copy()
        data_with_labels['sentiment_label'] = sentiment_labels
        
        sentiment_by_product = pd.crosstab(data_with_labels['product'], 
                                          data_with_labels['sentiment_label'])
        sentiment_by_product.plot(kind='bar', ax=axes[1, 0], stacked=True)
        axes[1, 0].set_title('Sentiment Classification by Product')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend(title='Sentiment')
        
        # 4. Product sentiment heatmap
        product_stats = data.groupby('product')[score_col].agg(['mean', 'std', 'count']).round(3)
        sns.heatmap(product_stats.T, annot=True, cmap='RdYlGn', center=0, ax=axes[1, 1])
        axes[1, 1].set_title('Product Sentiment Statistics')
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.results_dir, f'sentiment_by_product_{method}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved product sentiment plot to {plot_path}")
        
        return fig
    
    def plot_sentiment_timeline(self, data: pd.DataFrame, method: str = 'vader',
                               save_plot: bool = True) -> plt.Figure:
        """
        Plot sentiment scores over time.
        
        Args:
            data (pd.DataFrame): Data with sentiment scores and datetime information
            method (str): Sentiment analysis method used
            save_plot (bool): Whether to save the plot
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Sentiment Timeline Analysis - {method.upper()}', fontsize=16)
        
        # Get sentiment score column
        score_col = f"{method}_compound" if method == 'vader' else f"{method}_polarity"
        
        if score_col not in data.columns or 'datetime_parsed' not in data.columns:
            logger.error(f"Required columns not found in data")
            return fig
        
        # Sort by datetime
        timeline_data = data.sort_values('datetime_parsed').copy()
        
        # 1. Sentiment scores over time
        axes[0, 0].scatter(timeline_data['datetime_parsed'], timeline_data[score_col], 
                          alpha=0.6, s=50)
        axes[0, 0].set_title('Sentiment Scores Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Sentiment Score')
        axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Rolling average sentiment
        window_size = min(7, len(timeline_data) // 4)  # Adaptive window size
        if window_size > 1:
            rolling_avg = timeline_data[score_col].rolling(window=window_size).mean()
            axes[0, 1].plot(timeline_data['datetime_parsed'], rolling_avg, 
                           linewidth=2, color='red', label=f'{window_size}-point moving average')
            axes[0, 1].scatter(timeline_data['datetime_parsed'], timeline_data[score_col], 
                              alpha=0.4, s=30, color='blue', label='Individual scores')
            axes[0, 1].set_title('Sentiment Trend (Moving Average)')
            axes[0, 1].set_xlabel('Date')
            axes[0, 1].set_ylabel('Sentiment Score')
            axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[0, 1].legend()
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Sentiment by hour of day
        hourly_sentiment = timeline_data.groupby('hour')[score_col].mean()
        axes[1, 0].bar(hourly_sentiment.index, hourly_sentiment.values, 
                      color=['red' if x < 0 else 'green' for x in hourly_sentiment.values])
        axes[1, 0].set_title('Average Sentiment by Hour of Day')
        axes[1, 0].set_xlabel('Hour')
        axes[1, 0].set_ylabel('Average Sentiment Score')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 4. Sentiment by day of week
        daily_sentiment = timeline_data.groupby('day_of_week')[score_col].mean()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_sentiment = daily_sentiment.reindex([day for day in day_order if day in daily_sentiment.index])
        
        axes[1, 1].bar(range(len(daily_sentiment)), daily_sentiment.values,
                      color=['red' if x < 0 else 'green' for x in daily_sentiment.values])
        axes[1, 1].set_title('Average Sentiment by Day of Week')
        axes[1, 1].set_ylabel('Average Sentiment Score')
        axes[1, 1].set_xticks(range(len(daily_sentiment)))
        axes[1, 1].set_xticklabels(daily_sentiment.index, rotation=45)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.results_dir, f'sentiment_timeline_{method}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved timeline plot to {plot_path}")
        
        return fig
    
    def create_interactive_dashboard(self, data: pd.DataFrame, method: str = 'vader',
                                   save_html: bool = True) -> go.Figure:
        """
        Create an interactive dashboard using Plotly.
        
        Args:
            data (pd.DataFrame): Data with sentiment scores
            method (str): Sentiment analysis method used
            save_html (bool): Whether to save as HTML file
            
        Returns:
            go.Figure: Plotly figure
        """
        # Get sentiment score column
        score_col = f"{method}_compound" if method == 'vader' else f"{method}_polarity"
        
        if score_col not in data.columns:
            logger.error(f"Column {score_col} not found in data")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment Distribution', 'Sentiment by Product', 
                          'Sentiment Timeline', 'Sentiment by Hour'),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Sentiment distribution histogram
        fig.add_trace(
            go.Histogram(x=data[score_col], name='Sentiment Distribution', 
                        nbinsx=30, marker_color='skyblue'),
            row=1, col=1
        )
        
        # 2. Average sentiment by product
        product_means = data.groupby('product')[score_col].mean().sort_values()
        colors = ['red' if x < 0 else 'green' for x in product_means.values]
        
        fig.add_trace(
            go.Bar(x=list(product_means.index), y=list(product_means.values),
                  name='Product Sentiment', marker_color=colors),
            row=1, col=2
        )
        
        # 3. Sentiment timeline
        timeline_data = data.sort_values('datetime_parsed')
        fig.add_trace(
            go.Scatter(x=timeline_data['datetime_parsed'], y=timeline_data[score_col],
                      mode='markers', name='Timeline', marker=dict(size=8, opacity=0.6)),
            row=2, col=1
        )
        
        # 4. Sentiment by hour
        hourly_sentiment = data.groupby('hour')[score_col].mean()
        hour_colors = ['red' if x < 0 else 'green' for x in hourly_sentiment.values]
        
        fig.add_trace(
            go.Bar(x=list(hourly_sentiment.index), y=list(hourly_sentiment.values),
                  name='Hourly Sentiment', marker_color=hour_colors),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'Interactive Sentiment Analysis Dashboard - {method.upper()}',
            height=800,
            showlegend=False
        )
        
        # Update axes
        fig.update_xaxes(title_text="Sentiment Score", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Product", row=1, col=2)
        fig.update_yaxes(title_text="Average Sentiment", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Sentiment Score", row=2, col=1)
        fig.update_xaxes(title_text="Hour", row=2, col=2)
        fig.update_yaxes(title_text="Average Sentiment", row=2, col=2)
        
        if save_html:
            html_path = os.path.join(self.results_dir, f'interactive_dashboard_{method}.html')
            fig.write_html(html_path)
            logger.info(f"Saved interactive dashboard to {html_path}")
        
        return fig
    
    def create_word_cloud(self, data: pd.DataFrame, sentiment_type: str = 'positive',
                         save_plot: bool = True) -> plt.Figure:
        """
        Create word cloud for positive or negative sentiments.
        
        Args:
            data (pd.DataFrame): Data with quotes and sentiment scores
            sentiment_type (str): 'positive' or 'negative'
            save_plot (bool): Whether to save the plot
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        try:
            from wordcloud import WordCloud
        except ImportError:
            logger.warning("WordCloud not available. Install with: pip install wordcloud")
            return None
        
        # Filter data by sentiment
        if sentiment_type == 'positive':
            filtered_data = data[data['vader_compound'] >= 0.05]
        elif sentiment_type == 'negative':
            filtered_data = data[data['vader_compound'] <= -0.05]
        else:
            filtered_data = data
        
        if len(filtered_data) == 0:
            logger.warning(f"No {sentiment_type} sentiments found")
            return None
        
        # Combine all text
        text = ' '.join(filtered_data['quote_cleaned'].astype(str))
        
        # Create word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                            max_words=100, colormap='viridis').generate(text)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Word Cloud - {sentiment_type.capitalize()} Sentiments', fontsize=16)
        
        if save_plot:
            plot_path = os.path.join(self.results_dir, f'wordcloud_{sentiment_type}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved word cloud to {plot_path}")
        
        return fig
    
    def generate_all_visualizations(self, data: pd.DataFrame, method: str = 'vader') -> Dict:
        """
        Generate all visualizations for the sentiment analysis results.
        
        Args:
            data (pd.DataFrame): Data with sentiment analysis results
            method (str): Sentiment analysis method used
            
        Returns:
            Dict: Dictionary with paths to generated visualizations
        """
        logger.info("Generating all visualizations...")
        
        visualization_paths = {}
        
        # Generate all plots
        try:
            self.plot_sentiment_distribution(data, method)
            visualization_paths['distribution'] = f'sentiment_distribution_{method}.png'
        except Exception as e:
            logger.error(f"Error generating distribution plot: {e}")
        
        try:
            self.plot_sentiment_by_product(data, method)
            visualization_paths['product'] = f'sentiment_by_product_{method}.png'
        except Exception as e:
            logger.error(f"Error generating product plot: {e}")
        
        try:
            self.plot_sentiment_timeline(data, method)
            visualization_paths['timeline'] = f'sentiment_timeline_{method}.png'
        except Exception as e:
            logger.error(f"Error generating timeline plot: {e}")
        
        try:
            self.create_interactive_dashboard(data, method)
            visualization_paths['dashboard'] = f'interactive_dashboard_{method}.html'
        except Exception as e:
            logger.error(f"Error generating interactive dashboard: {e}")
        
        try:
            self.create_word_cloud(data, 'positive')
            visualization_paths['wordcloud_positive'] = 'wordcloud_positive.png'
        except Exception as e:
            logger.error(f"Error generating positive word cloud: {e}")
        
        try:
            self.create_word_cloud(data, 'negative')
            visualization_paths['wordcloud_negative'] = 'wordcloud_negative.png'
        except Exception as e:
            logger.error(f"Error generating negative word cloud: {e}")
        
        logger.info(f"Generated {len(visualization_paths)} visualizations")
        return visualization_paths 