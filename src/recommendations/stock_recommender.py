"""
Stock Recommendation Engine based on Sentiment Analysis

This module provides:
1. Buy/Sell/Hold recommendations based on sentiment scores
2. Historical tracking of recommendations
3. Performance analysis of past recommendations
"""
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockRecommender:
    """
    Generate stock recommendations based on sentiment analysis.

    Recommendation Logic:
    - STRONG BUY: Very positive sentiment (> 0.5) + positive momentum
    - BUY: Positive sentiment (> 0.2)
    - HOLD: Neutral sentiment (-0.2 to 0.2)
    - SELL: Negative sentiment (< -0.2)
    - STRONG SELL: Very negative sentiment (< -0.5) + negative momentum
    """

    def __init__(self, history_file: str = "data/recommendations/history.json"):
        """
        Initialize the recommender.

        Args:
            history_file: Path to store recommendation history
        """
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.history = self._load_history()

        # Thresholds for recommendations
        self.thresholds = {
            'strong_buy': 0.5,
            'buy': 0.2,
            'hold_upper': 0.2,
            'hold_lower': -0.2,
            'sell': -0.2,
            'strong_sell': -0.5
        }

    def _load_history(self) -> List[Dict]:
        """Load recommendation history from file"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load history: {e}")
        return []

    def _save_history(self):
        """Save recommendation history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save history: {e}")

    def get_recommendation(
        self,
        ticker: str,
        sentiment_score: float,
        sentiment_std: float = 0.0,
        num_articles: int = 1,
        price_change_pct: float = 0.0
    ) -> Dict:
        """
        Generate recommendation for a single ticker.

        Args:
            ticker: Stock ticker symbol
            sentiment_score: Average sentiment score (-1 to 1)
            sentiment_std: Standard deviation of sentiment
            num_articles: Number of articles/posts analyzed
            price_change_pct: Recent price change percentage

        Returns:
            Dictionary with recommendation details
        """
        # Calculate confidence based on number of articles and consistency
        confidence = self._calculate_confidence(num_articles, sentiment_std)

        # Determine recommendation
        if sentiment_score >= self.thresholds['strong_buy'] and price_change_pct > -5:
            recommendation = "STRONG BUY"
            action = "buy"
            strength = min(1.0, sentiment_score + 0.2)
        elif sentiment_score >= self.thresholds['buy']:
            recommendation = "BUY"
            action = "buy"
            strength = sentiment_score
        elif sentiment_score <= self.thresholds['strong_sell'] and price_change_pct < 5:
            recommendation = "STRONG SELL"
            action = "sell"
            strength = min(1.0, abs(sentiment_score) + 0.2)
        elif sentiment_score <= self.thresholds['sell']:
            recommendation = "SELL"
            action = "sell"
            strength = abs(sentiment_score)
        else:
            recommendation = "HOLD"
            action = "hold"
            strength = 0.5

        return {
            'ticker': ticker,
            'recommendation': recommendation,
            'action': action,
            'strength': round(strength, 3),
            'confidence': round(confidence, 3),
            'sentiment_score': round(sentiment_score, 3),
            'num_articles': num_articles,
            'timestamp': datetime.now().isoformat()
        }

    def _calculate_confidence(self, num_articles: int, sentiment_std: float) -> float:
        """
        Calculate confidence score for recommendation.

        Higher confidence when:
        - More articles analyzed
        - Lower standard deviation (more consistent sentiment)
        """
        # Article count factor (0.3 to 1.0)
        article_factor = min(1.0, 0.3 + (num_articles / 20) * 0.7)

        # Consistency factor (higher when std is lower)
        consistency_factor = max(0.3, 1.0 - sentiment_std)

        return (article_factor * 0.6 + consistency_factor * 0.4)

    def generate_recommendations(
        self,
        sentiment_df: pd.DataFrame,
        stock_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate recommendations for all tickers in sentiment data.

        Args:
            sentiment_df: DataFrame with sentiment analysis results
            stock_df: Optional DataFrame with stock price data

        Returns:
            DataFrame with recommendations for each ticker
        """
        recommendations = []

        # Aggregate sentiment by ticker
        if 'ticker' not in sentiment_df.columns:
            logger.error("sentiment_df must have 'ticker' column")
            return pd.DataFrame()

        # Group by ticker
        ticker_sentiment = sentiment_df.groupby('ticker').agg({
            'sentiment_score': ['mean', 'std', 'count']
        }).reset_index()

        ticker_sentiment.columns = ['ticker', 'avg_sentiment', 'sentiment_std', 'num_articles']
        ticker_sentiment['sentiment_std'] = ticker_sentiment['sentiment_std'].fillna(0)

        # Get price changes if stock data available
        price_changes = {}
        if stock_df is not None and not stock_df.empty:
            for ticker in ticker_sentiment['ticker'].unique():
                ticker_prices = stock_df[stock_df['ticker'] == ticker].sort_values('date')
                if len(ticker_prices) >= 2:
                    start_price = ticker_prices.iloc[0]['close']
                    end_price = ticker_prices.iloc[-1]['close']
                    price_changes[ticker] = ((end_price - start_price) / start_price) * 100

        # Generate recommendation for each ticker
        for _, row in ticker_sentiment.iterrows():
            rec = self.get_recommendation(
                ticker=row['ticker'],
                sentiment_score=row['avg_sentiment'],
                sentiment_std=row['sentiment_std'],
                num_articles=int(row['num_articles']),
                price_change_pct=price_changes.get(row['ticker'], 0)
            )
            recommendations.append(rec)

        rec_df = pd.DataFrame(recommendations)

        # Sort by recommendation strength
        action_order = {'STRONG BUY': 0, 'BUY': 1, 'HOLD': 2, 'SELL': 3, 'STRONG SELL': 4}
        rec_df['sort_order'] = rec_df['recommendation'].map(action_order)
        rec_df = rec_df.sort_values(['sort_order', 'strength'], ascending=[True, False])
        rec_df = rec_df.drop('sort_order', axis=1)

        # Save to history
        self._add_to_history(rec_df)

        return rec_df

    def _add_to_history(self, recommendations_df: pd.DataFrame):
        """Add recommendations to history with summary stats"""
        timestamp = datetime.now().isoformat()
        date_str = timestamp[:10]

        # Calculate summary stats
        records = recommendations_df.to_dict('records')
        buy_count = len([r for r in records if r.get('action') == 'buy'])
        hold_count = len([r for r in records if r.get('action') == 'hold'])
        sell_count = len([r for r in records if r.get('action') == 'sell'])

        history_entry = {
            'timestamp': timestamp,
            'date': date_str,
            'recommendations': records,
            'summary': {
                'total': len(records),
                'buy': buy_count,
                'hold': hold_count,
                'sell': sell_count,
                'avg_confidence': float(recommendations_df['confidence'].mean()) if 'confidence' in recommendations_df.columns else 0
            }
        }

        # Keep only last 100 entries to avoid file bloat
        self.history = self.history[-99:] + [history_entry]
        self._save_history()

        logger.info(f"Added {len(recommendations_df)} recommendations to history (BUY: {buy_count}, HOLD: {hold_count}, SELL: {sell_count})")

    def get_top_picks(self, recommendations_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """
        Get top N stock picks (highest conviction buys).

        Args:
            recommendations_df: DataFrame with recommendations
            n: Number of top picks to return

        Returns:
            DataFrame with top picks
        """
        buy_recs = recommendations_df[
            recommendations_df['action'] == 'buy'
        ].copy()

        if buy_recs.empty:
            return pd.DataFrame()

        # Score = strength * confidence
        buy_recs['score'] = buy_recs['strength'] * buy_recs['confidence']
        buy_recs = buy_recs.sort_values('score', ascending=False)

        return buy_recs.head(n)

    def get_stocks_to_avoid(self, recommendations_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """
        Get top N stocks to avoid (highest conviction sells).

        Args:
            recommendations_df: DataFrame with recommendations
            n: Number of stocks to return

        Returns:
            DataFrame with stocks to avoid
        """
        sell_recs = recommendations_df[
            recommendations_df['action'] == 'sell'
        ].copy()

        if sell_recs.empty:
            return pd.DataFrame()

        sell_recs['score'] = sell_recs['strength'] * sell_recs['confidence']
        sell_recs = sell_recs.sort_values('score', ascending=False)

        return sell_recs.head(n)

    def calculate_historical_performance(
        self,
        stock_df: pd.DataFrame,
        lookback_days: int = 7
    ) -> pd.DataFrame:
        """
        Calculate how past recommendations would have performed.

        Args:
            stock_df: DataFrame with historical stock prices
            lookback_days: Days to look back for performance calculation

        Returns:
            DataFrame with performance metrics for each historical recommendation
        """
        if not self.history:
            logger.warning("No historical recommendations found")
            return pd.DataFrame()

        performance_results = []

        for entry in self.history:
            rec_timestamp = datetime.fromisoformat(entry['timestamp'])

            for rec in entry['recommendations']:
                ticker = rec['ticker']
                action = rec['action']
                rec_date = rec_timestamp.date()

                # Find stock prices after recommendation
                ticker_prices = stock_df[
                    (stock_df['ticker'] == ticker) &
                    (pd.to_datetime(stock_df['date']).dt.date >= rec_date)
                ].sort_values('date')

                if len(ticker_prices) < 2:
                    continue

                # Calculate returns
                entry_price = ticker_prices.iloc[0]['close']
                exit_price = ticker_prices.iloc[-1]['close']
                actual_return = ((exit_price - entry_price) / entry_price) * 100

                # Determine if recommendation was correct
                if action == 'buy':
                    correct = actual_return > 0
                    theoretical_return = actual_return
                elif action == 'sell':
                    correct = actual_return < 0
                    theoretical_return = -actual_return  # Profit from shorting
                else:  # hold
                    correct = abs(actual_return) < 5  # Small movement
                    theoretical_return = 0

                performance_results.append({
                    'date': rec_date,
                    'ticker': ticker,
                    'recommendation': rec['recommendation'],
                    'action': action,
                    'confidence': rec['confidence'],
                    'entry_price': round(entry_price, 2),
                    'exit_price': round(exit_price, 2),
                    'actual_return_pct': round(actual_return, 2),
                    'theoretical_return_pct': round(theoretical_return, 2),
                    'correct': correct
                })

        if not performance_results:
            return pd.DataFrame()

        perf_df = pd.DataFrame(performance_results)
        return perf_df

    def get_performance_summary(self, performance_df: pd.DataFrame) -> Dict:
        """
        Get summary statistics of historical performance.

        Args:
            performance_df: DataFrame from calculate_historical_performance

        Returns:
            Dictionary with performance metrics
        """
        if performance_df.empty:
            return {}

        total_recs = len(performance_df)
        correct_recs = performance_df['correct'].sum()
        accuracy = correct_recs / total_recs if total_recs > 0 else 0

        buy_recs = performance_df[performance_df['action'] == 'buy']
        sell_recs = performance_df[performance_df['action'] == 'sell']

        return {
            'total_recommendations': total_recs,
            'correct_predictions': int(correct_recs),
            'accuracy_pct': round(accuracy * 100, 1),
            'avg_theoretical_return_pct': round(performance_df['theoretical_return_pct'].mean(), 2),
            'total_theoretical_return_pct': round(performance_df['theoretical_return_pct'].sum(), 2),
            'buy_accuracy_pct': round(buy_recs['correct'].mean() * 100, 1) if len(buy_recs) > 0 else 0,
            'sell_accuracy_pct': round(sell_recs['correct'].mean() * 100, 1) if len(sell_recs) > 0 else 0,
            'best_pick': performance_df.loc[performance_df['theoretical_return_pct'].idxmax()].to_dict() if len(performance_df) > 0 else None,
            'worst_pick': performance_df.loc[performance_df['theoretical_return_pct'].idxmin()].to_dict() if len(performance_df) > 0 else None
        }


def main():
    """Test the recommender"""
    # Create sample data
    sentiment_data = pd.DataFrame({
        'ticker': ['AAPL', 'AAPL', 'MSFT', 'MSFT', 'TSLA', 'TSLA', 'GOOGL', 'NVDA'],
        'sentiment_score': [0.7, 0.5, 0.1, -0.1, -0.6, -0.4, 0.3, 0.8],
        'title': ['Article 1', 'Article 2', 'Article 3', 'Article 4',
                  'Article 5', 'Article 6', 'Article 7', 'Article 8']
    })

    recommender = StockRecommender()

    print("=" * 60)
    print("STOCK RECOMMENDATIONS")
    print("=" * 60)

    recommendations = recommender.generate_recommendations(sentiment_data)

    print("\nAll Recommendations:")
    print(recommendations[['ticker', 'recommendation', 'strength', 'confidence', 'sentiment_score']])

    print("\nTop Picks (BUY):")
    top_picks = recommender.get_top_picks(recommendations)
    if not top_picks.empty:
        print(top_picks[['ticker', 'recommendation', 'strength', 'confidence']])
    else:
        print("No buy recommendations")

    print("\nStocks to Avoid (SELL):")
    avoid = recommender.get_stocks_to_avoid(recommendations)
    if not avoid.empty:
        print(avoid[['ticker', 'recommendation', 'strength', 'confidence']])
    else:
        print("No sell recommendations")


if __name__ == "__main__":
    main()
