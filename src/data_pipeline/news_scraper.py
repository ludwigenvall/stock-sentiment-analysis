"""
Financial news scraper using Alpha Vantage News API
"""
import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsCollector:
    """Collect financial news from Alpha Vantage"""

    def __init__(self, api_key: str = None):
        """
        Args:
            api_key: Alpha Vantage API key (or set in .env)
        """
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_KEY')
        self.base_url = "https://www.alphavantage.co/query"

        if not self.api_key:
            raise ValueError(
                "Alpha Vantage API key required! Set ALPHA_VANTAGE_KEY in .env")

        logger.info("✓ NewsCollector initialized with API key")

    def get_news_for_ticker(
        self,
        ticker: str,
        limit: int = 50
    ) -> List[Dict]:
        """
        Fetch latest news for a ticker

        Args:
            ticker: Stock ticker (e.g., 'AAPL')
            limit: Max number of articles

        Returns:
            List of dicts with keys: [title, url, time_published, summary, source, ticker]
        """
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ticker,
            'limit': limit,
            'apikey': self.api_key
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'feed' not in data:
                logger.warning(f"No news found for {ticker}")
                return []

            articles = []
            for item in data['feed']:
                # Parse ticker sentiment if available
                ticker_sentiment = None
                if 'ticker_sentiment' in item:
                    for ts in item['ticker_sentiment']:
                        if ts['ticker'] == ticker:
                            ticker_sentiment = float(
                                ts.get('ticker_sentiment_score', 0))
                            break

                article = {
                    'title': item.get('title', ''),
                    'url': item.get('url', ''),
                    'time_published': item.get('time_published', ''),
                    'summary': item.get('summary', ''),
                    'source': item.get('source', ''),
                    'ticker': ticker,
                    'sentiment_score': ticker_sentiment,
                    'collected_at': datetime.now()
                }
                articles.append(article)

            logger.info(f"✓ {ticker}: Collected {len(articles)} articles")
            return articles

        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return []

    def get_news_for_multiple_tickers(
        self,
        tickers: List[str],
        limit_per_ticker: int = 20
    ) -> pd.DataFrame:
        """
        Fetch news for multiple tickers

        Returns:
            DataFrame with columns: [ticker, title, summary, url, time_published, source, collected_at]
        """
        all_articles = []

        for ticker in tickers:
            articles = self.get_news_for_ticker(ticker, limit=limit_per_ticker)
            all_articles.extend(articles)

        if not all_articles:
            return pd.DataFrame()

        df = pd.DataFrame(all_articles)

        # Convert time_published to datetime
        df['time_published'] = pd.to_datetime(
            df['time_published'],
            format='%Y%m%dT%H%M%S',
            errors='coerce'
        )

        # Remove duplicates (same article mentioning multiple tickers)
        df = df.drop_duplicates(subset=['url'])

        return df


# Example usage
if __name__ == "__main__":
    collector = NewsCollector()

    # Test with a few tickers
    tickers = ['AAPL', 'TSLA', 'NVDA']

    print("\n📰 TESTING NEWS SCRAPER")
    print("="*60)

    df = collector.get_news_for_multiple_tickers(tickers, limit_per_ticker=10)

    if df.empty:
        print("\n❌ No news collected!")
        print("   Check your ALPHA_VANTAGE_KEY in .env")
        print("   Or you may have hit the 25 requests/day limit")
    else:
        print(f"\n✓ Collected {len(df)} unique articles")
        print(f"\nLatest headlines:")
        print(df[['ticker', 'title', 'time_published']].head(
            10).to_string(index=False))

        # Save to CSV
        df.to_csv('data/raw/news_articles.csv', index=False)
        print(f"\n✓ Saved to data/raw/news_articles.csv")

        print("\n📊 Articles per ticker:")
        print(df['ticker'].value_counts())
