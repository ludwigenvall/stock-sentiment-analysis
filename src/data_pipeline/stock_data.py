"""
Stock price data collection using yfinance
"""
import logging
from typing import List, Dict
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

# Fix for Yahoo Finance rate limiting
import requests
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
})


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataCollector:
    """Collect historical and real-time stock prices"""

    def __init__(self, tickers: List[str]):
        """
        Args:
            tickers: List of stock tickers (e.g., ['AAPL', 'TSLA', 'MSFT'])
        """
        self.tickers = tickers

    def get_historical_data(
        self,
        start_date: str,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today

        Returns:
            DataFrame with columns: [date, ticker, open, high, low, close, volume]
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        logger.info(
            f"Fetching data for {len(self.tickers)} tickers from {start_date} to {end_date}")

        all_data = []

        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker, session=session)
                df = stock.history(start=start_date, end=end_date)

                if df.empty:
                    logger.warning(f"No data for {ticker}")
                    continue

                df = df.reset_index()
                df['ticker'] = ticker
                df['date'] = pd.to_datetime(df['Date']).dt.date

                df = df[['date', 'ticker', 'Open',
                         'High', 'Low', 'Close', 'Volume']]
                df.columns = ['date', 'ticker', 'open',
                              'high', 'low', 'close', 'volume']

                all_data.append(df)
                logger.info(f"✓ {ticker}: {len(df)} rows")

            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)
        return combined

    def get_current_price(self, ticker: str) -> Dict:
        """
        Get current price and basic info for a ticker

        Returns:
            Dict with keys: ticker, price, change_pct, volume, market_cap
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            return {
                'ticker': ticker,
                'price': info.get('currentPrice', info.get('regularMarketPrice')),
                'change_pct': info.get('regularMarketChangePercent', 0),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error getting current price for {ticker}: {e}")
            return {}

    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily returns for each ticker

        Args:
            df: DataFrame with columns [date, ticker, close]

        Returns:
            DataFrame with added 'daily_return' column
        """
        df = df.sort_values(['ticker', 'date'])
        df['daily_return'] = df.groupby('ticker')['close'].pct_change()
        return df


# Example usage
if __name__ == "__main__":
    # Top stocks to track
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'META']

    collector = StockDataCollector(tickers)

    # Get last 30 days
    start = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    df = collector.get_historical_data(start)

    print(f"\nCollected {len(df)} rows")
    print(df.head(10))

    # Check if we got any data
    if df.empty:
        print("\n❌ No data collected! This could be due to:")
        print("   - Yahoo Finance rate limiting")
        print("   - Network issues")
        print("   - Weekend/market closed")
        print("\nTry again in a few minutes or check your internet connection.")
    else:
        # Calculate returns
        df = collector.calculate_returns(df)
        print(f"\nDaily returns calculated")
        print(df[['date', 'ticker', 'close', 'daily_return']].head(10))

        # Save to CSV
        df.to_csv('data/processed/stock_prices.csv', index=False)
        print("\n✓ Saved to data/processed/stock_prices.csv")
