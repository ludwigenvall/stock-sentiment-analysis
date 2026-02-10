#!/usr/bin/env python3
"""
Fetch Historical Data for Stock Sentiment Analysis

This script fetches historical data from multiple sources:
- Stock prices from yfinance (full history)
- News from Alpha Vantage (limited historical)
- SEC filings from EDGAR (full history)

Usage:
    python fetch_historical_data.py --start 2024-01-01 --end 2025-01-01
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# S&P 100 tickers
SP100_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'AVGO', 'CSCO', 'ADBE', 'CRM',
    'ORCL', 'ACN', 'IBM', 'INTC', 'TXN', 'QCOM', 'AMD', 'AMAT', 'MU', 'ADI',
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TGT', 'COST', 'WMT',
    'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK',
    'SCHW', 'USB', 'PNC', 'COF', 'BRK.B',
    'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY',
    'AMGN', 'GILD', 'MDT', 'CVS', 'CI',
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'VLO', 'PSX', 'OXY',
    'NEE', 'DUK', 'SO', 'D', 'AEP',
    'BA', 'CAT', 'HON', 'UPS', 'RTX', 'LMT', 'GE', 'MMM', 'DE', 'EMR',
    'T', 'VZ', 'TMUS', 'CMCSA', 'CHTR', 'DIS', 'NFLX',
    'PG', 'KO', 'PEP', 'CL', 'PM', 'MO',
    'PYPL', 'GOOG'
]


def fetch_stock_prices(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical stock prices from yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed")
        return pd.DataFrame()

    logger.info(f"Fetching stock prices for {len(tickers)} tickers...")
    all_data = []

    for i, ticker in enumerate(tickers):
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)

            if not data.empty:
                ticker_df = pd.DataFrame({
                    'date': data.index,
                    'ticker': ticker,
                    'open': data['Open'].values,
                    'high': data['High'].values,
                    'low': data['Low'].values,
                    'close': data['Close'].values,
                    'volume': data['Volume'].values
                })
                ticker_df['date'] = pd.to_datetime(ticker_df['date']).dt.tz_localize(None)
                all_data.append(ticker_df)

            if (i + 1) % 20 == 0:
                logger.info(f"  Fetched {i + 1}/{len(tickers)} tickers...")

        except Exception as e:
            logger.warning(f"Failed to fetch {ticker}: {e}")

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        logger.info(f"Fetched {len(result)} price records for {result['ticker'].nunique()} tickers")
        return result

    return pd.DataFrame()


def fetch_news_alpha_vantage(tickers: list, api_key: str, days_back: int = 30) -> pd.DataFrame:
    """
    Fetch news from Alpha Vantage.
    Note: Alpha Vantage typically limits to recent news.
    """
    import requests

    logger.info(f"Fetching news from Alpha Vantage for {len(tickers)} tickers...")
    all_news = []

    # Process in batches of 5 tickers
    batch_size = 5
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        ticker_str = ','.join(batch)

        try:
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker_str}&apikey={api_key}&limit=1000"
            response = requests.get(url, timeout=30)
            data = response.json()

            if 'feed' in data:
                for article in data['feed']:
                    # Get ticker-specific sentiment
                    for ticker_data in article.get('ticker_sentiment', []):
                        ticker = ticker_data.get('ticker', '')
                        if ticker in batch:
                            all_news.append({
                                'ticker': ticker,
                                'title': article.get('title', ''),
                                'summary': article.get('summary', ''),
                                'source': article.get('source', ''),
                                'url': article.get('url', ''),
                                'time_published': article.get('time_published', ''),
                                'relevance_score': float(ticker_data.get('relevance_score', 0)),
                                'ticker_sentiment_score': float(ticker_data.get('ticker_sentiment_score', 0)),
                                'ticker_sentiment_label': ticker_data.get('ticker_sentiment_label', '')
                            })

            logger.info(f"  Processed batch {i // batch_size + 1}/{(len(tickers) + batch_size - 1) // batch_size}")

            # Rate limiting - Alpha Vantage allows 5 calls/minute on free tier
            time.sleep(12)

        except Exception as e:
            logger.warning(f"Failed to fetch news for batch {batch}: {e}")

    if all_news:
        df = pd.DataFrame(all_news)
        # Parse time
        df['time_published'] = pd.to_datetime(df['time_published'], format='%Y%m%dT%H%M%S', errors='coerce')
        df['date'] = df['time_published'].dt.date
        logger.info(f"Fetched {len(df)} news articles")
        return df

    return pd.DataFrame()


def fetch_sec_filings(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch SEC filings from EDGAR."""
    import requests

    logger.info(f"Fetching SEC filings for {len(tickers)} tickers...")
    all_filings = []

    headers = {
        'User-Agent': 'StockSentimentAnalysis contact@example.com',
        'Accept-Encoding': 'gzip, deflate'
    }

    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    for i, ticker in enumerate(tickers):
        try:
            # Get CIK for ticker
            cik_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=8-K&dateb=&owner=include&count=100&output=atom"
            response = requests.get(cik_url, headers=headers, timeout=30)

            if response.status_code == 200:
                # Parse XML-like response
                import re
                entries = re.findall(r'<entry>(.*?)</entry>', response.text, re.DOTALL)

                for entry in entries[:20]:  # Limit to 20 most recent
                    title_match = re.search(r'<title>(.*?)</title>', entry)
                    date_match = re.search(r'<updated>(.*?)</updated>', entry)
                    link_match = re.search(r'<link href="(.*?)"', entry)

                    if title_match and date_match:
                        filing_date = datetime.fromisoformat(date_match.group(1).replace('Z', '+00:00'))

                        if start_dt <= filing_date.replace(tzinfo=None) <= end_dt:
                            all_filings.append({
                                'ticker': ticker,
                                'title': title_match.group(1),
                                'date': filing_date.date(),
                                'url': link_match.group(1) if link_match else '',
                                'filing_type': '8-K'
                            })

            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{len(tickers)} tickers...")

            time.sleep(0.5)  # Rate limiting for SEC

        except Exception as e:
            logger.warning(f"Failed to fetch SEC filings for {ticker}: {e}")

    if all_filings:
        df = pd.DataFrame(all_filings)
        logger.info(f"Fetched {len(df)} SEC filings")
        return df

    return pd.DataFrame()


def generate_synthetic_sentiment(prices_df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    Generate synthetic sentiment scores based on price momentum.
    This helps when historical news data is limited.

    The idea: past returns partially predict news sentiment
    (positive returns → more positive coverage)
    """
    logger.info("Generating synthetic sentiment from price momentum...")

    sentiment_data = []

    for ticker in prices_df['ticker'].unique():
        ticker_prices = prices_df[prices_df['ticker'] == ticker].copy()
        ticker_prices = ticker_prices.sort_values('date')

        if len(ticker_prices) < lookback + 1:
            continue

        # Calculate momentum
        ticker_prices['momentum'] = ticker_prices['close'].pct_change(lookback)
        ticker_prices['volatility'] = ticker_prices['close'].pct_change().rolling(lookback).std()

        for _, row in ticker_prices.dropna().iterrows():
            # Convert momentum to sentiment-like score
            # Clip to [-1, 1] range
            base_sentiment = np.clip(row['momentum'] * 5, -0.8, 0.8)

            # Add some noise
            noise = np.random.normal(0, 0.1)
            sentiment = np.clip(base_sentiment + noise, -1, 1)

            sentiment_data.append({
                'ticker': ticker,
                'date': row['date'],
                'sentiment_score': round(sentiment, 4),
                'sentiment_label': 'positive' if sentiment > 0.15 else 'negative' if sentiment < -0.15 else 'neutral',
                'content_type': 'synthetic',
                'title': f"Synthetic sentiment for {ticker}"
            })

    if sentiment_data:
        df = pd.DataFrame(sentiment_data)
        logger.info(f"Generated {len(df)} synthetic sentiment records")
        return df

    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description='Fetch historical stock data')
    parser.add_argument('--start', type=str, default='2024-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--tickers', type=str, default='sp100',
                       help='Ticker list: sp100, or comma-separated tickers')
    parser.add_argument('--skip-news', action='store_true',
                       help='Skip news fetching (API rate limited)')
    parser.add_argument('--skip-sec', action='store_true',
                       help='Skip SEC filings')
    parser.add_argument('--generate-synthetic', action='store_true',
                       help='Generate synthetic sentiment from price data')
    args = parser.parse_args()

    if args.end is None:
        args.end = datetime.now().strftime('%Y-%m-%d')

    # Get tickers
    if args.tickers.lower() == 'sp100':
        tickers = SP100_TICKERS
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(',')]

    # Remove duplicates
    tickers = list(dict.fromkeys(tickers))

    print("\n" + "=" * 60)
    print("FETCHING HISTORICAL DATA")
    print("=" * 60)
    print(f"Date range: {args.start} to {args.end}")
    print(f"Tickers: {len(tickers)}")

    # Create directories
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    Path('data/processed').mkdir(parents=True, exist_ok=True)

    # 1. Fetch stock prices
    print("\n--- Fetching Stock Prices ---")
    prices_df = fetch_stock_prices(tickers, args.start, args.end)

    if not prices_df.empty:
        prices_df.to_csv('data/processed/stock_prices.csv', index=False)
        print(f"Saved {len(prices_df)} price records to data/processed/stock_prices.csv")

    # 2. Fetch news (optional)
    news_df = pd.DataFrame()
    if not args.skip_news:
        print("\n--- Fetching News (this may take a while due to API limits) ---")
        from dotenv import load_dotenv
        import os
        load_dotenv()

        api_key = os.getenv('ALPHA_VANTAGE_KEY')
        if api_key:
            news_df = fetch_news_alpha_vantage(tickers[:20], api_key)  # Limit due to API
            if not news_df.empty:
                news_df.to_csv('data/raw/news_articles.csv', index=False)
                print(f"Saved {len(news_df)} news articles")
        else:
            print("ALPHA_VANTAGE_KEY not found in .env - skipping news")

    # 3. Fetch SEC filings (optional)
    sec_df = pd.DataFrame()
    if not args.skip_sec:
        print("\n--- Fetching SEC Filings ---")
        sec_df = fetch_sec_filings(tickers[:30], args.start, args.end)  # Limit to avoid rate limits
        if not sec_df.empty:
            sec_df.to_csv('data/raw/sec_filings.csv', index=False)
            print(f"Saved {len(sec_df)} SEC filings")

    # 4. Generate synthetic sentiment (optional)
    synthetic_df = pd.DataFrame()
    if args.generate_synthetic and not prices_df.empty:
        print("\n--- Generating Synthetic Sentiment ---")
        synthetic_df = generate_synthetic_sentiment(prices_df)
        if not synthetic_df.empty:
            # Combine with existing sentiment data if any
            existing_file = Path('data/processed/all_sentiment.csv')
            if existing_file.exists():
                existing_df = pd.read_csv(existing_file)
                # Remove old synthetic data
                existing_df = existing_df[existing_df['content_type'] != 'synthetic']
                combined = pd.concat([existing_df, synthetic_df], ignore_index=True)
            else:
                combined = synthetic_df

            combined.to_csv('data/processed/all_sentiment.csv', index=False)
            print(f"Saved {len(combined)} total sentiment records")

    # Summary
    print("\n" + "=" * 60)
    print("FETCH COMPLETE")
    print("=" * 60)
    print(f"\nData saved:")
    print(f"  - Stock prices: {len(prices_df)} records")
    if not news_df.empty:
        print(f"  - News articles: {len(news_df)} records")
    if not sec_df.empty:
        print(f"  - SEC filings: {len(sec_df)} records")
    if not synthetic_df.empty:
        print(f"  - Synthetic sentiment: {len(synthetic_df)} records")

    print(f"\nNext steps:")
    print("  1. Run sentiment analysis on news/filings:")
    print("     python analyze_with_recommendations.py --list sp100 --days 30")
    print("  2. Retrain ML model with new data:")
    print("     python train_model.py --start 2024-01-01")


if __name__ == "__main__":
    main()
