"""
Master script: Combine stock data, news, and sentiment analysis
"""
import sys
from pathlib import Path

# Add src to path so we can import our modules (MUST be before other imports)
sys.path.append(str(Path(__file__).parent / 'src'))

import logging
from datetime import datetime, timedelta
import pandas as pd
from sentiment.finbert_analyzer import FinBERTAnalyzer
from data_pipeline.news_scraper import NewsCollector
from data_pipeline.stock_data import StockDataCollector


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_market_sentiment(tickers: list, days_back: int = 7):
    """
    Main pipeline: Stock data → News → Sentiment analysis

    Args:
        tickers: List of stock tickers to analyze
        days_back: How many days back to fetch data
    """
    logger.info("="*70)
    logger.info("🚀 STARTING MARKET SENTIMENT ANALYSIS")
    logger.info("="*70)

    # ==================== STEP 1: GET STOCK DATA ====================
    logger.info("\n📊 STEP 1: Fetching stock data...")

    stock_collector = StockDataCollector(tickers)
    start_date = (datetime.now() - timedelta(days=days_back)
                  ).strftime('%Y-%m-%d')

    stock_df = stock_collector.get_historical_data(start_date)

    if stock_df.empty:
        logger.warning("⚠️  No stock data collected!")
        logger.info("   Using mock data instead...")
        # Fallback: run mock data script
        import subprocess
        subprocess.run(['python', 'create_mock_data.py'])
        stock_df = pd.read_csv('data/processed/stock_prices.csv')

    logger.info(f"✓ Collected stock data: {len(stock_df)} rows")

    # ==================== STEP 2: GET NEWS ====================
    logger.info("\n📰 STEP 2: Fetching financial news...")

    news_collector = NewsCollector()
    news_df = news_collector.get_news_for_multiple_tickers(
        tickers,
        limit_per_ticker=20
    )

    if news_df.empty:
        logger.error("❌ No news collected! Check your ALPHA_VANTAGE_KEY")
        return None

    logger.info(f"✓ Collected news: {len(news_df)} articles")

    # ==================== STEP 3: SENTIMENT ANALYSIS ====================
    logger.info("\n🤖 STEP 3: Running FinBERT sentiment analysis...")

    analyzer = FinBERTAnalyzer()

    # Combine title and summary for better sentiment analysis
    news_df['combined_text'] = news_df['title'] + \
        ' ' + news_df['summary'].fillna('')

    # Run sentiment analysis
    news_df = analyzer.analyze_dataframe(news_df, text_column='combined_text')

    logger.info(f"✓ Sentiment analysis complete!")

    # ==================== STEP 4: AGGREGATE & SAVE ====================
    logger.info("\n💾 STEP 4: Aggregating results and saving...")

    # Add date column for joining
    news_df['date'] = pd.to_datetime(news_df['time_published']).dt.date

    # Aggregate sentiment by ticker and date
    sentiment_agg = news_df.groupby(['ticker', 'date']).agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'sentiment_positive': 'mean',
        'sentiment_negative': 'mean'
    }).reset_index()

    # Flatten column names
    sentiment_agg.columns = [
        'ticker', 'date',
        'avg_sentiment', 'sentiment_std', 'num_articles',
        'avg_positive', 'avg_negative'
    ]

    # Merge with stock data
    combined_df = stock_df.merge(
        sentiment_agg,
        on=['ticker', 'date'],
        how='left'
    )

    # Save results
    combined_df.to_csv(
        'data/processed/stock_sentiment_combined.csv', index=False)
    news_df.to_csv('data/processed/news_with_sentiment.csv', index=False)

    logger.info("✓ Saved results:")
    logger.info("   - data/processed/stock_sentiment_combined.csv")
    logger.info("   - data/processed/news_with_sentiment.csv")

    return combined_df, news_df


def print_summary(combined_df, news_df):
    """
    Print analysis summary
    """
    print("\n" + "="*70)
    print("📊 ANALYSIS SUMMARY")
    print("="*70)

    # Stock data summary
    print("\n📈 STOCK DATA:")
    print(f"   Tickers: {combined_df['ticker'].nunique()}")
    print(
        f"   Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    print(f"   Total data points: {len(combined_df)}")

    # News summary
    print("\n📰 NEWS DATA:")
    print(f"   Total articles: {len(news_df)}")
    print(f"   Articles by ticker:")
    for ticker, count in news_df['ticker'].value_counts().items():
        print(f"      {ticker}: {count} articles")

    # Sentiment summary
    print("\n🤖 SENTIMENT ANALYSIS:")
    print(f"   Average sentiment: {news_df['sentiment_score'].mean():.3f}")
    print(f"   Sentiment distribution:")
    for label, count in news_df['sentiment_label'].value_counts().items():
        pct = (count / len(news_df)) * 100
        print(f"      {label.capitalize()}: {count} ({pct:.1f}%)")

    # Top positive news
    print("\n📈 TOP 3 MOST POSITIVE NEWS:")
    top_positive = news_df.nlargest(3, 'sentiment_score')[
        ['ticker', 'title', 'sentiment_score']]
    for idx, row in top_positive.iterrows():
        print(f"   [{row['ticker']}] {row['title'][:60]}...")
        print(f"             Sentiment: +{row['sentiment_score']:.3f}")

    # Top negative news
    print("\n📉 TOP 3 MOST NEGATIVE NEWS:")
    top_negative = news_df.nsmallest(3, 'sentiment_score')[
        ['ticker', 'title', 'sentiment_score']]
    for idx, row in top_negative.iterrows():
        print(f"   [{row['ticker']}] {row['title'][:60]}...")
        print(f"             Sentiment: {row['sentiment_score']:.3f}")

    # Correlation analysis (if we have both stock and sentiment data)
    valid_data = combined_df.dropna(subset=['avg_sentiment', 'daily_return'])
    if len(valid_data) > 10:
        correlation = valid_data['avg_sentiment'].corr(
            valid_data['daily_return'])
        print(f"\n🔗 CORRELATION:")
        print(f"   Sentiment vs Daily Return: {correlation:.3f}")
        if abs(correlation) > 0.3:
            print(f"   → Moderate correlation detected!")
        else:
            print(f"   → Weak correlation (need more data)")


if __name__ == "__main__":
    # Tickers to analyze
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'META']

    print("\n" + "="*70)
    print("🎯 STOCK MARKET SENTIMENT ANALYSIS")
    print("="*70)
    print(f"\nAnalyzing: {', '.join(tickers)}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run analysis
    result = analyze_market_sentiment(tickers, days_back=7)

    if result is not None:
        combined_df, news_df = result

        # Print summary
        print_summary(combined_df, news_df)

        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE!")
        print("="*70)
        print("\nResults saved to:")
        print("   📊 data/processed/stock_sentiment_combined.csv")
        print("   📰 data/processed/news_with_sentiment.csv")
        print("\nYou can now:")
        print("   1. Open CSV files in Excel/VS Code")
        print("   2. Create visualizations")
        print("   3. Build a dashboard")
    else:
        print("\n❌ Analysis failed. Check your API keys.")
