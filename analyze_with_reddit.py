"""
Enhanced analysis: Stock data + News + Reddit + Sentiment analysis
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
from data_pipeline.reddit_scraper import RedditScraper
from data_pipeline.stock_data import StockDataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_with_reddit(tickers: list, days_back: int = 7):
    """
    Enhanced pipeline: Stock data → News + Reddit → Sentiment analysis

    Args:
        tickers: List of stock tickers to analyze
        days_back: How many days back to fetch stock data
    """
    logger.info("="*70)
    logger.info("STARTING ENHANCED SENTIMENT ANALYSIS (News + Reddit)")
    logger.info("="*70)

    # ==================== STEP 1: GET STOCK DATA ====================
    logger.info("\nSTEP 1: Fetching stock data...")

    stock_collector = StockDataCollector(tickers)
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

    stock_df = stock_collector.get_historical_data(start_date)

    if stock_df.empty:
        logger.warning("No stock data collected!")
        logger.info("Using mock data instead...")
        import subprocess
        subprocess.run(['python', 'create_mock_data.py'])
        stock_df = pd.read_csv('data/processed/stock_prices.csv')

    logger.info(f"Collected stock data: {len(stock_df)} rows")

    # ==================== STEP 2: GET NEWS ====================
    logger.info("\nSTEP 2: Fetching financial news...")

    news_collector = NewsCollector()
    news_df = news_collector.get_news_for_multiple_tickers(
        tickers,
        limit_per_ticker=20
    )

    if news_df.empty:
        logger.error("No news collected! Check your ALPHA_VANTAGE_KEY")
        news_df = pd.DataFrame()
    else:
        logger.info(f"Collected news: {len(news_df)} articles")

    # ==================== STEP 3: GET REDDIT DATA ====================
    logger.info("\nSTEP 3: Fetching Reddit posts...")

    reddit_scraper = RedditScraper()
    reddit_df = reddit_scraper.scrape_multiple_subreddits(
        tickers=tickers,
        limit_per_sub=50
    )

    if reddit_df.empty:
        logger.warning("No Reddit data collected. Check your Reddit API credentials")
        reddit_df = pd.DataFrame()
    else:
        logger.info(f"Collected Reddit posts: {len(reddit_df)}")
        logger.info(f"   Breakdown: {dict(reddit_df['ticker'].value_counts())}")

        # Save Reddit data
        reddit_df.to_csv('data/raw/reddit_posts.csv', index=False)
        logger.info("Saved Reddit data to: data/raw/reddit_posts.csv")

    # ==================== STEP 4: SENTIMENT ANALYSIS ====================
    logger.info("\nSTEP 4: Running FinBERT sentiment analysis...")

    analyzer = FinBERTAnalyzer()

    # Analyze news if available
    if not news_df.empty:
        news_df['combined_text'] = news_df['title'] + ' ' + news_df['summary'].fillna('')
        news_df = analyzer.analyze_dataframe(news_df, text_column='combined_text')
        logger.info(f"Analyzed {len(news_df)} news articles")

    # Analyze Reddit posts if available
    if not reddit_df.empty:
        reddit_df['combined_text'] = reddit_df['title'] + ' ' + reddit_df['text'].fillna('')
        reddit_df = analyzer.analyze_dataframe(reddit_df, text_column='combined_text')
        logger.info(f"Analyzed {len(reddit_df)} Reddit posts")

    # ==================== STEP 5: COMBINE DATA ====================
    logger.info("\nSTEP 5: Combining and aggregating results...")

    all_content = []

    # Add news
    if not news_df.empty:
        news_df['date'] = pd.to_datetime(news_df['time_published']).dt.date
        news_df['content_type'] = 'news'
        all_content.append(news_df[['ticker', 'date', 'sentiment_score', 'sentiment_positive',
                                     'sentiment_negative', 'sentiment_neutral', 'sentiment_label',
                                     'content_type', 'title', 'source']])

    # Add Reddit
    if not reddit_df.empty:
        reddit_df['date'] = pd.to_datetime(reddit_df['created_utc']).dt.date
        reddit_df['content_type'] = 'reddit'
        all_content.append(reddit_df[['ticker', 'date', 'sentiment_score', 'sentiment_positive',
                                       'sentiment_negative', 'sentiment_neutral', 'sentiment_label',
                                       'content_type', 'title', 'source']])

    if not all_content:
        logger.error("No content to analyze!")
        return None

    # Combine all content
    combined_content = pd.concat(all_content, ignore_index=True)

    # Overall aggregation (all sources)
    overall_agg = combined_content.groupby(['ticker', 'date']).agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'sentiment_positive': 'mean',
        'sentiment_negative': 'mean',
        'sentiment_neutral': 'mean'
    }).reset_index()

    overall_agg.columns = [
        'ticker', 'date',
        'avg_sentiment', 'sentiment_std', 'total_mentions',
        'avg_positive', 'avg_negative', 'avg_neutral'
    ]

    # By content type
    by_type = combined_content.groupby(['ticker', 'date', 'content_type']).agg({
        'sentiment_score': ['mean', 'count']
    }).reset_index()

    by_type.columns = ['ticker', 'date', 'content_type', 'avg_sentiment', 'count']

    # Merge with stock data
    stock_df['date'] = pd.to_datetime(stock_df['date']).dt.date
    final_df = stock_df.merge(
        overall_agg,
        on=['ticker', 'date'],
        how='left'
    )

    # ==================== STEP 6: SAVE RESULTS ====================
    logger.info("\nSTEP 6: Saving results...")

    final_df.to_csv('data/processed/stock_sentiment_reddit_combined.csv', index=False)
    combined_content.to_csv('data/processed/all_content_with_sentiment.csv', index=False)
    by_type.to_csv('data/processed/sentiment_by_type.csv', index=False)

    logger.info("Saved results:")
    logger.info("   - data/processed/stock_sentiment_reddit_combined.csv")
    logger.info("   - data/processed/all_content_with_sentiment.csv")
    logger.info("   - data/processed/sentiment_by_type.csv")

    # ==================== STEP 7: SUMMARY ====================
    print_summary(combined_content, stock_df)

    return final_df, combined_content


def print_summary(content_df, stock_df):
    """Print analysis summary"""
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)

    # Stock data
    print("\nSTOCK DATA:")
    print(f"   Tickers: {stock_df['ticker'].nunique()}")
    print(f"   Date range: {stock_df['date'].min()} to {stock_df['date'].max()}")
    print(f"   Total data points: {len(stock_df)}")

    # Content breakdown
    print("\nCONTENT COLLECTED:")
    print(f"   Total items: {len(content_df)}")

    content_by_type = content_df['content_type'].value_counts()
    for content_type, count in content_by_type.items():
        print(f"   {content_type.capitalize()}: {count}")

    # Sentiment by content type
    print("\nSENTIMENT BY SOURCE:")
    for content_type in content_df['content_type'].unique():
        subset = content_df[content_df['content_type'] == content_type]
        avg_sentiment = subset['sentiment_score'].mean()
        print(f"   {content_type.capitalize()}: {avg_sentiment:.3f}")

    # Overall sentiment
    print("\nOVERALL SENTIMENT:")
    print(f"   Average: {content_df['sentiment_score'].mean():.3f}")
    print(f"   Distribution:")

    for label, count in content_df['sentiment_label'].value_counts().items():
        pct = (count / len(content_df)) * 100
        print(f"      {label.capitalize()}: {count} ({pct:.1f}%)")

    # Top mentions
    print("\nTOP MENTIONED TICKERS:")
    for ticker, count in content_df['ticker'].value_counts().head(5).items():
        print(f"   {ticker}: {count} mentions")

    # Most positive by type
    print("\nTOP POSITIVE CONTENT:")
    for content_type in content_df['content_type'].unique():
        subset = content_df[content_df['content_type'] == content_type]
        if len(subset) > 0:
            top = subset.nlargest(1, 'sentiment_score').iloc[0]
            print(f"   [{content_type.upper()}] [{top['ticker']}] {top['title'][:60]}...")
            print(f"      Sentiment: +{top['sentiment_score']:.3f}")

    # Most negative by type
    print("\nTOP NEGATIVE CONTENT:")
    for content_type in content_df['content_type'].unique():
        subset = content_df[content_df['content_type'] == content_type]
        if len(subset) > 0:
            bottom = subset.nsmallest(1, 'sentiment_score').iloc[0]
            print(f"   [{content_type.upper()}] [{bottom['ticker']}] {bottom['title'][:60]}...")
            print(f"      Sentiment: {bottom['sentiment_score']:.3f}")


if __name__ == "__main__":
    # Tickers to analyze
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'META']

    print("\n" + "="*70)
    print("ENHANCED STOCK SENTIMENT ANALYSIS (News + Reddit)")
    print("="*70)
    print(f"\nAnalyzing: {', '.join(tickers)}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run analysis
    result = analyze_with_reddit(tickers, days_back=7)

    if result is not None:
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print("\nNext steps:")
        print("   1. View results in data/processed/")
        print("   2. Run dashboard: streamlit run src/dashboard/app.py")
        print("   3. Compare News vs Reddit sentiment")
    else:
        print("\nAnalysis failed. Check your API keys.")
