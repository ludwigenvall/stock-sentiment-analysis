"""
Full Analysis Pipeline with Stock Recommendations

Combines:
- Stock data collection
- News sentiment analysis
- Reddit sentiment analysis (if configured)
- SEC filings analysis (free, no API key needed)
- Earnings data from Yahoo Finance (free)
- AI-powered stock recommendations
- Historical performance tracking
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import logging
from datetime import datetime, timedelta
import pandas as pd

from sentiment.finbert_analyzer import FinBERTAnalyzer
from data_pipeline.news_scraper import NewsCollector
from data_pipeline.stock_data import StockDataCollector
from recommendations.stock_recommender import StockRecommender
from config.tickers import get_tickers, TICKER_LISTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_full_analysis(tickers: list, days_back: int = 7):
    """
    Run complete analysis pipeline with recommendations.

    Args:
        tickers: List of stock tickers to analyze
        days_back: Days of historical data to fetch

    Returns:
        Tuple of (recommendations_df, sentiment_df, stock_df)
    """
    logger.info("=" * 70)
    logger.info("STOCK SENTIMENT ANALYSIS WITH RECOMMENDATIONS")
    logger.info("=" * 70)
    logger.info(f"Analyzing {len(tickers)} tickers")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ==================== STEP 1: COLLECT STOCK DATA ====================
    logger.info("\nSTEP 1: Fetching stock data...")

    stock_collector = StockDataCollector(tickers)
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    stock_df = stock_collector.get_historical_data(start_date)

    if stock_df.empty:
        logger.warning("No stock data collected, using mock data")
        try:
            stock_df = pd.read_csv('data/processed/stock_prices.csv')
        except Exception:
            logger.error("Could not load stock data")
            stock_df = pd.DataFrame()

    logger.info(f"Collected stock data: {len(stock_df)} rows")

    # ==================== STEP 2: COLLECT NEWS ====================
    logger.info("\nSTEP 2: Fetching financial news...")

    news_collector = NewsCollector()

    # Process tickers in batches to avoid API rate limits
    all_news = []
    batch_size = 10

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        logger.info(f"  Fetching news for batch {i // batch_size + 1}: {', '.join(batch[:5])}...")

        batch_news = news_collector.get_news_for_multiple_tickers(batch, limit_per_ticker=10)
        if not batch_news.empty:
            all_news.append(batch_news)

    if all_news:
        news_df = pd.concat(all_news, ignore_index=True)
        news_df = news_df.drop_duplicates(subset=['url'])
        logger.info(f"Collected news: {len(news_df)} unique articles")
    else:
        logger.warning("No news collected")
        news_df = pd.DataFrame()

    # ==================== STEP 3: REDDIT DATA (OPTIONAL) ====================
    logger.info("\nSTEP 3: Fetching Reddit data...")

    try:
        from data_pipeline.reddit_scraper import RedditScraper

        reddit_scraper = RedditScraper()
        if reddit_scraper.reddit:
            reddit_df = reddit_scraper.scrape_multiple_subreddits(
                tickers=tickers[:20],  # Limit to top 20 for Reddit
                limit_per_sub=30
            )
            logger.info(f"Collected Reddit posts: {len(reddit_df)}")
        else:
            logger.warning("Reddit API not configured, skipping")
            reddit_df = pd.DataFrame()
    except Exception as e:
        logger.warning(f"Reddit scraping failed: {e}")
        reddit_df = pd.DataFrame()

    # ==================== STEP 3b: SEC FILINGS ====================
    logger.info("\nSTEP 3b: Fetching SEC filings...")

    try:
        from data_pipeline.sec_collector import SECCollector

        sec_collector = SECCollector()
        sec_df = sec_collector.collect_for_tickers(
            tickers=tickers,
            filing_types=['10-K', '10-Q', '8-K'],
            limit_per_ticker=2
        )
        logger.info(f"Collected SEC filings: {len(sec_df)}")
    except Exception as e:
        logger.warning(f"SEC collection failed: {e}")
        sec_df = pd.DataFrame()

    # ==================== STEP 3c: EARNINGS DATA ====================
    logger.info("\nSTEP 3c: Fetching earnings data...")

    try:
        from data_pipeline.earnings_collector import EarningsCollector, YahooEarningsCollector

        # Try FMP first (requires premium), fallback to Yahoo
        earnings_collector = EarningsCollector()
        if earnings_collector.is_configured:
            earnings_df = earnings_collector.collect_for_tickers(
                tickers=tickers[:30],
                limit_per_ticker=2
            )
            if earnings_df.empty:
                logger.info("FMP returned no data, trying Yahoo fallback...")
                yahoo_collector = YahooEarningsCollector()
                earnings_df = yahoo_collector.collect_for_tickers(tickers[:30])
        else:
            # Use Yahoo fallback (free, no API key needed)
            logger.info("Using Yahoo Finance for earnings data (free)")
            yahoo_collector = YahooEarningsCollector()
            earnings_df = yahoo_collector.collect_for_tickers(tickers[:30])

        logger.info(f"Collected earnings data: {len(earnings_df)}")
    except Exception as e:
        logger.warning(f"Earnings collection failed: {e}")
        earnings_df = pd.DataFrame()

    # ==================== STEP 4: SENTIMENT ANALYSIS ====================
    logger.info("\nSTEP 4: Running FinBERT sentiment analysis...")

    analyzer = FinBERTAnalyzer()
    all_content = []

    # Analyze news
    if not news_df.empty:
        news_df['combined_text'] = news_df['title'] + ' ' + news_df['summary'].fillna('')
        news_df = analyzer.analyze_dataframe(news_df, text_column='combined_text')
        news_df['content_type'] = 'news'
        news_df['date'] = pd.to_datetime(news_df['time_published']).dt.date
        all_content.append(news_df[['ticker', 'date', 'sentiment_score', 'sentiment_label',
                                     'content_type', 'title']])
        logger.info(f"Analyzed {len(news_df)} news articles")

    # Analyze Reddit
    if not reddit_df.empty:
        reddit_df['combined_text'] = reddit_df['title'] + ' ' + reddit_df['text'].fillna('')
        reddit_df = analyzer.analyze_dataframe(reddit_df, text_column='combined_text')
        reddit_df['content_type'] = 'reddit'
        reddit_df['date'] = pd.to_datetime(reddit_df['created_utc']).dt.date
        all_content.append(reddit_df[['ticker', 'date', 'sentiment_score', 'sentiment_label',
                                       'content_type', 'title']])
        logger.info(f"Analyzed {len(reddit_df)} Reddit posts")

    # Analyze SEC filings
    if not sec_df.empty:
        sec_df['combined_text'] = sec_df['title'] + ' ' + sec_df['text'].fillna('')
        sec_df = analyzer.analyze_dataframe(sec_df, text_column='combined_text')
        sec_df['content_type'] = 'sec_filing'
        sec_df['date'] = pd.to_datetime(sec_df['date_filed']).dt.date
        all_content.append(sec_df[['ticker', 'date', 'sentiment_score', 'sentiment_label',
                                    'content_type', 'title']])
        logger.info(f"Analyzed {len(sec_df)} SEC filings")

    # Analyze Earnings data
    if not earnings_df.empty:
        # Handle both FMP format (title, text) and Yahoo format (title, text from _create_earnings_text)
        if 'text' in earnings_df.columns:
            earnings_df['combined_text'] = earnings_df['title'].fillna('') + ' ' + earnings_df['text'].fillna('')
        else:
            earnings_df['combined_text'] = earnings_df['title'].fillna('')

        earnings_df = analyzer.analyze_dataframe(earnings_df, text_column='combined_text')
        earnings_df['content_type'] = 'earnings'

        # Ensure date column exists
        if 'date' not in earnings_df.columns:
            earnings_df['date'] = pd.Timestamp.now().date()

        all_content.append(earnings_df[['ticker', 'date', 'sentiment_score', 'sentiment_label',
                                         'content_type', 'title']])
        logger.info(f"Analyzed {len(earnings_df)} earnings records")

    if not all_content:
        logger.error("No content to analyze!")
        return None

    combined_sentiment = pd.concat(all_content, ignore_index=True)

    # ==================== STEP 5: GENERATE RECOMMENDATIONS ====================
    logger.info("\nSTEP 5: Generating stock recommendations...")

    # Use ML-enhanced recommendations if model exists
    ml_model_path = Path("models/sentiment_predictor.pkl")
    use_ml = ml_model_path.exists()

    recommender = StockRecommender(use_ml=use_ml, ml_weight=0.3)
    recommendations = recommender.generate_recommendations(combined_sentiment, stock_df)

    if use_ml:
        logger.info("Using ML-enhanced recommendations (30% ML, 70% rule-based)")

    logger.info(f"Generated {len(recommendations)} recommendations")

    # ==================== STEP 6: SAVE RESULTS ====================
    logger.info("\nSTEP 6: Saving results...")

    # Create directories
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    Path('data/recommendations').mkdir(parents=True, exist_ok=True)

    # Save files
    combined_sentiment.to_csv('data/processed/all_sentiment.csv', index=False)
    recommendations.to_csv('data/recommendations/latest_recommendations.csv', index=False)

    if not stock_df.empty:
        stock_df.to_csv('data/processed/stock_prices.csv', index=False)

    logger.info("Saved:")
    logger.info("  - data/processed/all_sentiment.csv")
    logger.info("  - data/recommendations/latest_recommendations.csv")

    # ==================== STEP 7: PRINT SUMMARY ====================
    print_summary(recommendations, combined_sentiment, recommender)

    return recommendations, combined_sentiment, stock_df


def print_summary(recommendations, sentiment_df, recommender):
    """Print analysis summary with recommendations"""
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    # Overall stats
    print("\nDATA COLLECTED:")
    print(f"  Total content analyzed: {len(sentiment_df)}")
    if 'content_type' in sentiment_df.columns:
        for ctype, count in sentiment_df['content_type'].value_counts().items():
            print(f"    - {ctype.capitalize()}: {count}")

    # Sentiment overview
    print("\nOVERALL SENTIMENT:")
    print(f"  Average: {sentiment_df['sentiment_score'].mean():.3f}")
    print(f"  Distribution:")
    for label, count in sentiment_df['sentiment_label'].value_counts().items():
        pct = count / len(sentiment_df) * 100
        print(f"    - {label.capitalize()}: {count} ({pct:.1f}%)")

    # Recommendations
    print("\n" + "=" * 70)
    print("STOCK RECOMMENDATIONS")
    print("=" * 70)

    # Top Picks
    top_picks = recommender.get_top_picks(recommendations, n=10)
    if not top_picks.empty:
        print("\nTOP PICKS (BUY):")
        print("-" * 50)
        for idx, row in top_picks.iterrows():
            confidence_bar = "*" * int(row['confidence'] * 10)
            print(f"  {row['ticker']:6} | {row['recommendation']:12} | "
                  f"Confidence: {row['confidence']:.0%} {confidence_bar}")
            print(f"         | Sentiment: {row['sentiment_score']:+.3f} | "
                  f"Articles: {row['num_articles']}")

    # Stocks to Avoid
    avoid = recommender.get_stocks_to_avoid(recommendations, n=5)
    if not avoid.empty:
        print("\nSTOCKS TO AVOID (SELL):")
        print("-" * 50)
        for idx, row in avoid.iterrows():
            print(f"  {row['ticker']:6} | {row['recommendation']:12} | "
                  f"Sentiment: {row['sentiment_score']:+.3f}")

    # All recommendations summary
    print("\nRECOMMENDATION BREAKDOWN:")
    print("-" * 50)
    rec_counts = recommendations['recommendation'].value_counts()
    for rec, count in rec_counts.items():
        print(f"  {rec}: {count} stocks")

    # Historical performance (if available)
    try:
        stock_df = pd.read_csv('data/processed/stock_prices.csv')
        perf = recommender.calculate_historical_performance(stock_df)
        if not perf.empty:
            summary = recommender.get_performance_summary(perf)
            print("\nHISTORICAL PERFORMANCE:")
            print("-" * 50)
            print(f"  Total past recommendations: {summary['total_recommendations']}")
            print(f"  Accuracy: {summary['accuracy_pct']}%")
            print(f"  Avg theoretical return: {summary['avg_theoretical_return_pct']}%")
    except Exception:
        pass

    print("\n" + "=" * 70)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Stock Sentiment Analysis with Recommendations')
    parser.add_argument('--list', type=str, default='original',
                        choices=list(TICKER_LISTS.keys()),
                        help='Ticker list to analyze')
    parser.add_argument('--days', type=int, default=7,
                        help='Days of historical data')

    args = parser.parse_args()

    tickers = get_tickers(args.list)

    print(f"\nUsing ticker list: {args.list} ({len(tickers)} tickers)")
    print(f"Tickers: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")

    result = run_full_analysis(tickers, args.days)

    if result:
        print("\nANALYSIS COMPLETE!")
        print("\nFiles saved:")
        print("  - data/recommendations/latest_recommendations.csv")
        print("  - data/recommendations/history.json")
        print("\nRun dashboard to visualize:")
        print("  streamlit run src/dashboard/app.py")
    else:
        print("\nAnalysis failed!")


if __name__ == "__main__":
    main()
