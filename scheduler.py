"""
Automated Scheduler for Stock Sentiment Analysis

Runs the analysis pipeline on a schedule and generates recommendations.
"""
import sys
from pathlib import Path
import schedule
import time
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config.tickers import get_tickers, TICKER_LISTS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_daily_analysis():
    """Run the full analysis pipeline"""
    logger.info("=" * 60)
    logger.info("STARTING SCHEDULED ANALYSIS")
    logger.info("=" * 60)

    try:
        from analyze_with_recommendations import run_full_analysis

        # Get tickers to analyze
        tickers = get_tickers('sp100')  # Use S&P 100 list

        logger.info(f"Analyzing {len(tickers)} tickers")

        result = run_full_analysis(tickers, days_back=7)

        if result:
            logger.info("Analysis completed successfully")
        else:
            logger.error("Analysis failed")

    except Exception as e:
        logger.error(f"Error running analysis: {e}")

    logger.info("=" * 60)


def run_market_hours_update():
    """Quick update during market hours (less comprehensive)"""
    logger.info("Running market hours update...")

    try:
        from data_pipeline.news_scraper import NewsCollector
        from config.tickers import get_tickers

        # Quick news fetch for top tickers only
        tickers = get_tickers('top_tech')
        collector = NewsCollector()
        news_df = collector.get_news_for_multiple_tickers(tickers, limit_per_ticker=10)

        if not news_df.empty:
            logger.info(f"Fetched {len(news_df)} new articles")
        else:
            logger.warning("No new articles found")

    except Exception as e:
        logger.error(f"Error in market hours update: {e}")


def setup_schedule():
    """Configure the scheduler"""
    # Daily full analysis at 6:00 AM (before market opens)
    schedule.every().day.at("06:00").do(run_daily_analysis)

    # Quick updates during market hours (9:30 AM - 4:00 PM ET)
    schedule.every().monday.at("09:30").do(run_market_hours_update)
    schedule.every().monday.at("12:00").do(run_market_hours_update)
    schedule.every().monday.at("15:00").do(run_market_hours_update)

    schedule.every().tuesday.at("09:30").do(run_market_hours_update)
    schedule.every().tuesday.at("12:00").do(run_market_hours_update)
    schedule.every().tuesday.at("15:00").do(run_market_hours_update)

    schedule.every().wednesday.at("09:30").do(run_market_hours_update)
    schedule.every().wednesday.at("12:00").do(run_market_hours_update)
    schedule.every().wednesday.at("15:00").do(run_market_hours_update)

    schedule.every().thursday.at("09:30").do(run_market_hours_update)
    schedule.every().thursday.at("12:00").do(run_market_hours_update)
    schedule.every().thursday.at("15:00").do(run_market_hours_update)

    schedule.every().friday.at("09:30").do(run_market_hours_update)
    schedule.every().friday.at("12:00").do(run_market_hours_update)
    schedule.every().friday.at("15:00").do(run_market_hours_update)

    logger.info("Schedule configured:")
    logger.info("  - Daily full analysis at 06:00")
    logger.info("  - Market hours updates at 09:30, 12:00, 15:00 (Mon-Fri)")


def main():
    """Main entry point for scheduler"""
    # Create logs directory
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    Path("data/recommendations").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STOCK SENTIMENT ANALYSIS SCHEDULER")
    print("=" * 60)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nOptions:")
    print("  1. Run now (single analysis)")
    print("  2. Start scheduler (continuous)")
    print("  3. Exit")

    choice = input("\nSelect option (1/2/3): ").strip()

    if choice == "1":
        print("\nRunning analysis now...")
        run_daily_analysis()
        print("\nDone!")

    elif choice == "2":
        print("\nStarting scheduler...")
        setup_schedule()
        print("\nScheduler is running. Press Ctrl+C to stop.")

        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            print("\n\nScheduler stopped.")

    else:
        print("Exiting.")


if __name__ == "__main__":
    main()
