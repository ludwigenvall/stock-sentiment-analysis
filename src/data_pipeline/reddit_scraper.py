"""
Reddit scraper for collecting stock-related posts and comments
"""
import logging
import os
import re
from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd
import praw
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedditScraper:
    """Scrape Reddit for stock mentions and sentiment"""

    def __init__(self):
        """Initialize Reddit API client"""
        self.client_id = os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.user_agent = os.getenv('REDDIT_USER_AGENT', 'sentiment_tracker_v1')

        if not self.client_id or not self.client_secret:
            logger.warning("Reddit API credentials not found in .env file")
            self.reddit = None
            return

        try:
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent
            )
            # Test connection
            self.reddit.user.me()
            logger.info("Reddit API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Reddit API: {e}")
            self.reddit = None

    def extract_tickers(self, text: str, valid_tickers: List[str]) -> List[str]:
        """
        Extract stock tickers from text

        Args:
            text: Text to search for tickers
            valid_tickers: List of valid ticker symbols to look for

        Returns:
            List of found tickers
        """
        if not text:
            return []

        # Look for $TICKER or ticker mentions
        found_tickers = []
        text_upper = text.upper()

        for ticker in valid_tickers:
            # Match $TICKER or TICKER as whole word
            pattern = r'\$' + ticker + r'\b|\b' + ticker + r'\b'
            if re.search(pattern, text_upper):
                found_tickers.append(ticker)

        return list(set(found_tickers))

    def scrape_subreddit(
        self,
        subreddit_name: str,
        tickers: List[str],
        limit: int = 100,
        time_filter: str = 'week'
    ) -> pd.DataFrame:
        """
        Scrape posts from a subreddit

        Args:
            subreddit_name: Name of subreddit (e.g., 'wallstreetbets')
            tickers: List of tickers to search for
            limit: Maximum number of posts to fetch
            time_filter: Time filter ('hour', 'day', 'week', 'month', 'year', 'all')

        Returns:
            DataFrame with columns: title, text, score, num_comments, created_utc,
                                   url, ticker, source
        """
        if not self.reddit:
            logger.error("Reddit API not initialized")
            return pd.DataFrame()

        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts = []

            # Get hot posts
            for submission in subreddit.hot(limit=limit):
                # Skip stickied posts
                if submission.stickied:
                    continue

                # Combine title and selftext for ticker detection
                full_text = f"{submission.title} {submission.selftext}"
                found_tickers = self.extract_tickers(full_text, tickers)

                if found_tickers:
                    # Create entry for each ticker mentioned
                    for ticker in found_tickers:
                        posts.append({
                            'title': submission.title,
                            'text': submission.selftext[:500] if submission.selftext else '',  # Limit text length
                            'score': submission.score,
                            'num_comments': submission.num_comments,
                            'created_utc': datetime.fromtimestamp(submission.created_utc),
                            'url': f"https://reddit.com{submission.permalink}",
                            'ticker': ticker,
                            'source': f'r/{subreddit_name}'
                        })

            logger.info(f"r/{subreddit_name}: Found {len(posts)} posts mentioning target tickers")
            return pd.DataFrame(posts)

        except Exception as e:
            logger.error(f"Error scraping r/{subreddit_name}: {e}")
            return pd.DataFrame()

    def scrape_multiple_subreddits(
        self,
        tickers: List[str],
        subreddits: Optional[List[str]] = None,
        limit_per_sub: int = 100
    ) -> pd.DataFrame:
        """
        Scrape multiple subreddits for ticker mentions

        Args:
            tickers: List of stock tickers to search for
            subreddits: List of subreddit names (default: popular stock subreddits)
            limit_per_sub: Max posts to fetch per subreddit

        Returns:
            Combined DataFrame from all subreddits
        """
        if not self.reddit:
            logger.error("Reddit API not initialized. Check your credentials in .env file")
            return pd.DataFrame()

        if subreddits is None:
            subreddits = [
                'wallstreetbets',
                'stocks',
                'investing',
                'StockMarket',
                'options'
            ]

        all_posts = []

        for subreddit in subreddits:
            logger.info(f"Scraping r/{subreddit}...")
            df = self.scrape_subreddit(subreddit, tickers, limit=limit_per_sub)
            if not df.empty:
                all_posts.append(df)

        if not all_posts:
            logger.warning("No posts found across all subreddits")
            return pd.DataFrame()

        # Combine all dataframes
        combined_df = pd.concat(all_posts, ignore_index=True)

        # Remove duplicates (same URL)
        combined_df = combined_df.drop_duplicates(subset=['url'])

        # Sort by created time
        combined_df = combined_df.sort_values('created_utc', ascending=False)

        logger.info(f"Total unique posts found: {len(combined_df)}")

        return combined_df

    def search_ticker_mentions(
        self,
        ticker: str,
        subreddits: Optional[List[str]] = None,
        limit: int = 50
    ) -> pd.DataFrame:
        """
        Search for specific ticker mentions across subreddits

        Args:
            ticker: Stock ticker to search for
            subreddits: List of subreddit names
            limit: Max results per subreddit

        Returns:
            DataFrame with posts mentioning the ticker
        """
        if not self.reddit:
            logger.error("Reddit API not initialized")
            return pd.DataFrame()

        if subreddits is None:
            subreddits = ['wallstreetbets', 'stocks', 'investing']

        all_posts = []

        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)

                # Search for ticker
                search_query = f"${ticker} OR {ticker}"

                for submission in subreddit.search(search_query, limit=limit, time_filter='week'):
                    all_posts.append({
                        'title': submission.title,
                        'text': submission.selftext[:500] if submission.selftext else '',
                        'score': submission.score,
                        'num_comments': submission.num_comments,
                        'created_utc': datetime.fromtimestamp(submission.created_utc),
                        'url': f"https://reddit.com{submission.permalink}",
                        'ticker': ticker,
                        'source': f'r/{subreddit_name}'
                    })

                logger.info(f"r/{subreddit_name}: Found {len(all_posts)} posts for ${ticker}")

            except Exception as e:
                logger.error(f"Error searching r/{subreddit_name}: {e}")

        if not all_posts:
            return pd.DataFrame()

        df = pd.DataFrame(all_posts)
        df = df.drop_duplicates(subset=['url'])
        df = df.sort_values('created_utc', ascending=False)

        return df


def main():
    """Test the Reddit scraper"""
    scraper = RedditScraper()

    if not scraper.reddit:
        print("\nReddit API not configured.")
        print("Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env file")
        print("\nTo get credentials:")
        print("1. Go to https://www.reddit.com/prefs/apps")
        print("2. Create a new app (script type)")
        print("3. Add credentials to .env file")
        return

    # Test with popular tickers
    tickers = ['AAPL', 'TSLA', 'GME', 'NVDA', 'MSFT']

    print(f"\nSearching Reddit for mentions of: {', '.join(tickers)}")
    print("This may take a minute...\n")

    df = scraper.scrape_multiple_subreddits(
        tickers=tickers,
        limit_per_sub=50
    )

    if not df.empty:
        print(f"\nFound {len(df)} posts mentioning target tickers")
        print("\nBreakdown by ticker:")
        print(df['ticker'].value_counts())

        print("\nBreakdown by subreddit:")
        print(df['source'].value_counts())

        print("\nTop 5 posts by score:")
        top_posts = df.nlargest(5, 'score')[['ticker', 'title', 'score', 'source']]
        for idx, row in top_posts.iterrows():
            print(f"\n[{row['ticker']}] {row['title']}")
            print(f"   Score: {row['score']} | Source: {row['source']}")

        # Save to CSV
        output_file = 'data/raw/reddit_posts.csv'
        df.to_csv(output_file, index=False)
        print(f"\nSaved results to: {output_file}")
    else:
        print("No posts found. Try different tickers or check your API credentials.")


if __name__ == "__main__":
    main()