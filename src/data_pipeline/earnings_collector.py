"""
Earnings Call Transcript Collector

Collects earnings call transcripts for sentiment analysis.
Uses Financial Modeling Prep API (free tier available).

Set FMP_API_KEY in your .env file.
Get a free API key at: https://financialmodelingprep.com/developer/docs/
"""

import os
import requests
import pandas as pd
import logging
import time
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class EarningsCollector:
    """Collector for earnings call transcripts."""

    BASE_URL = "https://financialmodelingprep.com/api"

    def __init__(self, api_key: str = None):
        """
        Initialize Earnings Collector.

        Args:
            api_key: Financial Modeling Prep API key (or set FMP_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('FMP_API_KEY')
        self.is_configured = bool(self.api_key)

        if self.is_configured:
            logger.info("Earnings Collector initialized with API key")
        else:
            logger.warning(
                "Earnings Collector: FMP_API_KEY not set. "
                "Get a free key at https://financialmodelingprep.com/developer/docs/"
            )

    def get_transcripts(
        self,
        ticker: str,
        limit: int = 4,
        year: int = None
    ) -> pd.DataFrame:
        """
        Get earnings call transcripts for a ticker.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of transcripts to return
            year: Specific year to fetch (optional)

        Returns:
            DataFrame with transcript data
        """
        if not self.is_configured:
            logger.warning("Earnings API not configured, returning empty DataFrame")
            return pd.DataFrame()

        ticker = ticker.upper()
        transcripts = []

        try:
            # Get available transcripts
            url = f"{self.BASE_URL}/v4/earning_call_transcript"
            params = {
                'symbol': ticker,
                'apikey': self.api_key
            }

            if year:
                params['year'] = year

            response = requests.get(url, params=params, timeout=15)

            if response.status_code == 401:
                logger.error("Invalid FMP API key")
                self.is_configured = False
                return pd.DataFrame()

            if response.status_code == 429:
                logger.warning("FMP API rate limit reached")
                return pd.DataFrame()

            response.raise_for_status()
            data = response.json()

            if not data or isinstance(data, dict) and 'Error Message' in data:
                logger.debug(f"No transcripts found for {ticker}")
                return pd.DataFrame()

            # Process transcripts
            for item in data[:limit]:
                # Parse date
                date_str = item.get('date', '')
                try:
                    if date_str:
                        transcript_date = datetime.strptime(date_str.split()[0], '%Y-%m-%d')
                    else:
                        transcript_date = datetime.now()
                except ValueError:
                    transcript_date = datetime.now()

                quarter = item.get('quarter', 0)
                year = item.get('year', datetime.now().year)
                content = item.get('content', '')

                # Create summary for sentiment analysis (full transcripts are very long)
                summary = self._create_summary(content, ticker, quarter, year)

                transcripts.append({
                    'ticker': ticker,
                    'date': transcript_date,
                    'year': year,
                    'quarter': f"Q{quarter}",
                    'title': f"{ticker} Q{quarter} {year} Earnings Call",
                    'text': summary,
                    'full_content_length': len(content),
                    'source': 'earnings_call'
                })

            logger.info(f"Found {len(transcripts)} earnings transcripts for {ticker}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching earnings transcripts for {ticker}: {e}")
        except Exception as e:
            logger.error(f"Error processing earnings data for {ticker}: {e}")

        return pd.DataFrame(transcripts)

    def _create_summary(
        self,
        content: str,
        ticker: str,
        quarter: int,
        year: int,
        max_length: int = 1000
    ) -> str:
        """
        Create a summary of the earnings call for sentiment analysis.

        Extracts key sections: opening remarks, guidance, Q&A highlights.
        """
        if not content:
            return f"{ticker} Q{quarter} {year} earnings call transcript."

        # Clean content
        content = content.strip()

        # Try to extract key sections
        summary_parts = []

        # Look for CEO/CFO remarks (usually at the beginning)
        lines = content.split('\n')
        intro_lines = []
        for line in lines[:30]:  # First 30 lines usually contain intro
            line = line.strip()
            if len(line) > 50:  # Substantial content
                intro_lines.append(line)
            if len(' '.join(intro_lines)) > 400:
                break

        if intro_lines:
            summary_parts.append(' '.join(intro_lines[:3]))

        # Look for guidance keywords
        guidance_keywords = [
            'guidance', 'outlook', 'expect', 'forecast',
            'revenue growth', 'margin', 'profitability'
        ]

        content_lower = content.lower()
        for keyword in guidance_keywords:
            idx = content_lower.find(keyword)
            if idx > 0:
                # Get surrounding context
                start = max(0, idx - 50)
                end = min(len(content), idx + 200)
                context = content[start:end].strip()
                if context and context not in summary_parts:
                    summary_parts.append(f"...{context}...")
                    break

        # Combine and truncate
        summary = ' '.join(summary_parts)
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."

        if not summary:
            summary = content[:max_length] + "..." if len(content) > max_length else content

        return summary or f"{ticker} Q{quarter} {year} earnings call transcript."

    def get_earnings_calendar(
        self,
        from_date: str = None,
        to_date: str = None
    ) -> pd.DataFrame:
        """
        Get upcoming earnings dates.

        Args:
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with earnings calendar
        """
        if not self.is_configured:
            return pd.DataFrame()

        try:
            url = f"{self.BASE_URL}/v3/earning_calendar"
            params = {'apikey': self.api_key}

            if from_date:
                params['from'] = from_date
            if to_date:
                params['to'] = to_date

            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            if data:
                df = pd.DataFrame(data)
                return df

        except Exception as e:
            logger.error(f"Error fetching earnings calendar: {e}")

        return pd.DataFrame()

    def collect_for_tickers(
        self,
        tickers: List[str],
        limit_per_ticker: int = 2,
        year: int = None
    ) -> pd.DataFrame:
        """
        Collect earnings transcripts for multiple tickers.

        Args:
            tickers: List of ticker symbols
            limit_per_ticker: Max transcripts per ticker
            year: Specific year to fetch (optional)

        Returns:
            Combined DataFrame with all transcripts
        """
        if not self.is_configured:
            logger.warning("Earnings API not configured, returning empty DataFrame")
            return pd.DataFrame()

        all_transcripts = []

        logger.info(f"Collecting earnings transcripts for {len(tickers)} tickers...")

        for i, ticker in enumerate(tickers):
            try:
                df = self.get_transcripts(
                    ticker,
                    limit=limit_per_ticker,
                    year=year
                )

                if not df.empty:
                    all_transcripts.append(df)
                    logger.info(f"  [{i+1}/{len(tickers)}] {ticker}: {len(df)} transcripts")
                else:
                    logger.debug(f"  [{i+1}/{len(tickers)}] {ticker}: No transcripts")

                # Rate limiting for free tier
                time.sleep(0.3)

            except Exception as e:
                logger.error(f"Error collecting transcripts for {ticker}: {e}")
                continue

        if all_transcripts:
            combined = pd.concat(all_transcripts, ignore_index=True)

            # Sort by date
            combined = combined.sort_values('date', ascending=False)

            logger.info(f"Collected {len(combined)} total earnings transcripts")
            return combined

        return pd.DataFrame()


# Alternative: Earnings data from Yahoo Finance (limited but free)
class YahooEarningsCollector:
    """
    Alternative earnings collector using yfinance.
    Provides earnings dates and basic info, but not full transcripts.
    """

    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
            self.is_configured = True
            logger.info("Yahoo Earnings Collector initialized")
        except ImportError:
            self.yf = None
            self.is_configured = False
            logger.warning("yfinance not installed")

    def get_earnings_history(self, ticker: str) -> pd.DataFrame:
        """Get earnings history for a ticker."""
        if not self.is_configured:
            return pd.DataFrame()

        try:
            stock = self.yf.Ticker(ticker)

            # Suppress deprecation warnings
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                earnings = stock.earnings_history

            if earnings is not None and not earnings.empty:
                earnings = earnings.reset_index()
                earnings['ticker'] = ticker
                earnings['source'] = 'yahoo_earnings'

                # Create date from quarter column
                if 'quarter' in earnings.columns:
                    earnings['date'] = pd.to_datetime(earnings['quarter'])

                return earnings

        except Exception as e:
            logger.debug(f"Error fetching earnings for {ticker}: {e}")

        return pd.DataFrame()

    def get_earnings_dates(self, ticker: str) -> pd.DataFrame:
        """Get upcoming and past earnings dates."""
        if not self.is_configured:
            return pd.DataFrame()

        try:
            stock = self.yf.Ticker(ticker)
            dates = stock.earnings_dates

            if dates is not None and not dates.empty:
                dates['ticker'] = ticker
                return dates.reset_index()

        except Exception as e:
            logger.debug(f"Could not fetch earnings dates for {ticker}: {e}")

        return pd.DataFrame()

    def collect_for_tickers(self, tickers: List[str], **kwargs) -> pd.DataFrame:
        """Collect earnings data for multiple tickers."""
        all_data = []

        logger.info(f"Collecting Yahoo earnings for {len(tickers)} tickers...")

        for i, ticker in enumerate(tickers):
            df = self.get_earnings_history(ticker)
            if not df.empty:
                # Create text for sentiment analysis
                df['title'] = df.apply(
                    lambda x: f"{ticker} Earnings: EPS {x.get('epsActual', 'N/A')} vs {x.get('epsEstimate', 'N/A')} expected",
                    axis=1
                )
                df['text'] = df.apply(
                    lambda x: self._create_earnings_text(ticker, x),
                    axis=1
                )
                all_data.append(df)
                logger.debug(f"  [{i+1}/{len(tickers)}] {ticker}: {len(df)} earnings records")

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            logger.info(f"Collected {len(combined)} total earnings records")
            return combined

        return pd.DataFrame()

    def _create_earnings_text(self, ticker: str, row) -> str:
        """Create text summary from earnings data."""
        eps_actual = row.get('epsActual', 'N/A')
        eps_estimate = row.get('epsEstimate', 'N/A')
        surprise = row.get('epsSurprise', 0) or 0

        if surprise > 0:
            sentiment = "beat expectations"
        elif surprise < 0:
            sentiment = "missed expectations"
        else:
            sentiment = "met expectations"

        return (
            f"{ticker} reported earnings per share of {eps_actual}, "
            f"compared to analyst estimate of {eps_estimate}. "
            f"The company {sentiment} with a surprise of {surprise:.2%}."
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Try FMP collector first
    collector = EarningsCollector()

    test_tickers = ['AAPL', 'MSFT', 'GOOGL']

    print("Testing Earnings Collector...")
    print("-" * 50)

    if collector.is_configured:
        for ticker in test_tickers:
            df = collector.get_transcripts(ticker, limit=2)
            print(f"\n{ticker}: {len(df)} transcripts")
            if not df.empty:
                print(df[['quarter', 'year', 'title']].to_string(index=False))

        print("\n" + "-" * 50)
        print("Batch collection test:")
        combined = collector.collect_for_tickers(test_tickers, limit_per_ticker=1)
        print(f"\nTotal transcripts: {len(combined)}")
    else:
        print("\nFMP API not configured. Testing Yahoo fallback...")
        yahoo_collector = YahooEarningsCollector()

        if yahoo_collector.is_configured:
            for ticker in test_tickers:
                df = yahoo_collector.get_earnings_dates(ticker)
                print(f"\n{ticker}: {len(df)} earnings dates")
                if not df.empty:
                    print(df.head(3).to_string(index=False))
