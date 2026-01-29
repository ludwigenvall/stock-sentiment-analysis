"""
SEC EDGAR Filings Collector

Collects SEC filings (10-K, 10-Q, 8-K) for sentiment analysis.
Uses the free SEC EDGAR API - no API key required.
"""

import requests
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import re

logger = logging.getLogger(__name__)


class SECCollector:
    """Collector for SEC EDGAR filings."""

    BASE_URL = "https://data.sec.gov"
    SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

    # Mapping of common tickers to CIK numbers
    TICKER_TO_CIK = {
        'AAPL': '0000320193',
        'MSFT': '0000789019',
        'GOOGL': '0001652044',
        'GOOG': '0001652044',
        'META': '0001326801',
        'AMZN': '0001018724',
        'NVDA': '0001045810',
        'TSLA': '0001318605',
        'JPM': '0000019617',
        'V': '0001403161',
        'JNJ': '0000200406',
        'WMT': '0000104169',
        'PG': '0000080424',
        'MA': '0001141391',
        'UNH': '0000731766',
        'HD': '0000354950',
        'DIS': '0001744489',
        'BAC': '0000070858',
        'ADBE': '0000796343',
        'CRM': '0001108524',
        'NFLX': '0001065280',
        'INTC': '0000050863',
        'AMD': '0000002488',
        'PYPL': '0001633917',
        'CSCO': '0000858877',
        'PEP': '0000077476',
        'KO': '0000021344',
        'COST': '0000909832',
        'AVGO': '0001730168',
        'ORCL': '0001341439',
        'ACN': '0001467373',
        'IBM': '0000051143',
        'QCOM': '0000804328',
        'TXN': '0000097476',
        'ABBV': '0001551152',
        'MRK': '0000310158',
        'PFE': '0000078003',
        'LLY': '0000059478',
        'TMO': '0000097745',
        'ABT': '0000001800',
        'DHR': '0000313616',
        'BMY': '0000014272',
        'AMGN': '0000318154',
        'CVX': '0000093410',
        'XOM': '0000034088',
        'COP': '0001163165',
        'SLB': '0000087347',
        'EOG': '0000821189',
        'CAT': '0000018230',
        'BA': '0000012927',
        'HON': '0000773840',
        'GE': '0000040545',
        'MMM': '0000066740',
        'UPS': '0001090727',
        'RTX': '0000101829',
        'LMT': '0000936468',
        'DE': '0000315189',
        'GS': '0000886982',
        'MS': '0000895421',
        'C': '0000831001',
        'BLK': '0001364742',
        'SCHW': '0000316709',
        'AXP': '0000004962',
        'USB': '0000036104',
        'PNC': '0000713676',
        'COF': '0000927628',
        'WFC': '0000072971',
        'T': '0000732717',
        'VZ': '0000732712',
        'TMUS': '0001283699',
        'CMCSA': '0001166691',
        'CHTR': '0001091667',
        'NKE': '0000320187',
        'SBUX': '0000829224',
        'MCD': '0000063908',
        'LOW': '0000060667',
        'TGT': '0000027419',
        'CVS': '0000064803',
        'CI': '0001739940',
        'MDT': '0001613103',
        'GILD': '0000882095',
        'MU': '0000723125',
        'AMAT': '0000006951',
        'ADI': '0000006281',
        'MPC': '0001510295',
        'VLO': '0001035002',
        'PSX': '0001534701',
        'OXY': '0000797468',
        'EMR': '0000032604',
    }

    def __init__(self, user_agent: str = None):
        """
        Initialize SEC Collector.

        Args:
            user_agent: User agent string (SEC requires identifying info)
        """
        self.user_agent = user_agent or "StockSentimentAnalysis/1.0 (contact@example.com)"
        self.headers = {
            'User-Agent': self.user_agent,
            'Accept': 'application/json',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        logger.info("SEC Collector initialized")

    def _get_cik(self, ticker: str) -> Optional[str]:
        """Get CIK number for a ticker."""
        ticker = ticker.upper()

        # Check cache first
        if ticker in self.TICKER_TO_CIK:
            return self.TICKER_TO_CIK[ticker]

        # Try to fetch from SEC
        try:
            url = f"{self.BASE_URL}/cgi-bin/browse-edgar"
            params = {
                'action': 'getcompany',
                'CIK': ticker,
                'type': '',
                'dateb': '',
                'owner': 'include',
                'count': '1',
                'output': 'json'
            }
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and len(data['results']) > 0:
                    cik = data['results'][0].get('cik')
                    if cik:
                        self.TICKER_TO_CIK[ticker] = cik
                        return cik
        except Exception as e:
            logger.debug(f"Could not fetch CIK for {ticker}: {e}")

        return None

    def get_filings(
        self,
        ticker: str,
        filing_types: List[str] = None,
        limit: int = 5,
        days_back: int = 365
    ) -> pd.DataFrame:
        """
        Get SEC filings for a ticker.

        Args:
            ticker: Stock ticker symbol
            filing_types: Types of filings to fetch (default: ['10-K', '10-Q', '8-K'])
            limit: Maximum number of filings to return
            days_back: How far back to look for filings

        Returns:
            DataFrame with filing information
        """
        if filing_types is None:
            filing_types = ['10-K', '10-Q', '8-K']

        ticker = ticker.upper()
        cik = self._get_cik(ticker)

        if not cik:
            logger.warning(f"Could not find CIK for {ticker}")
            return pd.DataFrame()

        filings = []

        try:
            # Fetch company submissions
            cik_padded = cik.zfill(10)
            url = f"{self.BASE_URL}/submissions/CIK{cik_padded}.json"

            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()

            # Get recent filings
            recent = data.get('filings', {}).get('recent', {})
            if not recent:
                logger.warning(f"No filings found for {ticker}")
                return pd.DataFrame()

            forms = recent.get('form', [])
            dates = recent.get('filingDate', [])
            accessions = recent.get('accessionNumber', [])
            descriptions = recent.get('primaryDocDescription', [])

            cutoff_date = datetime.now() - timedelta(days=days_back)

            for i, (form, date_str, accession, desc) in enumerate(zip(forms, dates, accessions, descriptions)):
                if len(filings) >= limit:
                    break

                # Filter by filing type
                if form not in filing_types:
                    continue

                # Filter by date
                try:
                    filing_date = datetime.strptime(date_str, '%Y-%m-%d')
                    if filing_date < cutoff_date:
                        continue
                except ValueError:
                    continue

                # Get filing text summary
                accession_clean = accession.replace('-', '')
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/{accession}-index.htm"

                # Create summary text for sentiment analysis
                company_name = data.get('name', ticker)
                text = self._create_filing_summary(form, company_name, desc, date_str)

                filings.append({
                    'ticker': ticker,
                    'filing_type': form,
                    'date_filed': filing_date,
                    'title': f"{ticker} {form} Filing - {date_str}",
                    'text': text,
                    'description': desc or f"{form} filing",
                    'url': filing_url,
                    'accession_number': accession,
                    'source': 'sec_edgar'
                })

            time.sleep(0.1)  # Rate limiting

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching SEC filings for {ticker}: {e}")
        except Exception as e:
            logger.error(f"Error processing SEC filings for {ticker}: {e}")

        return pd.DataFrame(filings)

    def _create_filing_summary(
        self,
        form: str,
        company_name: str,
        description: str,
        date: str
    ) -> str:
        """Create a text summary of the filing for sentiment analysis."""
        form_descriptions = {
            '10-K': 'annual report with comprehensive overview of business and financial condition',
            '10-Q': 'quarterly report with unaudited financial statements',
            '8-K': 'current report announcing major events or corporate changes',
            '4': 'statement of changes in beneficial ownership',
            'DEF 14A': 'proxy statement for annual meeting',
            'S-1': 'registration statement for initial public offering',
            '13F': 'quarterly report of institutional investment managers',
        }

        form_desc = form_descriptions.get(form, f'{form} regulatory filing')
        desc_text = f" - {description}" if description else ""

        return f"{company_name} filed {form} ({form_desc}) on {date}{desc_text}."

    def get_filing_text(self, accession_number: str, cik: str) -> Optional[str]:
        """
        Get the full text of a specific filing.

        Note: This can be slow and returns large amounts of text.
        Consider using only for important filings.
        """
        try:
            accession_clean = accession_number.replace('-', '')
            url = f"{self.BASE_URL}/Archives/edgar/data/{cik}/{accession_clean}"

            # Get the filing index
            response = self.session.get(f"{url}/index.json", timeout=15)
            response.raise_for_status()

            index_data = response.json()
            directory = index_data.get('directory', {})
            items = directory.get('item', [])

            # Find the main document (usually .htm or .txt)
            main_doc = None
            for item in items:
                name = item.get('name', '')
                if name.endswith('.htm') and not name.endswith('-index.htm'):
                    main_doc = name
                    break

            if main_doc:
                doc_response = self.session.get(f"{url}/{main_doc}", timeout=30)
                doc_response.raise_for_status()

                # Extract text from HTML
                text = self._extract_text_from_html(doc_response.text)
                return text[:5000]  # Limit text length

        except Exception as e:
            logger.error(f"Error fetching filing text: {e}")

        return None

    def _extract_text_from_html(self, html: str) -> str:
        """Extract plain text from HTML content."""
        # Remove script and style elements
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    def collect_for_tickers(
        self,
        tickers: List[str],
        filing_types: List[str] = None,
        limit_per_ticker: int = 3,
        days_back: int = 365
    ) -> pd.DataFrame:
        """
        Collect SEC filings for multiple tickers.

        Args:
            tickers: List of ticker symbols
            filing_types: Types of filings to fetch
            limit_per_ticker: Max filings per ticker
            days_back: How far back to look

        Returns:
            Combined DataFrame with all filings
        """
        if filing_types is None:
            filing_types = ['10-K', '10-Q', '8-K']

        all_filings = []

        logger.info(f"Collecting SEC filings for {len(tickers)} tickers...")

        for i, ticker in enumerate(tickers):
            try:
                df = self.get_filings(
                    ticker,
                    filing_types=filing_types,
                    limit=limit_per_ticker,
                    days_back=days_back
                )

                if not df.empty:
                    all_filings.append(df)
                    logger.info(f"  [{i+1}/{len(tickers)}] {ticker}: {len(df)} filings")
                else:
                    logger.debug(f"  [{i+1}/{len(tickers)}] {ticker}: No filings found")

                # Rate limiting - SEC asks for max 10 requests/second
                time.sleep(0.15)

            except Exception as e:
                logger.error(f"Error collecting filings for {ticker}: {e}")
                continue

        if all_filings:
            combined = pd.concat(all_filings, ignore_index=True)

            # Remove duplicates
            combined = combined.drop_duplicates(subset=['accession_number'])

            # Sort by date
            combined = combined.sort_values('date_filed', ascending=False)

            logger.info(f"Collected {len(combined)} total SEC filings")
            return combined

        return pd.DataFrame()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    collector = SECCollector()

    # Test with a few tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']

    print("Testing SEC Collector...")
    print("-" * 50)

    for ticker in test_tickers:
        df = collector.get_filings(ticker, limit=3)
        print(f"\n{ticker}: {len(df)} filings")
        if not df.empty:
            print(df[['filing_type', 'date_filed', 'title']].to_string(index=False))

    print("\n" + "-" * 50)
    print("Batch collection test:")
    combined = collector.collect_for_tickers(test_tickers, limit_per_ticker=2)
    print(f"\nTotal filings: {len(combined)}")
    if not combined.empty:
        print(combined[['ticker', 'filing_type', 'date_filed']].head(10).to_string(index=False))
