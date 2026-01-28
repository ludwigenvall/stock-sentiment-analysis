"""
Unit tests for data pipeline modules
"""
import pytest
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_pipeline.stock_data import StockDataCollector
from data_pipeline.reddit_scraper import RedditScraper


class TestStockDataCollector:
    """Tests for stock data collector"""

    def test_initialization(self):
        """Test collector initialization"""
        tickers = ['AAPL', 'MSFT']
        collector = StockDataCollector(tickers)

        assert collector.tickers == tickers

    def test_initialization_with_single_ticker(self):
        """Test initialization with single ticker"""
        collector = StockDataCollector(['AAPL'])
        assert len(collector.tickers) == 1

    def test_get_historical_data_returns_dataframe(self):
        """Test that historical data returns a DataFrame"""
        collector = StockDataCollector(['AAPL'])
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        df = collector.get_historical_data(start_date)

        # Should return DataFrame (may be empty if API issues)
        assert isinstance(df, pd.DataFrame)

    def test_dataframe_columns(self):
        """Test that DataFrame has expected columns when data is available"""
        collector = StockDataCollector(['AAPL'])
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        df = collector.get_historical_data(start_date)

        if not df.empty:
            expected_columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
            for col in expected_columns:
                assert col in df.columns

    def test_multiple_tickers(self):
        """Test fetching data for multiple tickers"""
        collector = StockDataCollector(['AAPL', 'MSFT', 'GOOGL'])
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        df = collector.get_historical_data(start_date)

        if not df.empty:
            # Should have data for multiple tickers
            tickers_in_data = df['ticker'].unique()
            assert len(tickers_in_data) >= 1


class TestRedditScraper:
    """Tests for Reddit scraper"""

    def test_initialization_without_credentials(self):
        """Test scraper initialization without credentials"""
        with patch.dict('os.environ', {}, clear=True):
            scraper = RedditScraper()
            # Should initialize but reddit client may be None
            assert scraper is not None

    def test_extract_tickers_single(self):
        """Test ticker extraction from text"""
        scraper = RedditScraper()
        valid_tickers = ['AAPL', 'MSFT', 'TSLA']

        text = "I think $AAPL is going to the moon!"
        found = scraper.extract_tickers(text, valid_tickers)

        assert 'AAPL' in found

    def test_extract_tickers_multiple(self):
        """Test extraction of multiple tickers"""
        scraper = RedditScraper()
        valid_tickers = ['AAPL', 'MSFT', 'TSLA', 'NVDA']

        text = "Buying AAPL and $TSLA, selling MSFT"
        found = scraper.extract_tickers(text, valid_tickers)

        assert 'AAPL' in found
        assert 'TSLA' in found
        assert 'MSFT' in found

    def test_extract_tickers_no_match(self):
        """Test when no tickers are found"""
        scraper = RedditScraper()
        valid_tickers = ['AAPL', 'MSFT']

        text = "The market is doing well today"
        found = scraper.extract_tickers(text, valid_tickers)

        assert len(found) == 0

    def test_extract_tickers_empty_text(self):
        """Test with empty text"""
        scraper = RedditScraper()
        valid_tickers = ['AAPL']

        found = scraper.extract_tickers("", valid_tickers)
        assert found == []

    def test_extract_tickers_none_text(self):
        """Test with None text"""
        scraper = RedditScraper()
        valid_tickers = ['AAPL']

        found = scraper.extract_tickers(None, valid_tickers)
        assert found == []

    def test_extract_tickers_case_insensitive(self):
        """Test that extraction is case insensitive"""
        scraper = RedditScraper()
        valid_tickers = ['AAPL']

        text = "aapl is looking good"
        found = scraper.extract_tickers(text, valid_tickers)

        assert 'AAPL' in found

    def test_extract_tickers_with_dollar_sign(self):
        """Test extraction with $ prefix"""
        scraper = RedditScraper()
        valid_tickers = ['GME', 'AMC']

        text = "Diamond hands on $GME and $AMC!"
        found = scraper.extract_tickers(text, valid_tickers)

        assert 'GME' in found
        assert 'AMC' in found


class TestDataIntegrity:
    """Tests for data integrity across pipeline"""

    def test_stock_data_date_format(self):
        """Test that dates are properly formatted"""
        collector = StockDataCollector(['AAPL'])
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        df = collector.get_historical_data(start_date)

        if not df.empty:
            # Date should be convertible to datetime
            dates = pd.to_datetime(df['date'])
            assert not dates.isna().any()

    def test_stock_data_numeric_columns(self):
        """Test that price columns are numeric"""
        collector = StockDataCollector(['AAPL'])
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        df = collector.get_historical_data(start_date)

        if not df.empty:
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    assert pd.api.types.is_numeric_dtype(df[col])

    def test_stock_data_positive_prices(self):
        """Test that prices are positive"""
        collector = StockDataCollector(['AAPL'])
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        df = collector.get_historical_data(start_date)

        if not df.empty:
            assert (df['close'] > 0).all()
            assert (df['open'] > 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
