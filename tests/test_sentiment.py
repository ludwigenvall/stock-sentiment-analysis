"""
Unit tests for sentiment analysis module
"""
import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sentiment.finbert_analyzer import FinBERTAnalyzer


class TestFinBERTAnalyzer:
    """Tests for FinBERT sentiment analyzer"""

    @pytest.fixture(scope="class")
    def analyzer(self):
        """Create analyzer instance (reused across tests)"""
        return FinBERTAnalyzer()

    def test_analyzer_initialization(self, analyzer):
        """Test that analyzer initializes correctly"""
        assert analyzer is not None
        assert analyzer.model is not None
        assert analyzer.tokenizer is not None

    def test_analyze_positive_sentiment(self, analyzer):
        """Test analysis of positive text"""
        text = "Apple stock surges to all-time high after strong earnings report"
        result = analyzer.analyze_sentiment(text)

        assert 'sentiment_score' in result
        assert 'sentiment_label' in result
        assert 'sentiment_positive' in result
        assert 'sentiment_negative' in result
        assert 'sentiment_neutral' in result

        # Should be positive
        assert result['sentiment_score'] > 0
        assert result['sentiment_label'] == 'positive'

    def test_analyze_negative_sentiment(self, analyzer):
        """Test analysis of negative text"""
        text = "Company announces massive layoffs amid declining sales and market crash"
        result = analyzer.analyze_sentiment(text)

        # Should be negative
        assert result['sentiment_score'] < 0
        assert result['sentiment_label'] == 'negative'

    def test_analyze_neutral_sentiment(self, analyzer):
        """Test analysis of neutral text"""
        text = "The company released its quarterly report today"
        result = analyzer.analyze_sentiment(text)

        # Should be close to neutral (within reasonable range)
        assert -0.5 < result['sentiment_score'] < 0.5

    def test_analyze_batch(self, analyzer):
        """Test batch analysis"""
        texts = [
            "Great earnings beat expectations",
            "Stock price crashed after scandal",
            "Company announces new product"
        ]
        results = analyzer.analyze_batch(texts)

        assert len(results) == 3
        assert all('sentiment_score' in r for r in results)

    def test_analyze_dataframe(self, analyzer):
        """Test DataFrame analysis"""
        df = pd.DataFrame({
            'title': [
                "Stock rises on positive outlook",
                "Market falls amid uncertainty",
                "Company reports quarterly results"
            ],
            'ticker': ['AAPL', 'MSFT', 'GOOGL']
        })

        result_df = analyzer.analyze_dataframe(df, text_column='title')

        assert 'sentiment_score' in result_df.columns
        assert 'sentiment_label' in result_df.columns
        assert len(result_df) == 3

    def test_sentiment_score_range(self, analyzer):
        """Test that sentiment scores are within valid range"""
        texts = [
            "Excellent performance, stock soars!",
            "Terrible losses, investors flee",
            "Normal trading day"
        ]

        for text in texts:
            result = analyzer.analyze_sentiment(text)
            assert -1 <= result['sentiment_score'] <= 1
            assert 0 <= result['sentiment_positive'] <= 1
            assert 0 <= result['sentiment_negative'] <= 1
            assert 0 <= result['sentiment_neutral'] <= 1

    def test_empty_text(self, analyzer):
        """Test handling of empty text"""
        result = analyzer.analyze_sentiment("")

        # Should return valid result structure
        assert 'sentiment_score' in result

    def test_long_text_truncation(self, analyzer):
        """Test that long texts are handled (truncated)"""
        # Create very long text
        long_text = "Stock price increases. " * 1000
        result = analyzer.analyze_sentiment(long_text)

        # Should still return valid result
        assert 'sentiment_score' in result
        assert -1 <= result['sentiment_score'] <= 1


class TestSentimentLabels:
    """Test sentiment label classification"""

    @pytest.fixture(scope="class")
    def analyzer(self):
        return FinBERTAnalyzer()

    def test_positive_label_threshold(self, analyzer):
        """Test positive label assignment"""
        # Strongly positive text
        result = analyzer.analyze_sentiment(
            "Incredible growth! Record profits! Best quarter ever!"
        )
        assert result['sentiment_label'] == 'positive'

    def test_negative_label_threshold(self, analyzer):
        """Test negative label assignment"""
        # Strongly negative text
        result = analyzer.analyze_sentiment(
            "Catastrophic losses. Bankruptcy imminent. Investors devastated."
        )
        assert result['sentiment_label'] == 'negative'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
