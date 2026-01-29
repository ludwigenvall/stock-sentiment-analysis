"""
SQLAlchemy Models for Stock Sentiment Analysis

Database schema for storing:
- Stock prices
- Sentiment data
- Recommendations
- Backtest results
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Date, Text, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class StockPrice(Base):
    """Stock price data."""
    __tablename__ = 'stock_prices'

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float, nullable=False)
    volume = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('ix_stock_prices_ticker_date', 'ticker', 'date', unique=True),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'ticker': self.ticker,
            'date': self.date.isoformat() if self.date else None,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }


class SentimentData(Base):
    """Sentiment analysis data."""
    __tablename__ = 'sentiment_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    content_type = Column(String(50), index=True)  # news, reddit, sec_filing, earnings
    title = Column(Text)
    text = Column(Text)
    sentiment_score = Column(Float, nullable=False)
    sentiment_label = Column(String(20))  # positive, negative, neutral
    source = Column(String(100))
    url = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('ix_sentiment_ticker_date', 'ticker', 'date'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'ticker': self.ticker,
            'date': self.date.isoformat() if self.date else None,
            'content_type': self.content_type,
            'title': self.title,
            'sentiment_score': self.sentiment_score,
            'sentiment_label': self.sentiment_label,
            'source': self.source
        }


class Recommendation(Base):
    """Stock recommendations."""
    __tablename__ = 'recommendations'

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    recommendation = Column(String(10), nullable=False)  # BUY, SELL, HOLD
    confidence = Column(Float)
    sentiment_score = Column(Float)
    num_articles = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('ix_recommendations_ticker_date', 'ticker', 'date'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'ticker': self.ticker,
            'date': self.date.isoformat() if self.date else None,
            'recommendation': self.recommendation,
            'confidence': self.confidence,
            'sentiment_score': self.sentiment_score,
            'num_articles': self.num_articles
        }


class BacktestResult(Base):
    """Backtest results."""
    __tablename__ = 'backtest_results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_date = Column(DateTime, default=datetime.utcnow, index=True)
    initial_capital = Column(Float)
    final_value = Column(Float)
    total_return_pct = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown_pct = Column(Float)
    num_trades = Column(Integer)
    win_rate_pct = Column(Float)
    parameters = Column(Text)  # JSON string of backtest parameters

    def to_dict(self):
        return {
            'id': self.id,
            'run_date': self.run_date.isoformat() if self.run_date else None,
            'initial_capital': self.initial_capital,
            'final_value': self.final_value,
            'total_return_pct': self.total_return_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown_pct': self.max_drawdown_pct,
            'num_trades': self.num_trades,
            'win_rate_pct': self.win_rate_pct
        }


class Trade(Base):
    """Individual trades from backtesting."""
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True, autoincrement=True)
    backtest_id = Column(Integer, ForeignKey('backtest_results.id'), index=True)
    ticker = Column(String(10), nullable=False)
    entry_date = Column(Date)
    exit_date = Column(Date)
    entry_price = Column(Float)
    exit_price = Column(Float)
    shares = Column(Integer)
    pnl = Column(Float)
    pnl_pct = Column(Float)
    direction = Column(String(10))  # long, short

    def to_dict(self):
        return {
            'id': self.id,
            'ticker': self.ticker,
            'entry_date': self.entry_date.isoformat() if self.entry_date else None,
            'exit_date': self.exit_date.isoformat() if self.exit_date else None,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'shares': self.shares,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'direction': self.direction
        }
