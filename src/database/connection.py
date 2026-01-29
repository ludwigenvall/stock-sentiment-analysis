"""
Database Connection Module

Handles PostgreSQL database connections and session management.
Falls back to SQLite for local development.
"""

import os
import logging
from typing import Generator
from contextlib import contextmanager
from dotenv import load_dotenv

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .models import Base

load_dotenv()
logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Database connection manager.

    Supports PostgreSQL for production and SQLite for development.
    """

    def __init__(self, database_url: str = None):
        """
        Initialize database connection.

        Args:
            database_url: Database URL (postgresql://... or sqlite:///...)
                          If not provided, uses DATABASE_URL env var or SQLite fallback
        """
        self.database_url = database_url or os.getenv('DATABASE_URL')

        if not self.database_url:
            # Fallback to SQLite for local development
            self.database_url = "sqlite:///data/sentiment_analysis.db"
            logger.info("Using SQLite database (local development)")
        else:
            # Handle Heroku-style postgres:// URLs
            if self.database_url.startswith('postgres://'):
                self.database_url = self.database_url.replace('postgres://', 'postgresql://', 1)
            logger.info("Using PostgreSQL database")

        # Create engine
        if self.database_url.startswith('sqlite'):
            # SQLite specific settings
            self.engine = create_engine(
                self.database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool
            )
        else:
            # PostgreSQL settings
            self.engine = create_engine(
                self.database_url,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True
            )

        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

    def create_tables(self):
        """Create all tables in database."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created")

    def drop_tables(self):
        """Drop all tables (use with caution!)."""
        Base.metadata.drop_all(bind=self.engine)
        logger.info("Database tables dropped")

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions.

        Usage:
            with db.session_scope() as session:
                session.query(...)
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


# Global database instance
_db: DatabaseConnection = None


def get_db() -> DatabaseConnection:
    """Get global database connection instance."""
    global _db
    if _db is None:
        _db = DatabaseConnection()
        _db.create_tables()
    return _db


def get_session() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI routes.

    Usage:
        @app.get("/items")
        def get_items(db: Session = Depends(get_session)):
            ...
    """
    db = get_db()
    session = db.get_session()
    try:
        yield session
    finally:
        session.close()


# Utility functions for storing data
def store_stock_prices(df, session: Session = None):
    """Store stock prices from DataFrame to database."""
    from .models import StockPrice
    import pandas as pd

    db = get_db()
    close_session = session is None
    if session is None:
        session = db.get_session()

    try:
        for _, row in df.iterrows():
            # Check if exists
            existing = session.query(StockPrice).filter(
                StockPrice.ticker == row['ticker'],
                StockPrice.date == pd.to_datetime(row['date']).date()
            ).first()

            if existing:
                # Update
                existing.open = row.get('open')
                existing.high = row.get('high')
                existing.low = row.get('low')
                existing.close = row['close']
                existing.volume = row.get('volume')
            else:
                # Insert
                price = StockPrice(
                    ticker=row['ticker'],
                    date=pd.to_datetime(row['date']).date(),
                    open=row.get('open'),
                    high=row.get('high'),
                    low=row.get('low'),
                    close=row['close'],
                    volume=row.get('volume')
                )
                session.add(price)

        session.commit()
        logger.info(f"Stored {len(df)} stock prices")

    except Exception as e:
        session.rollback()
        logger.error(f"Error storing stock prices: {e}")
        raise
    finally:
        if close_session:
            session.close()


def store_sentiment_data(df, session: Session = None):
    """Store sentiment data from DataFrame to database."""
    from .models import SentimentData
    import pandas as pd

    db = get_db()
    close_session = session is None
    if session is None:
        session = db.get_session()

    try:
        for _, row in df.iterrows():
            sentiment = SentimentData(
                ticker=row['ticker'],
                date=pd.to_datetime(row['date']).date(),
                content_type=row.get('content_type'),
                title=row.get('title'),
                sentiment_score=row['sentiment_score'],
                sentiment_label=row.get('sentiment_label'),
                source=row.get('source')
            )
            session.add(sentiment)

        session.commit()
        logger.info(f"Stored {len(df)} sentiment records")

    except Exception as e:
        session.rollback()
        logger.error(f"Error storing sentiment data: {e}")
        raise
    finally:
        if close_session:
            session.close()


def store_recommendations(df, session: Session = None):
    """Store recommendations from DataFrame to database."""
    from .models import Recommendation
    import pandas as pd
    from datetime import date

    db = get_db()
    close_session = session is None
    if session is None:
        session = db.get_session()

    try:
        for _, row in df.iterrows():
            rec = Recommendation(
                ticker=row['ticker'],
                date=date.today(),
                recommendation=row['recommendation'],
                confidence=row.get('confidence'),
                sentiment_score=row.get('sentiment_score'),
                num_articles=row.get('num_articles')
            )
            session.add(rec)

        session.commit()
        logger.info(f"Stored {len(df)} recommendations")

    except Exception as e:
        session.rollback()
        logger.error(f"Error storing recommendations: {e}")
        raise
    finally:
        if close_session:
            session.close()
