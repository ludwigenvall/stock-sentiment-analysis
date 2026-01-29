"""
REST API for Stock Sentiment Analysis

FastAPI endpoints for:
- Getting stock recommendations
- Fetching sentiment data
- Running analysis
- Backtest results
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Stock Sentiment Analysis API",
    description="API for sentiment-based stock recommendations",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RECOMMENDATIONS_FILE = DATA_DIR / "recommendations" / "latest_recommendations.csv"
SENTIMENT_FILE = DATA_DIR / "processed" / "all_sentiment.csv"
STOCK_FILE = DATA_DIR / "processed" / "stock_prices.csv"


# Pydantic models
class Recommendation(BaseModel):
    ticker: str
    recommendation: str
    confidence: float
    sentiment_score: float
    num_articles: int


class SentimentData(BaseModel):
    ticker: str
    date: str
    sentiment_score: float
    sentiment_label: str
    content_type: str
    title: Optional[str] = None


class StockPrice(BaseModel):
    ticker: str
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class BacktestResult(BaseModel):
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    num_trades: int


class AnalysisRequest(BaseModel):
    tickers: List[str]
    days_back: int = 7


# Helper functions
def load_recommendations() -> pd.DataFrame:
    """Load latest recommendations."""
    if RECOMMENDATIONS_FILE.exists():
        return pd.read_csv(RECOMMENDATIONS_FILE)
    return pd.DataFrame()


def load_sentiment() -> pd.DataFrame:
    """Load sentiment data."""
    if SENTIMENT_FILE.exists():
        return pd.read_csv(SENTIMENT_FILE)
    return pd.DataFrame()


def load_prices() -> pd.DataFrame:
    """Load stock prices."""
    if STOCK_FILE.exists():
        return pd.read_csv(STOCK_FILE)
    return pd.DataFrame()


# API Endpoints
@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "Stock Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": [
            "/recommendations",
            "/sentiment/{ticker}",
            "/prices/{ticker}",
            "/tickers",
            "/health"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_available": {
            "recommendations": RECOMMENDATIONS_FILE.exists(),
            "sentiment": SENTIMENT_FILE.exists(),
            "prices": STOCK_FILE.exists()
        }
    }


@app.get("/tickers", response_model=List[str])
async def get_tickers():
    """Get list of available tickers."""
    df = load_recommendations()
    if df.empty:
        df = load_sentiment()

    if df.empty:
        raise HTTPException(status_code=404, detail="No data available")

    return sorted(df['ticker'].unique().tolist())


@app.get("/recommendations", response_model=List[Recommendation])
async def get_recommendations(
    recommendation: Optional[str] = Query(None, description="Filter by recommendation type (BUY/SELL/HOLD)"),
    min_confidence: float = Query(0, description="Minimum confidence threshold"),
    limit: int = Query(20, description="Maximum results to return")
):
    """Get stock recommendations."""
    df = load_recommendations()

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="No recommendations available. Run analysis first."
        )

    # Filter by recommendation type
    if recommendation:
        df = df[df['recommendation'].str.upper() == recommendation.upper()]

    # Filter by confidence
    df = df[df['confidence'] >= min_confidence]

    # Sort and limit
    df = df.sort_values('confidence', ascending=False).head(limit)

    return df.to_dict('records')


@app.get("/recommendations/{ticker}", response_model=Recommendation)
async def get_recommendation_for_ticker(ticker: str):
    """Get recommendation for specific ticker."""
    df = load_recommendations()

    if df.empty:
        raise HTTPException(status_code=404, detail="No recommendations available")

    ticker_data = df[df['ticker'].str.upper() == ticker.upper()]

    if ticker_data.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No recommendation found for {ticker}"
        )

    return ticker_data.iloc[0].to_dict()


@app.get("/sentiment/{ticker}", response_model=List[SentimentData])
async def get_sentiment_for_ticker(
    ticker: str,
    days: int = Query(7, description="Number of days of data"),
    content_type: Optional[str] = Query(None, description="Filter by content type")
):
    """Get sentiment data for a ticker."""
    df = load_sentiment()

    if df.empty:
        raise HTTPException(status_code=404, detail="No sentiment data available")

    # Filter by ticker
    df = df[df['ticker'].str.upper() == ticker.upper()]

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No sentiment data found for {ticker}"
        )

    # Filter by content type
    if content_type and 'content_type' in df.columns:
        df = df[df['content_type'] == content_type]

    # Filter by date
    df['date'] = pd.to_datetime(df['date'])
    cutoff = datetime.now() - pd.Timedelta(days=days)
    df = df[df['date'] >= cutoff]

    # Sort by date
    df = df.sort_values('date', ascending=False)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    return df.to_dict('records')


@app.get("/sentiment/summary/{ticker}")
async def get_sentiment_summary(ticker: str):
    """Get sentiment summary for a ticker."""
    df = load_sentiment()

    if df.empty:
        raise HTTPException(status_code=404, detail="No sentiment data available")

    ticker_data = df[df['ticker'].str.upper() == ticker.upper()]

    if ticker_data.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No sentiment data found for {ticker}"
        )

    # Calculate summary
    summary = {
        "ticker": ticker.upper(),
        "total_articles": len(ticker_data),
        "avg_sentiment": ticker_data['sentiment_score'].mean(),
        "sentiment_std": ticker_data['sentiment_score'].std(),
        "positive_pct": (ticker_data['sentiment_label'] == 'positive').mean() * 100,
        "negative_pct": (ticker_data['sentiment_label'] == 'negative').mean() * 100,
        "neutral_pct": (ticker_data['sentiment_label'] == 'neutral').mean() * 100,
    }

    # Content type breakdown
    if 'content_type' in ticker_data.columns:
        summary['by_content_type'] = ticker_data.groupby('content_type').agg({
            'sentiment_score': ['mean', 'count']
        }).to_dict()

    return summary


@app.get("/prices/{ticker}", response_model=List[StockPrice])
async def get_prices_for_ticker(
    ticker: str,
    days: int = Query(30, description="Number of days of data")
):
    """Get stock prices for a ticker."""
    df = load_prices()

    if df.empty:
        raise HTTPException(status_code=404, detail="No price data available")

    # Filter by ticker
    df = df[df['ticker'].str.upper() == ticker.upper()]

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No price data found for {ticker}"
        )

    # Filter by date
    df['date'] = pd.to_datetime(df['date'])
    cutoff = datetime.now() - pd.Timedelta(days=days)
    df = df[df['date'] >= cutoff]

    # Sort and format
    df = df.sort_values('date', ascending=False)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    return df.to_dict('records')


@app.get("/top-picks")
async def get_top_picks(n: int = Query(10, description="Number of picks")):
    """Get top N buy recommendations."""
    df = load_recommendations()

    if df.empty:
        raise HTTPException(status_code=404, detail="No recommendations available")

    # Filter to BUY recommendations
    buy_recs = df[df['recommendation'].str.upper() == 'BUY']
    buy_recs = buy_recs.sort_values('confidence', ascending=False).head(n)

    return {
        "top_picks": buy_recs.to_dict('records'),
        "generated_at": datetime.now().isoformat()
    }


@app.get("/stocks-to-avoid")
async def get_stocks_to_avoid(n: int = Query(5, description="Number of stocks")):
    """Get stocks with SELL recommendations."""
    df = load_recommendations()

    if df.empty:
        raise HTTPException(status_code=404, detail="No recommendations available")

    # Filter to SELL recommendations
    sell_recs = df[df['recommendation'].str.upper() == 'SELL']
    sell_recs = sell_recs.sort_values('sentiment_score', ascending=True).head(n)

    return {
        "stocks_to_avoid": sell_recs.to_dict('records'),
        "generated_at": datetime.now().isoformat()
    }


# Run with: uvicorn src.api.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
