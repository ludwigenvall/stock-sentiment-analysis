#!/usr/bin/env python3
"""
Train ML Model for Stock Sentiment Prediction

Usage:
    python train_model.py [--start 2024-01-01] [--end 2025-12-31]
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.feature_engineer import FeatureEngineer
from src.ml.model_trainer import SentimentPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sentiment_data(path: str = "data/processed/all_sentiment.csv") -> pd.DataFrame:
    """Load sentiment data from CSV."""
    if not Path(path).exists():
        logger.error(f"Sentiment data not found at {path}")
        logger.info("Run 'python analyze_with_recommendations.py' first to generate data")
        return pd.DataFrame()

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} sentiment records")
    return df


def load_price_data(path: str = "data/processed/stock_prices.csv") -> pd.DataFrame:
    """Load price data from CSV."""
    if not Path(path).exists():
        logger.warning(f"Price data not found at {path}, will fetch from yfinance")
        return pd.DataFrame()

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} price records")
    return df


def fetch_price_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch price data from yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return pd.DataFrame()

    all_data = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            if not data.empty:
                ticker_df = pd.DataFrame({
                    'ticker': ticker,
                    'date': data.index,
                    'close': data['Close'].values,
                    'volume': data['Volume'].values
                })
                all_data.append(ticker_df)
                logger.info(f"Fetched {len(ticker_df)} days of data for {ticker}")
        except Exception as e:
            logger.warning(f"Failed to fetch {ticker}: {e}")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description='Train sentiment prediction model')
    parser.add_argument('--start', type=str, default='2024-01-01',
                       help='Start date for training data (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                       help='End date for training data (YYYY-MM-DD)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data to use for testing')
    parser.add_argument('--target-horizon', type=int, default=5,
                       help='Days ahead to predict returns')
    args = parser.parse_args()

    if args.end is None:
        args.end = datetime.now().strftime('%Y-%m-%d')

    print("\n" + "=" * 60)
    print("TRAINING SENTIMENT PREDICTION MODEL")
    print("=" * 60)
    print(f"\nDate range: {args.start} to {args.end}")
    print(f"Test size: {args.test_size * 100:.0f}%")
    print(f"Prediction horizon: {args.target_horizon} days")

    # Load sentiment data
    sentiment_df = load_sentiment_data()
    if sentiment_df.empty:
        logger.error("No sentiment data available")
        return

    # Filter date range
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], format='mixed')
    start_dt = pd.to_datetime(args.start)
    end_dt = pd.to_datetime(args.end)
    sentiment_df = sentiment_df[
        (sentiment_df['date'] >= start_dt) &
        (sentiment_df['date'] <= end_dt)
    ]

    if sentiment_df.empty:
        logger.error("No sentiment data in specified date range")
        return

    # Get unique tickers
    tickers = sentiment_df['ticker'].unique().tolist()
    logger.info(f"Found {len(tickers)} tickers in sentiment data")

    # Load or fetch price data
    price_df = load_price_data()

    # Check if we have enough price data (need at least 30 days for momentum calculations)
    need_more_data = False
    if price_df.empty:
        need_more_data = True
    else:
        price_df['date'] = pd.to_datetime(price_df['date'])
        date_range = (price_df['date'].max() - price_df['date'].min()).days
        if date_range < 30:
            logger.info(f"Only {date_range} days of price data available, fetching more...")
            need_more_data = True

    if need_more_data or not set(tickers).intersection(set(price_df['ticker'].unique() if not price_df.empty else [])):
        logger.info("Fetching price data from yfinance...")
        price_df = fetch_price_data(tickers, args.start, args.end)

    if price_df.empty:
        logger.error("No price data available")
        return

    # Create features
    print("\n--- Feature Engineering ---")
    feature_eng = FeatureEngineer(
        momentum_windows=[5, 10, 20],
        target_horizon=args.target_horizon,
        min_articles=1  # Lower threshold for limited data
    )

    features_df = feature_eng.create_features(
        sentiment_df,
        price_df,
        include_target=True
    )

    if features_df.empty:
        logger.error("Failed to create features")
        return

    print(f"Created {len(features_df)} samples with {len(feature_eng.get_feature_columns())} features")

    # Save features
    feature_eng.save_features(features_df, "data/ml/features.csv")

    # Split data
    print("\n--- Train/Test Split ---")
    X_train, X_test, y_train, y_test = feature_eng.prepare_train_test_split(
        features_df,
        test_size=args.test_size,
        time_based=True
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Train model
    print("\n--- Training Model ---")
    predictor = SentimentPredictor(
        task='regression',
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6
    )

    metrics = predictor.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        early_stopping_rounds=20
    )

    # Evaluate
    print("\n--- Evaluation ---")
    test_metrics = predictor.evaluate(X_test, y_test)
    print("\nTest Set Metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Cross-validation
    print("\n--- Cross-Validation ---")
    cv_results = predictor.cross_validate(
        pd.concat([X_train, X_test]),
        pd.concat([y_train, y_test]),
        cv=5
    )
    print(f"CV MSE: {cv_results['cv_mean']:.6f} (+/- {cv_results['cv_std']:.6f})")

    # Save model
    model_path = "models/sentiment_predictor.pkl"
    predictor.save(model_path)

    # Print summary
    predictor.print_summary()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nModel saved to: {model_path}")
    print(f"Features saved to: data/ml/features.csv")
    print("\nTo use the model:")
    print("  from src.ml.model_trainer import SentimentPredictor")
    print("  predictor = SentimentPredictor.load('models/sentiment_predictor.pkl')")
    print("  predictions = predictor.predict(features)")


if __name__ == "__main__":
    main()
