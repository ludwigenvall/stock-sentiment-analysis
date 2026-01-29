"""
Feature Engineering for Stock Sentiment ML Model

Creates features from sentiment data and price data for training ML models.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates ML features from sentiment and price data.

    Features include:
    - Sentiment statistics (mean, std, count)
    - Price momentum indicators
    - Volume changes
    - Day of week/month effects
    """

    def __init__(
        self,
        momentum_windows: List[int] = [5, 10, 20],
        target_horizon: int = 5,
        min_articles: int = 3
    ):
        """
        Initialize feature engineer.

        Args:
            momentum_windows: List of lookback windows for momentum features
            target_horizon: Days ahead to predict returns
            min_articles: Minimum articles required for a valid sample
        """
        self.momentum_windows = momentum_windows
        self.target_horizon = target_horizon
        self.min_articles = min_articles

        self.feature_columns = []

    def create_features(
        self,
        sentiment_df: pd.DataFrame,
        price_df: pd.DataFrame,
        include_target: bool = True
    ) -> pd.DataFrame:
        """
        Create feature matrix from sentiment and price data.

        Args:
            sentiment_df: DataFrame with [ticker, date, sentiment_score]
            price_df: DataFrame with [ticker, date, close, volume (optional)]
            include_target: Whether to calculate forward returns as target

        Returns:
            DataFrame with features and optionally target variable
        """
        # Ensure date columns are datetime (timezone-naive)
        sentiment_df = sentiment_df.copy()
        price_df = price_df.copy()
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.tz_localize(None)
        price_df['date'] = pd.to_datetime(price_df['date']).dt.tz_localize(None)

        # Aggregate sentiment by ticker and date
        sentiment_agg = self._aggregate_sentiment(sentiment_df)

        # Create price features
        price_features = self._create_price_features(price_df)

        # Merge sentiment and price features
        features = pd.merge(
            sentiment_agg,
            price_features,
            on=['ticker', 'date'],
            how='inner'
        )

        # Add calendar features
        features = self._add_calendar_features(features)

        # Create target variable
        if include_target:
            features = self._create_target(features)

        # Filter samples with minimum articles
        features = features[features['num_articles'] >= self.min_articles]

        # Drop rows with NaN in features
        feature_cols = [c for c in features.columns
                       if c not in ['ticker', 'date', 'target', 'target_direction']]
        features = features.dropna(subset=feature_cols)

        # Store feature columns
        self.feature_columns = feature_cols

        logger.info(f"Created {len(features)} samples with {len(feature_cols)} features")

        return features

    def _aggregate_sentiment(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sentiment scores by ticker and date."""
        agg = sentiment_df.groupby(['ticker', 'date']).agg({
            'sentiment_score': ['mean', 'std', 'min', 'max', 'count']
        }).reset_index()

        # Flatten column names
        agg.columns = ['ticker', 'date', 'sentiment_mean', 'sentiment_std',
                       'sentiment_min', 'sentiment_max', 'num_articles']

        # Fill NaN std with 0 (single article)
        agg['sentiment_std'] = agg['sentiment_std'].fillna(0)

        # Sentiment range
        agg['sentiment_range'] = agg['sentiment_max'] - agg['sentiment_min']

        return agg

    def _create_price_features(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features."""
        features_list = []

        for ticker in price_df['ticker'].unique():
            ticker_data = price_df[price_df['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date')

            # Price momentum
            for window in self.momentum_windows:
                ticker_data[f'momentum_{window}d'] = ticker_data['close'].pct_change(window)

            # Volatility
            ticker_data['volatility_10d'] = ticker_data['close'].pct_change().rolling(10).std()
            ticker_data['volatility_20d'] = ticker_data['close'].pct_change().rolling(20).std()

            # Price relative to moving averages
            ticker_data['price_vs_ma10'] = ticker_data['close'] / ticker_data['close'].rolling(10).mean() - 1
            ticker_data['price_vs_ma20'] = ticker_data['close'] / ticker_data['close'].rolling(20).mean() - 1

            # Volume features (if available)
            if 'volume' in ticker_data.columns:
                ticker_data['volume_change'] = ticker_data['volume'].pct_change()
                ticker_data['volume_ma_ratio'] = ticker_data['volume'] / ticker_data['volume'].rolling(10).mean()

            features_list.append(ticker_data)

        return pd.concat(features_list, ignore_index=True)

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar-based features."""
        df = df.copy()
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['is_month_start'] = (df['date'].dt.day <= 5).astype(int)
        df['is_month_end'] = (df['date'].dt.day >= 25).astype(int)

        return df

    def _create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create forward return target variable."""
        df = df.copy()

        # Calculate forward returns for each ticker
        target_list = []
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date')

            # Forward return
            ticker_data['target'] = (
                ticker_data['close'].shift(-self.target_horizon) / ticker_data['close'] - 1
            )

            # Direction classification
            ticker_data['target_direction'] = np.where(
                ticker_data['target'] > 0.01, 1,  # Up > 1%
                np.where(ticker_data['target'] < -0.01, -1, 0)  # Down < -1%, else flat
            )

            target_list.append(ticker_data)

        return pd.concat(target_list, ignore_index=True)

    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names."""
        return self.feature_columns

    def prepare_train_test_split(
        self,
        features_df: pd.DataFrame,
        test_size: float = 0.2,
        time_based: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split features into train and test sets.

        Args:
            features_df: Feature DataFrame with target
            test_size: Fraction of data to use for testing
            time_based: If True, use time-based split to avoid look-ahead bias

        Returns:
            X_train, X_test, y_train, y_test
        """
        df = features_df.dropna(subset=['target'])

        if time_based:
            # Sort by date and split
            df = df.sort_values('date')
            split_idx = int(len(df) * (1 - test_size))
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
        else:
            # Random split
            train_df = df.sample(frac=1-test_size, random_state=42)
            test_df = df.drop(train_df.index)

        feature_cols = self.feature_columns

        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]
        y_train = train_df['target']
        y_test = test_df['target']

        logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

        return X_train, X_test, y_train, y_test

    def save_features(self, features_df: pd.DataFrame, output_path: str = "data/ml/features.csv"):
        """Save feature DataFrame to CSV."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(output_path, index=False)
        logger.info(f"Features saved to {output_path}")

    @staticmethod
    def load_features(input_path: str = "data/ml/features.csv") -> pd.DataFrame:
        """Load feature DataFrame from CSV."""
        return pd.read_csv(input_path)
