"""
ML Model Training for Stock Sentiment Prediction

Trains LightGBM models to predict stock returns based on sentiment and price features.
"""

import pandas as pd
import numpy as np
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import cross_val_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


class SentimentPredictor:
    """
    LightGBM-based predictor for stock returns using sentiment features.

    Can be used for both regression (predicting returns) and classification
    (predicting direction).
    """

    def __init__(
        self,
        task: str = 'regression',
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize predictor.

        Args:
            task: 'regression' for returns, 'classification' for direction
            n_estimators: Number of boosting rounds
            learning_rate: Step size for updates
            max_depth: Maximum tree depth
            num_leaves: Maximum number of leaves
            min_child_samples: Minimum samples per leaf
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            random_state: Random seed for reproducibility
        """
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM is required. Install with: pip install lightgbm")

        self.task = task
        self.params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'min_child_samples': min_child_samples,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'verbose': -1,
            'n_jobs': -1
        }

        if task == 'regression':
            self.model = lgb.LGBMRegressor(**self.params)
        else:
            self.model = lgb.LGBMClassifier(**self.params)

        self.feature_names = None
        self.training_metrics = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 20
    ) -> Dict:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            early_stopping_rounds: Rounds without improvement before stopping

        Returns:
            Dict with training metrics
        """
        self.feature_names = list(X_train.columns)

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks
        )

        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        metrics = self._calculate_metrics(y_train, train_pred, prefix='train')

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, val_pred, prefix='val')
            metrics.update(val_metrics)

        self.training_metrics = metrics
        logger.info(f"Training complete. Metrics: {metrics}")

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence scores.

        For regression, confidence is based on prediction magnitude.
        For classification, returns class probabilities.
        """
        predictions = self.predict(X)

        if self.task == 'classification':
            confidence = self.model.predict_proba(X).max(axis=1)
        else:
            # For regression, higher absolute predictions = higher confidence
            confidence = np.abs(predictions) / (np.abs(predictions).max() + 1e-8)

        return predictions, confidence

    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        prefix: str = ''
    ) -> Dict:
        """Calculate evaluation metrics."""
        if not HAS_SKLEARN:
            return {f'{prefix}_mse': 0, f'{prefix}_mae': 0, f'{prefix}_r2': 0}

        if self.task == 'regression':
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            # Direction accuracy
            direction_correct = ((y_pred > 0) == (y_true > 0)).mean()

            return {
                f'{prefix}_mse': mse,
                f'{prefix}_rmse': np.sqrt(mse),
                f'{prefix}_mae': mae,
                f'{prefix}_r2': r2,
                f'{prefix}_direction_accuracy': direction_correct
            }
        else:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            return {
                f'{prefix}_accuracy': accuracy_score(y_true, y_pred),
                f'{prefix}_precision': precision_score(y_true, y_pred, average='weighted'),
                f'{prefix}_recall': recall_score(y_true, y_pred, average='weighted'),
                f'{prefix}_f1': f1_score(y_true, y_pred, average='weighted')
            }

    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        })
        return importance.sort_values('importance', ascending=False)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate model on test data."""
        predictions = self.predict(X_test)
        return self._calculate_metrics(y_test, predictions, prefix='test')

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5
    ) -> Dict:
        """Perform cross-validation."""
        if not HAS_SKLEARN:
            return {'cv_mean': 0, 'cv_std': 0}

        scoring = 'neg_mean_squared_error' if self.task == 'regression' else 'accuracy'
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)

        if self.task == 'regression':
            scores = -scores  # Convert back to positive MSE

        return {
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'cv_scores': scores.tolist()
        }

    def save(self, path: str = "models/sentiment_predictor.pkl"):
        """Save model to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'model': self.model,
            'task': self.task,
            'params': self.params,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics
        }

        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str = "models/sentiment_predictor.pkl") -> 'SentimentPredictor':
        """Load model from file."""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)

        predictor = cls.__new__(cls)
        predictor.model = save_data['model']
        predictor.task = save_data['task']
        predictor.params = save_data['params']
        predictor.feature_names = save_data['feature_names']
        predictor.training_metrics = save_data['training_metrics']

        logger.info(f"Model loaded from {path}")
        return predictor

    def print_summary(self):
        """Print model summary."""
        print("\n" + "=" * 50)
        print("SENTIMENT PREDICTOR SUMMARY")
        print("=" * 50)
        print(f"\nTask: {self.task}")
        print(f"Features: {len(self.feature_names) if self.feature_names else 'Not trained'}")

        if self.training_metrics:
            print("\nTraining Metrics:")
            for key, value in self.training_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

        if self.feature_names:
            print("\nTop 10 Features:")
            importance = self.feature_importance()
            for _, row in importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.0f}")

        print("=" * 50)
