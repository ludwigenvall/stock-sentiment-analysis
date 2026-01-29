"""
Recommendation History Tracker

Tracks recommendations over time and calculates hypothetical portfolio performance.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Default paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RECOMMENDATIONS_DIR = DATA_DIR / "recommendations"
HISTORY_FILE = RECOMMENDATIONS_DIR / "history.json"
PERFORMANCE_FILE = RECOMMENDATIONS_DIR / "performance_history.csv"


class RecommendationTracker:
    """
    Tracks recommendation history and calculates portfolio performance.
    """

    def __init__(self, history_file: Path = None, initial_capital: float = 100000):
        """
        Initialize the tracker.

        Args:
            history_file: Path to the history JSON file
            initial_capital: Starting capital for portfolio simulation
        """
        self.history_file = history_file or HISTORY_FILE
        self.initial_capital = initial_capital
        self.history = self._load_history()

    def _load_history(self) -> Dict:
        """Load history from file."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                data = json.load(f)
            # Handle both list format and dict format
            if isinstance(data, list):
                return {"recommendations": data, "portfolios": {}}
            return data
        return {"recommendations": [], "portfolios": {}}

    def _save_history(self):
        """Save history to file."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

    def add_recommendations(self, recommendations_df: pd.DataFrame, timestamp: str = None):
        """
        Add a new set of recommendations to history.

        Args:
            recommendations_df: DataFrame with columns [ticker, recommendation, confidence, sentiment_score]
            timestamp: ISO format timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        # Convert DataFrame to records
        records = recommendations_df.to_dict('records')

        # Create entry
        entry = {
            "timestamp": timestamp,
            "date": timestamp[:10],
            "recommendations": records,
            "summary": {
                "total": len(records),
                "buy": len([r for r in records if r.get('recommendation') == 'BUY']),
                "hold": len([r for r in records if r.get('recommendation') == 'HOLD']),
                "sell": len([r for r in records if r.get('recommendation') == 'SELL']),
                "avg_confidence": np.mean([r.get('confidence', 0) for r in records])
            }
        }

        self.history["recommendations"].append(entry)
        self._save_history()

        logger.info(f"Added {len(records)} recommendations to history")
        return entry

    def get_history(self, days: int = None) -> List[Dict]:
        """
        Get recommendation history.

        Args:
            days: Limit to last N days (None for all)

        Returns:
            List of recommendation entries
        """
        history = self.history.get("recommendations", [])

        if days:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            history = [h for h in history if h["timestamp"] >= cutoff]

        return history

    def get_ticker_history(self, ticker: str) -> pd.DataFrame:
        """
        Get recommendation history for a specific ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with recommendation history for the ticker
        """
        records = []

        for entry in self.history.get("recommendations", []):
            for rec in entry.get("recommendations", []):
                if rec.get("ticker") == ticker:
                    records.append({
                        "date": entry["date"],
                        "timestamp": entry["timestamp"],
                        "recommendation": rec.get("recommendation"),
                        "confidence": rec.get("confidence"),
                        "sentiment_score": rec.get("sentiment_score")
                    })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')

    def calculate_portfolio_performance(
        self,
        price_df: pd.DataFrame,
        strategy: str = "buy_signals",
        holding_days: int = 5,
        position_size: float = 0.1
    ) -> Dict:
        """
        Calculate hypothetical portfolio performance following recommendations.

        Args:
            price_df: DataFrame with columns [ticker, date, close]
            strategy: "buy_signals", "sell_signals", "all_signals"
            holding_days: Days to hold each position
            position_size: Fraction of capital per position

        Returns:
            Dict with performance metrics
        """
        history = self.history.get("recommendations", [])

        if not history:
            return {"error": "No recommendation history available"}

        # Initialize portfolio
        capital = self.initial_capital
        positions = {}  # {ticker: {shares, entry_price, entry_date}}
        trades = []
        portfolio_values = []

        # Sort price data
        price_df = price_df.copy()
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df = price_df.sort_values('date')

        # Get unique dates from price data
        all_dates = sorted(price_df['date'].unique())

        for current_date in all_dates:
            # Check for recommendations on this date
            date_str = str(current_date)[:10]

            for entry in history:
                if entry.get("date") == date_str:
                    for rec in entry.get("recommendations", []):
                        ticker = rec.get("ticker")
                        signal = rec.get("recommendation")
                        confidence = rec.get("confidence", 0.5)

                        # Skip if already in position
                        if ticker in positions:
                            continue

                        # Get current price
                        ticker_prices = price_df[
                            (price_df['ticker'] == ticker) &
                            (price_df['date'] == current_date)
                        ]

                        if ticker_prices.empty:
                            continue

                        price = ticker_prices['close'].iloc[0]

                        # Execute based on strategy
                        should_buy = False
                        if strategy == "buy_signals" and signal == "BUY":
                            should_buy = True
                        elif strategy == "sell_signals" and signal == "SELL":
                            # Short selling not implemented, skip
                            continue
                        elif strategy == "all_signals" and signal in ["BUY", "HOLD"]:
                            should_buy = True

                        if should_buy and confidence >= 0.6:
                            # Calculate position size
                            position_value = capital * position_size
                            shares = int(position_value / price)

                            if shares > 0 and position_value <= capital:
                                capital -= shares * price
                                positions[ticker] = {
                                    'shares': shares,
                                    'entry_price': price,
                                    'entry_date': current_date,
                                    'signal': signal,
                                    'confidence': confidence
                                }

            # Check for exits (holding period reached)
            tickers_to_close = []
            for ticker, pos in positions.items():
                days_held = (current_date - pos['entry_date']).days
                if days_held >= holding_days:
                    tickers_to_close.append(ticker)

            # Close positions
            for ticker in tickers_to_close:
                ticker_prices = price_df[
                    (price_df['ticker'] == ticker) &
                    (price_df['date'] == current_date)
                ]

                if not ticker_prices.empty:
                    exit_price = ticker_prices['close'].iloc[0]
                    pos = positions[ticker]

                    pnl = (exit_price - pos['entry_price']) * pos['shares']
                    pnl_pct = (exit_price / pos['entry_price'] - 1) * 100

                    capital += pos['shares'] * exit_price

                    trades.append({
                        'ticker': ticker,
                        'entry_date': pos['entry_date'],
                        'exit_date': current_date,
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'shares': pos['shares'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'signal': pos['signal'],
                        'confidence': pos['confidence']
                    })

                    del positions[ticker]

            # Calculate portfolio value
            portfolio_value = capital
            for ticker, pos in positions.items():
                ticker_prices = price_df[
                    (price_df['ticker'] == ticker) &
                    (price_df['date'] == current_date)
                ]
                if not ticker_prices.empty:
                    portfolio_value += pos['shares'] * ticker_prices['close'].iloc[0]

            portfolio_values.append({
                'date': current_date,
                'value': portfolio_value,
                'capital': capital,
                'num_positions': len(positions)
            })

        # Calculate metrics
        if not trades:
            return {
                "strategy": strategy,
                "total_return_pct": 0,
                "num_trades": 0,
                "message": "No trades executed"
            }

        trades_df = pd.DataFrame(trades)
        portfolio_df = pd.DataFrame(portfolio_values)

        # Calculate daily returns
        portfolio_df['daily_return'] = portfolio_df['value'].pct_change()

        # Metrics
        final_value = portfolio_df['value'].iloc[-1]
        total_return = (final_value / self.initial_capital - 1) * 100

        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = winning_trades / len(trades_df) * 100 if len(trades_df) > 0 else 0

        # Sharpe ratio
        if len(portfolio_df) > 1:
            avg_return = portfolio_df['daily_return'].mean()
            std_return = portfolio_df['daily_return'].std()
            sharpe = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        portfolio_df['cummax'] = portfolio_df['value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['value'] - portfolio_df['cummax']) / portfolio_df['cummax']
        max_drawdown = portfolio_df['drawdown'].min() * 100

        return {
            "strategy": strategy,
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "total_return_pct": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_drawdown,
            "num_trades": len(trades_df),
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate_pct": win_rate,
            "avg_return_per_trade_pct": trades_df['pnl_pct'].mean(),
            "best_trade_pct": trades_df['pnl_pct'].max(),
            "worst_trade_pct": trades_df['pnl_pct'].min(),
            "trades": trades_df.to_dict('records'),
            "portfolio_history": portfolio_df.to_dict('records')
        }

    def get_recommendation_accuracy(self, price_df: pd.DataFrame, forward_days: int = 5) -> pd.DataFrame:
        """
        Calculate accuracy of past recommendations.

        Args:
            price_df: DataFrame with [ticker, date, close]
            forward_days: Days to look forward for price change

        Returns:
            DataFrame with recommendation accuracy
        """
        results = []
        price_df = price_df.copy()
        price_df['date'] = pd.to_datetime(price_df['date'])

        for entry in self.history.get("recommendations", []):
            rec_date = pd.to_datetime(entry["date"])
            forward_date = rec_date + timedelta(days=forward_days)

            for rec in entry.get("recommendations", []):
                ticker = rec.get("ticker")
                signal = rec.get("recommendation")

                # Get price on recommendation date
                price_at_rec = price_df[
                    (price_df['ticker'] == ticker) &
                    (price_df['date'] >= rec_date)
                ].head(1)

                # Get price after forward_days
                price_after = price_df[
                    (price_df['ticker'] == ticker) &
                    (price_df['date'] >= forward_date)
                ].head(1)

                if price_at_rec.empty or price_after.empty:
                    continue

                price_start = price_at_rec['close'].iloc[0]
                price_end = price_after['close'].iloc[0]
                actual_return = (price_end / price_start - 1) * 100

                # Determine if recommendation was correct
                if signal == "BUY":
                    correct = actual_return > 0
                elif signal == "SELL":
                    correct = actual_return < 0
                else:  # HOLD
                    correct = abs(actual_return) < 2  # Within 2%

                results.append({
                    "date": entry["date"],
                    "ticker": ticker,
                    "recommendation": signal,
                    "confidence": rec.get("confidence"),
                    "sentiment_score": rec.get("sentiment_score"),
                    "actual_return_pct": actual_return,
                    "correct": correct
                })

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)

    def get_summary_stats(self) -> Dict:
        """Get summary statistics from recommendation history."""
        history = self.history.get("recommendations", [])

        if not history:
            return {"error": "No history available"}

        total_sessions = len(history)

        # Count by signal type - handle both nested and flat formats
        all_recs = []
        for h in history:
            # Check if entry has nested recommendations or is flat
            if "recommendations" in h and isinstance(h["recommendations"], list):
                all_recs.extend(h["recommendations"])
            elif "ticker" in h:
                # Entry is itself a recommendation
                all_recs.append(h)

        total_recs = len(all_recs)

        buy_count = len([r for r in all_recs if r.get("recommendation") == "BUY"])
        hold_count = len([r for r in all_recs if r.get("recommendation") == "HOLD"])
        sell_count = len([r for r in all_recs if r.get("recommendation") == "SELL"])

        # Date range - handle both 'date' and 'timestamp' keys
        dates = []
        for h in history:
            if "date" in h:
                dates.append(h["date"])
            elif "timestamp" in h:
                dates.append(h["timestamp"][:10])  # Extract date from timestamp

        return {
            "total_sessions": total_sessions,
            "total_recommendations": total_recs,
            "date_range": {
                "start": min(dates) if dates else None,
                "end": max(dates) if dates else None
            },
            "signal_distribution": {
                "BUY": buy_count,
                "HOLD": hold_count,
                "SELL": sell_count
            },
            "avg_per_session": total_recs / total_sessions if total_sessions > 0 else 0
        }


def main():
    """Test the recommendation tracker."""
    tracker = RecommendationTracker()

    print("=" * 60)
    print("RECOMMENDATION TRACKER")
    print("=" * 60)

    # Show summary stats
    stats = tracker.get_summary_stats()
    print(f"\nSummary Statistics:")
    print(f"  Total sessions: {stats.get('total_sessions', 0)}")
    print(f"  Total recommendations: {stats.get('total_recommendations', 0)}")

    if stats.get('date_range'):
        print(f"  Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")

    if stats.get('signal_distribution'):
        print(f"\nSignal Distribution:")
        for signal, count in stats['signal_distribution'].items():
            print(f"  {signal}: {count}")

    # Show recent history
    recent = tracker.get_history(days=7)
    print(f"\nRecent history (last 7 days): {len(recent)} sessions")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
