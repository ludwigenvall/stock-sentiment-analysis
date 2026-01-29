"""
Backtesting Module for Stock Sentiment Analysis

Tests how sentiment-based recommendations would have performed historically.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class Backtester:
    """
    Backtests sentiment-based stock recommendations.

    Simulates trading based on sentiment signals and calculates
    performance metrics like returns, Sharpe ratio, and accuracy.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        position_size: float = 0.1,  # 10% of capital per position
        sentiment_threshold_buy: float = 0.2,
        sentiment_threshold_sell: float = -0.2,
        holding_period: int = 5  # Days to hold position
    ):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital in dollars
            position_size: Fraction of capital per position (0.1 = 10%)
            sentiment_threshold_buy: Min sentiment to trigger buy
            sentiment_threshold_sell: Max sentiment to trigger sell
            holding_period: Days to hold each position
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.sentiment_threshold_buy = sentiment_threshold_buy
        self.sentiment_threshold_sell = sentiment_threshold_sell
        self.holding_period = holding_period

        self.trades = []
        self.portfolio_history = []

    def run_backtest(
        self,
        sentiment_df: pd.DataFrame,
        price_df: pd.DataFrame,
        start_date: str = None,
        end_date: str = None
    ) -> Dict:
        """
        Run full backtest on historical data.

        Args:
            sentiment_df: DataFrame with columns [ticker, date, sentiment_score]
            price_df: DataFrame with columns [ticker, date, close]
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)

        Returns:
            Dict with backtest results
        """
        # Prepare data
        sentiment_df = sentiment_df.copy()
        price_df = price_df.copy()

        # Ensure date columns are datetime
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        price_df['date'] = pd.to_datetime(price_df['date'])

        # Filter date range
        if start_date:
            start_dt = pd.to_datetime(start_date)
            sentiment_df = sentiment_df[sentiment_df['date'] >= start_dt]
            price_df = price_df[price_df['date'] >= start_dt]

        if end_date:
            end_dt = pd.to_datetime(end_date)
            sentiment_df = sentiment_df[sentiment_df['date'] <= end_dt]
            price_df = price_df[price_df['date'] <= end_dt]

        # Aggregate sentiment by ticker and date
        daily_sentiment = sentiment_df.groupby(['ticker', 'date']).agg({
            'sentiment_score': 'mean'
        }).reset_index()

        # Merge with prices
        merged = pd.merge(
            daily_sentiment,
            price_df[['ticker', 'date', 'close']],
            on=['ticker', 'date'],
            how='inner'
        )

        if merged.empty:
            logger.warning("No overlapping data between sentiment and prices")
            return self._empty_results()

        # Sort by date
        merged = merged.sort_values('date')
        dates = sorted(merged['date'].unique())

        # Initialize portfolio
        capital = self.initial_capital
        positions = {}  # {ticker: {'shares': n, 'entry_price': p, 'entry_date': d}}
        self.trades = []
        self.portfolio_history = []

        # Simulate trading
        for date in dates:
            day_data = merged[merged['date'] == date]

            # Check for exits (holding period reached)
            tickers_to_close = []
            for ticker, pos in positions.items():
                days_held = (date - pos['entry_date']).days
                if days_held >= self.holding_period:
                    tickers_to_close.append(ticker)

            # Close positions
            for ticker in tickers_to_close:
                ticker_price = day_data[day_data['ticker'] == ticker]
                if not ticker_price.empty:
                    exit_price = ticker_price['close'].iloc[0]
                    pos = positions[ticker]

                    # Calculate P&L
                    pnl = (exit_price - pos['entry_price']) * pos['shares']
                    pnl_pct = (exit_price / pos['entry_price'] - 1) * 100

                    capital += pos['shares'] * exit_price

                    self.trades.append({
                        'ticker': ticker,
                        'entry_date': pos['entry_date'],
                        'exit_date': date,
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'shares': pos['shares'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'sentiment': pos['sentiment'],
                        'direction': pos['direction']
                    })

                    del positions[ticker]

            # Check for new entries
            for _, row in day_data.iterrows():
                ticker = row['ticker']
                sentiment = row['sentiment_score']
                price = row['close']

                # Skip if already in position
                if ticker in positions:
                    continue

                # Check buy signal
                if sentiment >= self.sentiment_threshold_buy:
                    position_value = capital * self.position_size
                    shares = int(position_value / price)

                    if shares > 0 and position_value <= capital:
                        capital -= shares * price
                        positions[ticker] = {
                            'shares': shares,
                            'entry_price': price,
                            'entry_date': date,
                            'sentiment': sentiment,
                            'direction': 'long'
                        }

            # Calculate portfolio value
            portfolio_value = capital
            for ticker, pos in positions.items():
                ticker_price = day_data[day_data['ticker'] == ticker]
                if not ticker_price.empty:
                    current_price = ticker_price['close'].iloc[0]
                    portfolio_value += pos['shares'] * current_price

            self.portfolio_history.append({
                'date': date,
                'capital': capital,
                'positions_value': portfolio_value - capital,
                'total_value': portfolio_value,
                'num_positions': len(positions)
            })

        # Calculate results
        return self._calculate_results()

    def _calculate_results(self) -> Dict:
        """Calculate backtest performance metrics."""
        if not self.portfolio_history:
            return self._empty_results()

        portfolio_df = pd.DataFrame(self.portfolio_history)
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()

        # Basic metrics
        initial_value = self.initial_capital
        final_value = portfolio_df['total_value'].iloc[-1]
        total_return = (final_value / initial_value - 1) * 100

        # Calculate daily returns
        portfolio_df['daily_return'] = portfolio_df['total_value'].pct_change()

        # Sharpe ratio (assuming 252 trading days, 0% risk-free rate)
        if len(portfolio_df) > 1:
            avg_daily_return = portfolio_df['daily_return'].mean()
            std_daily_return = portfolio_df['daily_return'].std()
            sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return > 0 else 0
        else:
            sharpe_ratio = 0

        # Max drawdown
        portfolio_df['cummax'] = portfolio_df['total_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['total_value'] - portfolio_df['cummax']) / portfolio_df['cummax']
        max_drawdown = portfolio_df['drawdown'].min() * 100

        # Trade statistics
        if not trades_df.empty:
            num_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            win_rate = winning_trades / num_trades * 100 if num_trades > 0 else 0

            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

            avg_return_per_trade = trades_df['pnl_pct'].mean()
        else:
            num_trades = winning_trades = losing_trades = 0
            win_rate = avg_win = avg_loss = profit_factor = avg_return_per_trade = 0

        results = {
            'initial_capital': initial_value,
            'final_value': final_value,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'num_trades': num_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate_pct': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_return_per_trade_pct': avg_return_per_trade,
            'portfolio_history': portfolio_df.to_dict('records'),
            'trades': trades_df.to_dict('records') if not trades_df.empty else []
        }

        return results

    def _empty_results(self) -> Dict:
        """Return empty results structure."""
        return {
            'initial_capital': self.initial_capital,
            'final_value': self.initial_capital,
            'total_return_pct': 0,
            'sharpe_ratio': 0,
            'max_drawdown_pct': 0,
            'num_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate_pct': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'avg_return_per_trade_pct': 0,
            'portfolio_history': [],
            'trades': []
        }

    def print_results(self, results: Dict):
        """Print formatted backtest results."""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)

        print(f"\nPORTFOLIO PERFORMANCE:")
        print(f"  Initial Capital:    ${results['initial_capital']:,.2f}")
        print(f"  Final Value:        ${results['final_value']:,.2f}")
        print(f"  Total Return:       {results['total_return_pct']:+.2f}%")
        print(f"  Sharpe Ratio:       {results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:       {results['max_drawdown_pct']:.2f}%")

        print(f"\nTRADE STATISTICS:")
        print(f"  Total Trades:       {results['num_trades']}")
        print(f"  Winning Trades:     {results['winning_trades']}")
        print(f"  Losing Trades:      {results['losing_trades']}")
        print(f"  Win Rate:           {results['win_rate_pct']:.1f}%")
        print(f"  Avg Win:            ${results['avg_win']:,.2f}")
        print(f"  Avg Loss:           ${results['avg_loss']:,.2f}")
        print(f"  Profit Factor:      {results['profit_factor']:.2f}")
        print(f"  Avg Return/Trade:   {results['avg_return_per_trade_pct']:+.2f}%")

        print("\n" + "=" * 60)

    def save_results(self, results: Dict, output_dir: str = "data/backtest"):
        """Save backtest results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save portfolio history
        if results['portfolio_history']:
            portfolio_df = pd.DataFrame(results['portfolio_history'])
            portfolio_df.to_csv(output_path / "portfolio_history.csv", index=False)

        # Save trades
        if results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            trades_df.to_csv(output_path / "trades.csv", index=False)

        # Save summary
        summary = {k: v for k, v in results.items()
                   if k not in ['portfolio_history', 'trades']}
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(output_path / "backtest_summary.csv", index=False)

        logger.info(f"Results saved to {output_path}")


def run_backtest_from_files(
    sentiment_file: str = "data/processed/all_sentiment.csv",
    price_file: str = "data/processed/stock_prices.csv",
    **kwargs
) -> Dict:
    """
    Convenience function to run backtest from CSV files.

    Args:
        sentiment_file: Path to sentiment data CSV
        price_file: Path to price data CSV
        **kwargs: Additional arguments for Backtester

    Returns:
        Backtest results dict
    """
    # Load data
    sentiment_df = pd.read_csv(sentiment_file)
    price_df = pd.read_csv(price_file)

    # Run backtest
    backtester = Backtester(**kwargs)
    results = backtester.run_backtest(sentiment_df, price_df)

    # Print and save
    backtester.print_results(results)
    backtester.save_results(results)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Run backtest with default files
    try:
        results = run_backtest_from_files(
            initial_capital=100000,
            position_size=0.1,
            sentiment_threshold_buy=0.2,
            holding_period=5
        )
    except FileNotFoundError as e:
        print(f"Data files not found: {e}")
        print("Run 'python analyze_with_recommendations.py' first to generate data.")
