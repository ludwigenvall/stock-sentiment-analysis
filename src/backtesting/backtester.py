"""
Backtesting Module for Stock Sentiment Analysis

Tests how sentiment-based recommendations would have performed historically.
Includes benchmark comparison against S&P 500 (SPY).
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    yf = None

logger = logging.getLogger(__name__)


def get_benchmark_data(start_date: str, end_date: str, ticker: str = "SPY") -> pd.DataFrame:
    """
    Fetch benchmark (SPY) data for comparison.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        ticker: Benchmark ticker (default: SPY for S&P 500)

    Returns:
        DataFrame with date and daily returns
    """
    if yf is None:
        logger.warning("yfinance not installed, cannot fetch benchmark data")
        return pd.DataFrame()

    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return pd.DataFrame()

        benchmark_df = pd.DataFrame({
            'date': data.index,
            'close': data['Close'].values,
            'daily_return': data['Close'].pct_change().values
        })
        benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
        return benchmark_df.dropna()
    except Exception as e:
        logger.error(f"Failed to fetch benchmark data: {e}")
        return pd.DataFrame()


def calculate_benchmark_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02
) -> Dict:
    """
    Calculate benchmark-adjusted performance metrics.

    Args:
        portfolio_returns: Daily portfolio returns
        benchmark_returns: Daily benchmark returns
        risk_free_rate: Annual risk-free rate (default 2%)

    Returns:
        Dict with alpha, beta, information_ratio, sortino_ratio
    """
    # Align the series
    aligned = pd.DataFrame({
        'portfolio': portfolio_returns,
        'benchmark': benchmark_returns
    }).dropna()

    if len(aligned) < 30:
        return {
            'alpha': 0,
            'beta': 0,
            'information_ratio': 0,
            'sortino_ratio': 0,
            'tracking_error': 0,
            'benchmark_return': 0,
            'benchmark_sharpe': 0
        }

    port_ret = aligned['portfolio']
    bench_ret = aligned['benchmark']

    # Daily risk-free rate
    rf_daily = risk_free_rate / 252

    # Beta: Cov(portfolio, benchmark) / Var(benchmark)
    covariance = port_ret.cov(bench_ret)
    benchmark_variance = bench_ret.var()
    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

    # Annualized returns
    portfolio_annual = port_ret.mean() * 252
    benchmark_annual = bench_ret.mean() * 252

    # Alpha: Portfolio return - (Risk-free + Beta * (Benchmark - Risk-free))
    alpha = portfolio_annual - (risk_free_rate + beta * (benchmark_annual - risk_free_rate))

    # Tracking Error: Std of excess returns
    excess_returns = port_ret - bench_ret
    tracking_error = excess_returns.std() * np.sqrt(252)

    # Information Ratio: Excess return / Tracking error
    excess_return_annual = portfolio_annual - benchmark_annual
    information_ratio = excess_return_annual / tracking_error if tracking_error > 0 else 0

    # Sortino Ratio: (Return - Rf) / Downside deviation
    excess_over_rf = port_ret - rf_daily
    downside_returns = excess_over_rf[excess_over_rf < 0]
    downside_deviation = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = (portfolio_annual - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0

    # Benchmark Sharpe
    benchmark_sharpe = (benchmark_annual - risk_free_rate) / (bench_ret.std() * np.sqrt(252)) if bench_ret.std() > 0 else 0

    return {
        'alpha': alpha * 100,  # As percentage
        'beta': beta,
        'information_ratio': information_ratio,
        'sortino_ratio': sortino_ratio,
        'tracking_error': tracking_error * 100,  # As percentage
        'benchmark_return': benchmark_annual * 100,  # As percentage
        'benchmark_sharpe': benchmark_sharpe
    }


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
        self.benchmark_data = None

    def run_backtest(
        self,
        sentiment_df: pd.DataFrame,
        price_df: pd.DataFrame,
        start_date: str = None,
        end_date: str = None,
        include_benchmark: bool = True
    ) -> Dict:
        """
        Run full backtest on historical data.

        Args:
            sentiment_df: DataFrame with columns [ticker, date, sentiment_score]
            price_df: DataFrame with columns [ticker, date, close]
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            include_benchmark: Whether to include SPY benchmark comparison

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

        # Fetch benchmark data if requested
        self.benchmark_data = None
        if include_benchmark:
            actual_start = dates[0].strftime('%Y-%m-%d')
            actual_end = dates[-1].strftime('%Y-%m-%d')
            self.benchmark_data = get_benchmark_data(actual_start, actual_end)
            if not self.benchmark_data.empty:
                logger.info(f"Loaded {len(self.benchmark_data)} days of SPY benchmark data")

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

        # Add benchmark comparison if available
        if self.benchmark_data is not None and not self.benchmark_data.empty:
            # Merge portfolio and benchmark data
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            benchmark_merged = pd.merge(
                portfolio_df[['date', 'daily_return']],
                self.benchmark_data[['date', 'daily_return']].rename(columns={'daily_return': 'benchmark_return'}),
                on='date',
                how='inner'
            )

            if len(benchmark_merged) > 0:
                benchmark_metrics = calculate_benchmark_metrics(
                    benchmark_merged['daily_return'],
                    benchmark_merged['benchmark_return']
                )
                results['benchmark'] = benchmark_metrics

                # Add benchmark cumulative returns to portfolio history
                self.benchmark_data['cumulative_return'] = (1 + self.benchmark_data['daily_return']).cumprod() - 1
                results['benchmark_history'] = self.benchmark_data[['date', 'close', 'cumulative_return']].to_dict('records')

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
            'trades': [],
            'benchmark': None,
            'benchmark_history': []
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

        # Benchmark comparison
        if results.get('benchmark'):
            bench = results['benchmark']
            print(f"\nBENCHMARK COMPARISON (vs S&P 500):")
            print(f"  Benchmark Return:   {bench['benchmark_return']:+.2f}%")
            print(f"  Alpha:              {bench['alpha']:+.2f}%")
            print(f"  Beta:               {bench['beta']:.2f}")
            print(f"  Information Ratio:  {bench['information_ratio']:.2f}")
            print(f"  Sortino Ratio:      {bench['sortino_ratio']:.2f}")
            print(f"  Tracking Error:     {bench['tracking_error']:.2f}%")
            print(f"  Benchmark Sharpe:   {bench['benchmark_sharpe']:.2f}")

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

        # Save benchmark history if available
        if results.get('benchmark_history'):
            benchmark_df = pd.DataFrame(results['benchmark_history'])
            benchmark_df.to_csv(output_path / "benchmark_history.csv", index=False)

        # Save summary (flatten benchmark metrics)
        summary = {k: v for k, v in results.items()
                   if k not in ['portfolio_history', 'trades', 'benchmark', 'benchmark_history']}
        if results.get('benchmark'):
            for key, value in results['benchmark'].items():
                summary[f'benchmark_{key}'] = value
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
