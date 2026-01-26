"""
Create mock stock data for testing when Yahoo Finance doesn't work
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create dates
start = datetime.now() - timedelta(days=30)
dates = pd.date_range(start=start, periods=20, freq='D')

# Tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'META']

# Create mock data
all_data = []

for ticker in tickers:
    # Random walk for prices
    base_price = np.random.uniform(100, 500)
    # 0.1% avg return, 2% volatility
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))

    for i, date in enumerate(dates):
        price = prices[i]
        open_price = price * np.random.uniform(0.98, 1.02)
        high = max(open_price, price) * np.random.uniform(1.00, 1.03)
        low = min(open_price, price) * np.random.uniform(0.97, 1.00)
        volume = int(np.random.uniform(10_000_000, 100_000_000))

        all_data.append({
            'date': date.date(),
            'ticker': ticker,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(price, 2),
            'volume': volume
        })

df = pd.DataFrame(all_data)

# Calculate returns
df = df.sort_values(['ticker', 'date'])
df['daily_return'] = df.groupby('ticker')['close'].pct_change()

# Save
df.to_csv('data/processed/stock_prices.csv', index=False)

print(f"✅ Created mock data: {len(df)} rows for {len(tickers)} tickers")
print(f"\nSample:")
print(df.head(10))
print(f"\n✓ Saved to data/processed/stock_prices.csv")
