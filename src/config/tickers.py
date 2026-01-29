"""
Stock Ticker Lists

Contains various lists of stock tickers for different analysis scopes.
"""
from typing import List

# S&P 100 Components (100 largest US companies)
SP100_TICKERS = [
    # Technology
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'AVGO', 'CSCO', 'ADBE', 'CRM',
    'ORCL', 'ACN', 'IBM', 'INTC', 'AMD', 'QCOM', 'TXN', 'AMAT', 'ADI', 'MU',

    # Consumer & Retail
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'COST', 'WMT',

    # Financial
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'V',
    'MA', 'PYPL', 'COF', 'USB', 'PNC',

    # Healthcare & Pharma
    'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY',
    'AMGN', 'GILD', 'MDT', 'CVS', 'CI',

    # Industrial & Manufacturing
    'CAT', 'BA', 'HON', 'GE', 'MMM', 'UPS', 'RTX', 'LMT', 'DE', 'EMR',

    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'VLO', 'PSX', 'OXY',

    # Communication
    'DIS', 'CMCSA', 'NFLX', 'T', 'VZ', 'TMUS', 'CHTR',

    # Consumer Goods
    'PG', 'KO', 'PEP', 'PM', 'MO', 'CL', 'EL', 'MDLZ', 'KHC', 'GIS',

    # Utilities & Real Estate
    'NEE', 'DUK', 'SO', 'D', 'AEP',

    # Other
    'BRK-B', 'SPY', 'QQQ'
]

# Top 10 Tech Stocks
TOP_TECH = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'ADBE', 'CRM'
]

# Original 7 (backwards compatibility)
ORIGINAL_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'META']

# Meme Stocks (popular on Reddit)
MEME_STOCKS = [
    'GME', 'AMC', 'BB', 'BBBY', 'NOK', 'PLTR', 'WISH', 'CLOV', 'SOFI', 'RIVN'
]

# High Growth / Volatile
HIGH_GROWTH = [
    'TSLA', 'NVDA', 'AMD', 'SQ', 'SHOP', 'ROKU', 'SNOW', 'NET', 'CRWD', 'DDOG',
    'ZS', 'OKTA', 'TWLO', 'MDB', 'FSLY'
]

# Dividend Stocks
DIVIDEND_STOCKS = [
    'JNJ', 'PG', 'KO', 'PEP', 'MCD', 'T', 'VZ', 'XOM', 'CVX', 'IBM',
    'MMM', 'CAT', 'HD', 'WMT', 'ABT'
]

# Value Stocks (potentially undervalued, low P/E)
VALUE_STOCKS = [
    # Financials (often undervalued)
    'BRK-B', 'JPM', 'BAC', 'C', 'WFC', 'USB', 'PNC', 'TFC', 'FITB', 'KEY',
    # Energy (cyclical, often undervalued)
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'BKR', 'DVN', 'FANG', 'MRO',
    # Healthcare value
    'CVS', 'CI', 'HUM', 'ANTM', 'WBA', 'CAH', 'MCK',
    # Industrial value
    'GE', 'F', 'GM', 'DAL', 'UAL', 'LUV', 'AAL',
    # Telecom (high dividend, potential value)
    'T', 'VZ', 'TMUS', 'LUMN',
    # Retail value
    'KSS', 'M', 'GPS', 'BBY', 'DG', 'DLTR'
]

# Small/Mid Cap Growth Potential
SMALL_MID_CAP = [
    # Tech small caps
    'CRWD', 'ZS', 'DDOG', 'NET', 'SNOW', 'MDB', 'CFLT', 'PATH', 'S', 'GTLB',
    # Fintech
    'SOFI', 'AFRM', 'UPST', 'HOOD', 'COIN', 'NU', 'BILL', 'TOST',
    # Healthcare/Biotech small caps
    'EXAS', 'HALO', 'RARE', 'BEAM', 'CRSP', 'NTLA', 'EDIT', 'VERV',
    # Clean Energy
    'ENPH', 'SEDG', 'RUN', 'NOVA', 'FSLR', 'PLUG', 'BE', 'CHPT', 'LCID', 'RIVN',
    # E-commerce/Consumer
    'SHOP', 'ETSY', 'CHWY', 'W', 'PTON', 'FVRR', 'UPWK'
]

# Emerging Sectors (AI, Quantum, Space, etc.)
EMERGING_TECH = [
    # AI & Machine Learning
    'NVDA', 'AMD', 'GOOGL', 'MSFT', 'PLTR', 'AI', 'BBAI', 'SOUN', 'UPST',
    # Semiconductors
    'TSM', 'ASML', 'KLAC', 'LRCX', 'AMAT', 'MRVL', 'ON', 'SWKS', 'QRVO',
    # Cybersecurity
    'CRWD', 'PANW', 'ZS', 'FTNT', 'OKTA', 'CYBR', 'QLYS', 'TENB', 'RPD',
    # Cloud Infrastructure
    'AMZN', 'MSFT', 'GOOGL', 'SNOW', 'NET', 'DDOG', 'MDB', 'ESTC',
    # Space & Defense Tech
    'RKLB', 'SPCE', 'ASTS', 'RDW', 'LUNR', 'LMT', 'NOC', 'RTX', 'BA'
]

# Contrarian Picks (beaten down stocks that may recover)
CONTRARIAN = [
    'INTC', 'PARA', 'WBD', 'PYPL', 'SNAP', 'PINS', 'ZM', 'DOCU', 'PTON',
    'ROKU', 'TWLO', 'OKTA', 'NET', 'U', 'RBLX', 'MTCH', 'BMBL', 'ABNB',
    'DASH', 'LYFT', 'UBER', 'GRAB', 'SE', 'BABA', 'JD', 'PDD', 'NIO', 'XPEV', 'LI'
]

# All available ticker lists
TICKER_LISTS = {
    'sp100': SP100_TICKERS,
    'top_tech': TOP_TECH,
    'original': ORIGINAL_TICKERS,
    'meme': MEME_STOCKS,
    'high_growth': HIGH_GROWTH,
    'dividend': DIVIDEND_STOCKS,
    'value': VALUE_STOCKS,
    'small_mid': SMALL_MID_CAP,
    'emerging': EMERGING_TECH,
    'contrarian': CONTRARIAN
}


def get_tickers(list_name: str = 'original') -> List[str]:
    """
    Get a list of tickers by name.

    Args:
        list_name: Name of the ticker list:
            - 'sp100': S&P 100 components (~100 tickers)
            - 'top_tech': Top 10 tech stocks
            - 'original': Original 7 tickers
            - 'meme': Meme/Reddit popular stocks
            - 'high_growth': High growth/volatile stocks
            - 'dividend': Stable dividend stocks

    Returns:
        List of ticker symbols
    """
    if list_name not in TICKER_LISTS:
        print(f"Unknown list '{list_name}'. Available lists: {list(TICKER_LISTS.keys())}")
        return ORIGINAL_TICKERS

    return TICKER_LISTS[list_name]


def get_all_tickers() -> List[str]:
    """Get all unique tickers across all lists"""
    all_tickers = set()
    for tickers in TICKER_LISTS.values():
        all_tickers.update(tickers)
    return sorted(list(all_tickers))


def main():
    """Show available ticker lists"""
    print("=" * 60)
    print("AVAILABLE TICKER LISTS")
    print("=" * 60)

    for name, tickers in TICKER_LISTS.items():
        print(f"\n{name.upper()} ({len(tickers)} tickers):")
        print(f"  {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")

    print(f"\nTotal unique tickers: {len(get_all_tickers())}")


if __name__ == "__main__":
    main()
