"""
Debug Yahoo Finance connection
"""
import yfinance as yf
import requests
from datetime import datetime, timedelta

print("\n🔍 DEBUGGING YAHOO FINANCE\n")
print("="*60)

# Test 1: Basic connection
print("\n1️⃣ Testing basic yfinance (no session)...")
try:
    stock = yf.Ticker("AAPL")
    info = stock.info
    print(f"   ✓ Connection works!")
    print(f"   Company: {info.get('longName', 'Unknown')}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: With session + User-Agent
print("\n2️⃣ Testing with custom session...")
try:
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })

    stock = yf.Ticker("AAPL", session=session)
    info = stock.info
    print(f"   ✓ Session works!")
    print(f"   Company: {info.get('longName', 'Unknown')}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Historical data (short period)
print("\n3️⃣ Testing historical data (last 7 days)...")
try:
    stock = yf.Ticker("AAPL", session=session)

    # Use period instead of dates
    df = stock.history(period="5d")  # Last 5 trading days

    if df.empty:
        print(f"   ⚠️  DataFrame is empty!")
        print(f"   This might mean:")
        print(f"      - Market is closed")
        print(f"      - Yahoo Finance blocking requests")
        print(f"      - Network issue")
    else:
        print(f"   ✓ Got {len(df)} rows of data")
        print(f"\n   Latest data:")
        print(df.tail(2))

except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check yfinance version
print("\n4️⃣ Checking yfinance version...")
print(f"   yfinance version: {yf.__version__}")
print(f"   (Latest stable: 0.2.48+)")

# Test 5: Try download method (alternative)
print("\n5️⃣ Testing yf.download() method...")
try:
    start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    end = datetime.now().strftime('%Y-%m-%d')

    df = yf.download("AAPL", start=start, end=end, progress=False)

    if df.empty:
        print(f"   ⚠️  DataFrame is empty!")
    else:
        print(f"   ✓ Got {len(df)} rows")
        print(f"   Columns: {list(df.columns)}")

except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*60)
print("DEBUG COMPLETE")
print("="*60)
