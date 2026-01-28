"""
Quick test to verify API keys work
"""
import os
from dotenv import load_dotenv
import requests
import praw

load_dotenv()


def test_alpha_vantage():
    """Test Alpha Vantage API"""
    print("\n" + "="*60)
    print("TESTING ALPHA VANTAGE API")
    print("="*60)

    key = os.getenv('ALPHA_VANTAGE_KEY')
    if not key:
        print("❌ ALPHA_VANTAGE_KEY not found in .env")
        return False

    print(f"Key found: {key[:10]}... (first 10 chars)")

    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=IBM&apikey={key}"

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if 'Global Quote' in data:
            print("✅ Alpha Vantage API works!")
            price = data['Global Quote'].get('05. price', 'N/A')
            print(f"   Test query: IBM stock price = ${price}")
            return True
        elif 'Note' in data:
            print("⚠️  API call limit reached (max 25/day on free tier)")
            print("   But key is valid! This is expected on free tier.")
            return True
        else:
            print(f"❌ Unexpected response: {data}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_reddit():
    """Test Reddit API"""
    print("\n" + "="*60)
    print("TESTING REDDIT API")
    print("="*60)

    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    user_agent = os.getenv('REDDIT_USER_AGENT')

    if not all([client_id, client_secret, user_agent]):
        print("❌ Reddit credentials not found in .env")
        print(
            f"   Found: client_id={bool(client_id)}, secret={bool(client_secret)}, user_agent={bool(user_agent)}")
        return False

    print(f"Client ID: {client_id[:8]}... (first 8 chars)")
    print(f"User Agent: {user_agent}")

    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )

        # Try to fetch one post from r/wallstreetbets
        subreddit = reddit.subreddit('wallstreetbets')
        post = next(subreddit.hot(limit=1))

        print("✅ Reddit API works!")
        print(f"   Latest WSB post: '{post.title[:60]}...'")
        print(f"   Score: {post.score} | Comments: {post.num_comments}")
        return True

    except Exception as e:
        print(f"❌ Reddit error: {e}")
        return False


if __name__ == "__main__":
    print("\n🔑 TESTING API KEYS")
    print("Make sure you have .env file configured!\n")

    # Test each API
    alpha_ok = test_alpha_vantage()
    reddit_ok = test_reddit()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if alpha_ok and reddit_ok:
        print("✅ ALL API KEYS WORKING!")
        print("\nYou're ready to continue with News + Reddit scrapers!")
    else:
        if not alpha_ok:
            print("⚠️  Alpha Vantage: Check your key at https://www.alphavantage.co/")
        if not reddit_ok:
            print("⚠️  Reddit: Check credentials at https://www.reddit.com/prefs/apps")

    print("="*60)
