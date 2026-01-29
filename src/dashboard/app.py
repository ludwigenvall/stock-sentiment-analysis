"""
Streamlit Dashboard för Stock Market Sentiment Analysis
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Page config
st.set_page_config(
    page_title="Stock Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants - find project root dynamically
def find_project_root():
    """Find project root by looking for data directory"""
    # Try relative to this file
    path1 = Path(__file__).parent.parent.parent / "data"
    if path1.exists():
        return path1.parent

    # Try current working directory
    path2 = Path.cwd() / "data"
    if path2.exists():
        return Path.cwd()

    # Try parent of cwd (sometimes Streamlit runs from src/)
    path3 = Path.cwd().parent / "data"
    if path3.exists():
        return Path.cwd().parent

    # Fallback to relative path
    return Path(__file__).parent.parent.parent

PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'META']


@st.cache_data(ttl=300)  # Cache for 5 minutes, then refresh
def load_data():
    """Load all available data files"""
    # Use the dynamically computed paths
    processed_dir = PROJECT_ROOT / "data" / "processed"
    raw_dir = PROJECT_ROOT / "data" / "raw"

    data = {}

    # Load news articles
    news_file = raw_dir / "news_articles.csv"
    if news_file.exists():
        df = pd.read_csv(news_file)
        df['time_published'] = pd.to_datetime(df['time_published'])
        df['date'] = df['time_published'].dt.date
        data['news'] = df
    else:
        data['news'] = None

    # Load Reddit posts
    reddit_file = raw_dir / "reddit_posts.csv"
    if reddit_file.exists():
        df = pd.read_csv(reddit_file)
        df['created_utc'] = pd.to_datetime(df['created_utc'])
        df['date'] = df['created_utc'].dt.date
        data['reddit'] = df
    else:
        data['reddit'] = None

    # Load stock prices
    stock_file = processed_dir / "stock_prices.csv"
    if stock_file.exists():
        df = pd.read_csv(stock_file)
        df['date'] = pd.to_datetime(df['date']).dt.date
        data['stock'] = df
    else:
        data['stock'] = None

    # Load combined data (if available)
    combined_file = processed_dir / "stock_sentiment_combined.csv"
    if combined_file.exists():
        df = pd.read_csv(combined_file)
        df['date'] = pd.to_datetime(df['date']).dt.date
        data['combined'] = df
    else:
        data['combined'] = None

    # Load combined with Reddit (if available)
    reddit_combined_file = processed_dir / "stock_sentiment_reddit_combined.csv"
    if reddit_combined_file.exists():
        df = pd.read_csv(reddit_combined_file)
        df['date'] = pd.to_datetime(df['date']).dt.date
        data['reddit_combined'] = df
    else:
        data['reddit_combined'] = None

    # Load all content with sentiment (News + Reddit + Twitter + SEC + Earnings)
    all_content_file = processed_dir / "all_sentiment.csv"
    if all_content_file.exists():
        df = pd.read_csv(all_content_file)
        df['date'] = pd.to_datetime(df['date'])
        data['all_content'] = df
    else:
        # Try alternate filename
        all_content_file = processed_dir / "all_content_with_sentiment.csv"
        if all_content_file.exists():
            df = pd.read_csv(all_content_file)
            df['date'] = pd.to_datetime(df['date'])
            data['all_content'] = df
        else:
            data['all_content'] = None

    # Load news with sentiment (if available)
    news_sentiment_file = processed_dir / "news_with_sentiment.csv"
    if news_sentiment_file.exists():
        df = pd.read_csv(news_sentiment_file)
        df['time_published'] = pd.to_datetime(df['time_published'])
        df['date'] = df['time_published'].dt.date
        data['news_sentiment'] = df
    else:
        data['news_sentiment'] = None

    return data


def get_sentiment_label(score):
    """Convert sentiment score to label"""
    if score > 0.15:
        return 'positive'
    elif score < -0.15:
        return 'negative'
    else:
        return 'neutral'


def get_sentiment_color(label):
    """Get color for sentiment label"""
    colors = {
        'positive': '#00CC66',
        'negative': '#FF4B4B',
        'neutral': '#808080'
    }
    return colors.get(label, '#808080')


def create_price_chart(stock_df, selected_ticker):
    """Create stock price line chart"""
    df = stock_df[stock_df['ticker'] == selected_ticker].copy()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#1f77b4', width=2)
    ))

    fig.update_layout(
        title=f"{selected_ticker} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=400
    )

    return fig


def create_candlestick_chart(stock_df, selected_ticker):
    """Create candlestick chart with volume"""
    df = stock_df[stock_df['ticker'] == selected_ticker].copy()

    # Create figure with secondary y-axis
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='OHLC'
    ))

    fig.update_layout(
        title=f"{selected_ticker} - OHLC Chart",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        xaxis_rangeslider_visible=False
    )

    return fig


def create_volume_chart(stock_df, selected_ticker):
    """Create volume bar chart"""
    df = stock_df[stock_df['ticker'] == selected_ticker].copy()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['date'],
        y=df['volume'],
        name='Volume',
        marker_color='rgba(31, 119, 180, 0.5)'
    ))

    fig.update_layout(
        title=f"{selected_ticker} Trading Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        height=250
    )

    return fig


def create_sentiment_timeline(news_df, selected_tickers):
    """Create sentiment timeline for selected tickers"""
    df = news_df[news_df['ticker'].isin(selected_tickers)].copy()

    # Group by date and ticker
    daily_sentiment = df.groupby(['date', 'ticker'])['sentiment_score'].mean().reset_index()

    fig = px.line(
        daily_sentiment,
        x='date',
        y='sentiment_score',
        color='ticker',
        title='Sentiment Timeline',
        labels={'sentiment_score': 'Avg Sentiment', 'date': 'Date'}
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=400)

    return fig


def create_correlation_scatter(combined_df, selected_ticker):
    """Create scatter plot for sentiment vs returns correlation"""
    # Check if daily_return exists, calculate if not
    if 'daily_return' not in combined_df.columns and 'close' in combined_df.columns:
        combined_df = combined_df.sort_values(['ticker', 'date'])
        combined_df['daily_return'] = combined_df.groupby('ticker')['close'].pct_change() * 100

    # Check if we have the required columns
    if 'daily_return' not in combined_df.columns or 'avg_sentiment' not in combined_df.columns:
        return None

    df = combined_df[combined_df['ticker'] == selected_ticker].dropna(
        subset=['avg_sentiment', 'daily_return']
    )

    if len(df) == 0:
        return None

    fig = px.scatter(
        df,
        x='avg_sentiment',
        y='daily_return',
        title=f'{selected_ticker} - Sentiment vs Daily Return',
        labels={'avg_sentiment': 'Average Sentiment', 'daily_return': 'Daily Return (%)'},
        trendline="ols"
    )

    fig.update_layout(height=400)

    return fig


def create_sentiment_distribution(news_df):
    """Create pie chart for sentiment distribution"""
    # Add sentiment labels
    news_df['sentiment_label'] = news_df['sentiment_score'].apply(get_sentiment_label)

    sentiment_counts = news_df['sentiment_label'].value_counts()

    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title='Sentiment Distribution',
        color=sentiment_counts.index,
        color_discrete_map={
            'positive': '#00CC66',
            'negative': '#FF4B4B',
            'neutral': '#808080'
        }
    )

    fig.update_layout(height=400)

    return fig


def main():
    # Header
    st.title("📊 Stock Market Sentiment Analysis Dashboard")

    # Load data
    data = load_data()

    # Check if we have any data
    if data['news'] is None and data['stock'] is None and data['reddit'] is None and data['all_content'] is None:
        st.error("❌ No data found! Please run `python analyze_news_sentiment.py` or `python analyze_with_reddit.py` first.")
        # Debug info
        with st.expander("🔧 Debug Info"):
            st.write(f"**PROJECT_ROOT:** {PROJECT_ROOT}")
            st.write(f"**DATA_DIR:** {DATA_DIR}")
            st.write(f"**DATA_DIR exists:** {DATA_DIR.exists()}")
            st.write(f"**PROCESSED_DIR:** {PROCESSED_DIR}")
            st.write(f"**PROCESSED_DIR exists:** {PROCESSED_DIR.exists()}")
            if PROCESSED_DIR.exists():
                st.write(f"**Files in PROCESSED_DIR:** {list(PROCESSED_DIR.iterdir())}")
            st.write(f"**CWD:** {Path.cwd()}")
            st.write(f"**__file__:** {__file__}")
        st.stop()

    # Get last update time and show data sources
    sources = []
    if data['news'] is not None:
        last_update = data['news']['time_published'].max()
        st.caption(f"📅 Last updated: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        sources.append("News")
    elif data['all_content'] is not None:
        last_update = pd.to_datetime(data['all_content']['date']).max()
        st.caption(f"📅 Last updated: {last_update.strftime('%Y-%m-%d')}")
    if data['reddit'] is not None:
        sources.append("Reddit")
    if data['all_content'] is not None and 'content_type' in data['all_content'].columns:
        content_types = data['all_content']['content_type'].unique()
        if 'sec_filing' in content_types:
            sources.append("SEC Filings")
        if 'earnings_call' in content_types:
            sources.append("Earnings Calls")
    if data['stock'] is not None:
        sources.append("Stock Prices")
    sources.append("FinBERT")

    st.info(f"📌 Data sources: {', '.join(sources)}")

    # Sidebar
    st.sidebar.header("⚙️ Settings")

    # Ticker selection
    if data['news'] is not None:
        available_tickers = sorted(data['news']['ticker'].unique())
    elif data['stock'] is not None:
        available_tickers = sorted(data['stock']['ticker'].unique())
    else:
        available_tickers = TICKERS

    selected_tickers = st.sidebar.multiselect(
        "Select Tickers",
        options=available_tickers,
        default=available_tickers[:3] if len(available_tickers) >= 3 else available_tickers
    )

    # Content source filter
    content_sources = []
    if data['all_content'] is not None and 'content_type' in data['all_content'].columns:
        # Get unique content types from data
        available_types = data['all_content']['content_type'].unique().tolist()
        type_labels = {
            'news': 'News',
            'reddit': 'Reddit',
            'sec_filing': 'SEC Filings',
            'earnings_call': 'Earnings Calls',
            'earnings': 'Earnings'
        }
        content_sources = [type_labels.get(t, t.title()) for t in available_types]
        content_sources.append('All Sources')

        content_filter = st.sidebar.multiselect(
            "Content Sources",
            options=content_sources,
            default=['All Sources']
        )
    else:
        content_filter = None

    # Date range
    if data['news'] is not None:
        min_date = data['news']['date'].min()
        max_date = data['news']['date'].max()
    elif data['stock'] is not None:
        min_date = data['stock']['date'].min()
        max_date = data['stock']['date'].max()
    else:
        min_date = datetime.now().date() - timedelta(days=30)
        max_date = datetime.now().date()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Info button
    st.sidebar.markdown("---")
    with st.sidebar.expander("ℹ️ About"):
        st.markdown("""
        This dashboard visualizes stock market sentiment analysis.

        **Features:**
        - Stock price tracking
        - News sentiment analysis (Alpha Vantage)
        - Reddit sentiment analysis
        - SEC filings analysis
        - Earnings data analysis
        - AI-powered stock recommendations
        - Correlation analysis
        - Multi-ticker comparison

        **Data Sources:**
        - Alpha Vantage (Financial News)
        - Reddit (Social Sentiment)
        - SEC EDGAR (Official Filings)
        - Yahoo Finance (Stock Prices + Earnings)
        - FinBERT (Sentiment Analysis)

        **Run Analysis:**
        ```
        python analyze_with_recommendations.py --list sp100
        ```
        """)

    # Filter data by selected tickers and date range
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range[0]

    # Filter news
    if data['news'] is not None:
        news_filtered = data['news'][
            (data['news']['ticker'].isin(selected_tickers)) &
            (data['news']['date'] >= start_date) &
            (data['news']['date'] <= end_date)
        ].copy()
    else:
        news_filtered = None

    # Filter stock
    if data['stock'] is not None:
        stock_filtered = data['stock'][
            (data['stock']['ticker'].isin(selected_tickers)) &
            (data['stock']['date'] >= start_date) &
            (data['stock']['date'] <= end_date)
        ].copy()
    else:
        stock_filtered = None

    # Filter combined
    if data['combined'] is not None:
        combined_filtered = data['combined'][
            (data['combined']['ticker'].isin(selected_tickers)) &
            (data['combined']['date'] >= start_date) &
            (data['combined']['date'] <= end_date)
        ].copy()
    else:
        combined_filtered = None

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 Prices & Sentiment",
        "📰 News Analysis",
        "🎯 Per-Ticker Analysis",
        "🔍 Data Explorer"
    ])

    # TAB 1: OVERVIEW
    with tab1:
        st.header("Overview")

        # Use all_content if available (has multiple sources)
        overview_df = None
        if data['all_content'] is not None:
            overview_df = data['all_content'][
                (data['all_content']['ticker'].isin(selected_tickers))
            ].copy()
            if 'date' in overview_df.columns:
                overview_df['date'] = pd.to_datetime(overview_df['date']).dt.date
                overview_df = overview_df[
                    (overview_df['date'] >= start_date) &
                    (overview_df['date'] <= end_date)
                ]
        elif news_filtered is not None:
            overview_df = news_filtered

        if overview_df is not None and len(overview_df) > 0:
            # KPIs
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Analyzed Tickers", len(selected_tickers))

            with col2:
                st.metric("Total Content", len(overview_df))

            with col3:
                avg_sentiment = overview_df['sentiment_score'].mean()
                st.metric("Avg Sentiment", f"{avg_sentiment:.3f}")

            with col4:
                if 'sentiment_label' not in overview_df.columns:
                    overview_df['sentiment_label'] = overview_df['sentiment_score'].apply(get_sentiment_label)
                positive_pct = (overview_df['sentiment_label'] == 'positive').sum() / len(overview_df) * 100
                st.metric("Positive %", f"{positive_pct:.1f}%")

            # Content type breakdown (if multiple sources)
            if 'content_type' in overview_df.columns:
                st.subheader("Content by Source")
                type_counts = overview_df['content_type'].value_counts()
                type_labels = {
                    'news': '📰 News',
                    'reddit': '💬 Reddit',
                    'sec_filing': '📋 SEC Filings',
                    'earnings_call': '📞 Earnings Calls',
                    'earnings': '📊 Earnings'
                }

                cols = st.columns(len(type_counts))
                for i, (ctype, count) in enumerate(type_counts.items()):
                    with cols[i]:
                        label = type_labels.get(ctype, ctype.title())
                        avg_sent = overview_df[overview_df['content_type'] == ctype]['sentiment_score'].mean()
                        st.metric(label, count, f"Avg: {avg_sent:+.2f}")

            # Sentiment distribution
            st.plotly_chart(create_sentiment_distribution(overview_df), width="stretch")

            # Sentiment by ticker
            st.subheader("Sentiment by Ticker")
            ticker_sentiment = overview_df.groupby('ticker').agg({
                'sentiment_score': ['mean', 'std', 'count']
            }).reset_index()
            ticker_sentiment.columns = ['Ticker', 'Avg Sentiment', 'Std Dev', 'Content Count']
            ticker_sentiment = ticker_sentiment.sort_values('Avg Sentiment', ascending=False)

            st.dataframe(ticker_sentiment, width="stretch", hide_index=True)
        else:
            st.warning("No data available for selected filters.")

    # TAB 2: PRICES & SENTIMENT
    with tab2:
        st.header("Stock Prices & Sentiment")

        if stock_filtered is not None and len(stock_filtered) > 0:
            # Ticker selector for charts
            chart_ticker = st.selectbox(
                "Select ticker for detailed view",
                options=selected_tickers,
                key='price_ticker'
            )

            # Price chart
            st.plotly_chart(
                create_price_chart(stock_filtered, chart_ticker),
                width="stretch"
            )

            # Sentiment timeline
            if news_filtered is not None and len(news_filtered) > 0:
                st.plotly_chart(
                    create_sentiment_timeline(news_filtered, selected_tickers),
                    width="stretch"
                )

            # Correlation analysis
            if combined_filtered is not None and len(combined_filtered) > 0:
                st.subheader("Correlation Analysis")

                col1, col2 = st.columns([2, 1])

                with col1:
                    scatter_fig = create_correlation_scatter(combined_filtered, chart_ticker)
                    if scatter_fig:
                        st.plotly_chart(scatter_fig, width="stretch")
                    else:
                        st.info("Not enough data for correlation analysis")

                with col2:
                    # Calculate correlation
                    # Calculate daily_return if not present
                    combined_temp = combined_filtered.copy()
                    if 'daily_return' not in combined_temp.columns and 'close' in combined_temp.columns:
                        combined_temp = combined_temp.sort_values(['ticker', 'date'])
                        combined_temp['daily_return'] = combined_temp.groupby('ticker')['close'].pct_change() * 100

                    if 'daily_return' in combined_temp.columns and 'avg_sentiment' in combined_temp.columns:
                        ticker_data = combined_temp[combined_temp['ticker'] == chart_ticker].dropna(
                            subset=['avg_sentiment', 'daily_return']
                        )
                        if len(ticker_data) > 1:
                            correlation = ticker_data['avg_sentiment'].corr(ticker_data['daily_return'])
                            st.metric("Correlation Coefficient", f"{correlation:.3f}")

                            if abs(correlation) > 0.3:
                                st.success("Moderate correlation detected")
                            else:
                                st.info("Weak correlation (more data needed)")
                        else:
                            st.info("Not enough data points")
                    else:
                        st.info("Missing required columns for correlation")
            else:
                st.info("💡 Run `python analyze_news_sentiment.py` to generate correlation analysis")
        else:
            st.warning("No stock data available for selected filters.")

    # TAB 3: NEWS ANALYSIS
    with tab3:
        st.header("News Analysis")

        if news_filtered is not None and len(news_filtered) > 0:
            # Add sentiment labels
            news_filtered['sentiment_label'] = news_filtered['sentiment_score'].apply(get_sentiment_label)

            # Top positive news
            with st.expander("📈 Top 5 Most Positive News", expanded=True):
                top_positive = news_filtered.nlargest(5, 'sentiment_score')[
                    ['ticker', 'title', 'sentiment_score', 'time_published', 'url']
                ]
                for idx, row in top_positive.iterrows():
                    st.markdown(
                        f"**[{row['ticker']}]** [{row['title']}]({row['url']}) "
                        f"<span style='color: #00CC66;'>+{row['sentiment_score']:.3f}</span>",
                        unsafe_allow_html=True
                    )
                    st.caption(f"📅 {row['time_published']}")
                    st.markdown("---")

            # Top negative news
            with st.expander("📉 Top 5 Most Negative News"):
                top_negative = news_filtered.nsmallest(5, 'sentiment_score')[
                    ['ticker', 'title', 'sentiment_score', 'time_published', 'url']
                ]
                for idx, row in top_negative.iterrows():
                    st.markdown(
                        f"**[{row['ticker']}]** [{row['title']}]({row['url']}) "
                        f"<span style='color: #FF4B4B;'>{row['sentiment_score']:.3f}</span>",
                        unsafe_allow_html=True
                    )
                    st.caption(f"📅 {row['time_published']}")
                    st.markdown("---")

            # All news table
            st.subheader("All News Articles")

            # Format for display
            display_df = news_filtered[['date', 'ticker', 'title', 'sentiment_score', 'sentiment_label', 'source']].copy()
            display_df = display_df.sort_values('date', ascending=False)

            st.dataframe(
                display_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "date": "Date",
                    "ticker": "Ticker",
                    "title": st.column_config.TextColumn("Title", width="large"),
                    "sentiment_score": st.column_config.NumberColumn("Sentiment", format="%.3f"),
                    "sentiment_label": "Label",
                    "source": "Source"
                }
            )
        else:
            st.warning("No news data available for selected filters.")

    # TAB 4: PER-TICKER ANALYSIS
    with tab4:
        st.header("Per-Ticker Analysis")

        if len(selected_tickers) > 0:
            ticker_choice = st.selectbox(
                "Select Ticker",
                options=selected_tickers,
                key='ticker_analysis'
            )

            col1, col2 = st.columns(2)

            # Stock metrics
            if stock_filtered is not None:
                ticker_stock = stock_filtered[stock_filtered['ticker'] == ticker_choice]

                if len(ticker_stock) > 0:
                    with col1:
                        st.subheader("Stock Metrics")
                        latest_price = ticker_stock.iloc[-1]['close']
                        price_change = ticker_stock['daily_return'].iloc[-1] if 'daily_return' in ticker_stock.columns else 0

                        st.metric("Latest Price", f"${latest_price:.2f}", f"{price_change:.2%}")

                        avg_volume = ticker_stock['volume'].mean()
                        st.metric("Avg Volume", f"{avg_volume:,.0f}")

            # Sentiment metrics
            if news_filtered is not None:
                ticker_news = news_filtered[news_filtered['ticker'] == ticker_choice]

                if len(ticker_news) > 0:
                    with col2:
                        st.subheader("Sentiment Metrics")
                        avg_sent = ticker_news['sentiment_score'].mean()
                        std_sent = ticker_news['sentiment_score'].std()

                        st.metric("Avg Sentiment", f"{avg_sent:.3f}")
                        st.metric("Std Deviation", f"{std_sent:.3f}")
                        st.metric("Article Count", len(ticker_news))

            # Charts
            if stock_filtered is not None and len(stock_filtered[stock_filtered['ticker'] == ticker_choice]) > 0:
                st.plotly_chart(
                    create_candlestick_chart(stock_filtered, ticker_choice),
                    width="stretch"
                )

                st.plotly_chart(
                    create_volume_chart(stock_filtered, ticker_choice),
                    width="stretch"
                )

            # News for this ticker
            if news_filtered is not None:
                ticker_news = news_filtered[news_filtered['ticker'] == ticker_choice]

                if len(ticker_news) > 0:
                    st.subheader(f"News for {ticker_choice}")

                    news_display = ticker_news[['date', 'title', 'sentiment_score', 'source']].sort_values('date', ascending=False)
                    st.dataframe(news_display, width="stretch", hide_index=True)
        else:
            st.warning("Please select at least one ticker from the sidebar.")

    # TAB 5: DATA EXPLORER
    with tab5:
        st.header("Data Explorer")

        # Stock data
        if stock_filtered is not None and len(stock_filtered) > 0:
            with st.expander("📊 Stock Prices Data", expanded=False):
                st.dataframe(stock_filtered, width="stretch", hide_index=True)

                csv = stock_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Stock Data CSV",
                    data=csv,
                    file_name=f"stock_prices_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

        # News data
        if news_filtered is not None and len(news_filtered) > 0:
            with st.expander("📰 News Articles Data", expanded=False):
                st.dataframe(news_filtered, width="stretch", hide_index=True)

                csv = news_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download News Data CSV",
                    data=csv,
                    file_name=f"news_articles_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

        # Reddit data
        if data['reddit'] is not None:
            reddit_filtered = data['reddit'][
                (data['reddit']['ticker'].isin(selected_tickers)) &
                (data['reddit']['date'] >= start_date) &
                (data['reddit']['date'] <= end_date)
            ].copy()

            if len(reddit_filtered) > 0:
                with st.expander("💬 Reddit Posts Data", expanded=False):
                    st.dataframe(reddit_filtered, width="stretch", hide_index=True)

                    csv = reddit_filtered.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Reddit Data CSV",
                        data=csv,
                        file_name=f"reddit_posts_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

        # Combined data
        if combined_filtered is not None and len(combined_filtered) > 0:
            with st.expander("🔗 Combined Stock + Sentiment Data", expanded=False):
                st.dataframe(combined_filtered, width="stretch", hide_index=True)

                csv = combined_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Combined Data CSV",
                    data=csv,
                    file_name=f"stock_sentiment_combined_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )


if __name__ == "__main__":
    main()
