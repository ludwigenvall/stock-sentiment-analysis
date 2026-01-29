"""
Streamlit Dashboard for Stock Market Sentiment Analysis
Modern, Professional Data Science / ML Theme
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

# =============================================================================
# PAGE CONFIG & THEME
# =============================================================================
st.set_page_config(
    page_title="Stock Sentiment Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS - Modern Data Science Theme
# =============================================================================
# Design choices:
# - Dark theme with blue accent (#3B82F6) for professional ML/DS look
# - Card-based layout with subtle shadows for depth
# - Consistent spacing (1rem = 16px base)
# - Clean typography with good contrast
# - Accent colors: Blue (primary), Green (positive), Red (negative), Gray (neutral)

st.markdown("""
<style>
    /* === GLOBAL STYLES === */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* === HEADER STYLES === */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }

    .main-header h1 {
        color: #f8fafc;
        font-size: 2.25rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }

    /* === METRIC CARDS === */
    .metric-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.1);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }

    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
    }

    .metric-card .metric-label {
        font-size: 0.875rem;
        color: #94a3b8;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-card.positive .metric-value { color: #22c55e; }
    .metric-card.negative .metric-value { color: #ef4444; }
    .metric-card.accent .metric-value { color: #3b82f6; }

    /* === SECTION CARDS === */
    .section-card {
        background: #0f172a;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.1);
        margin-bottom: 1rem;
    }

    .section-title {
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* === RECOMMENDATION CARDS === */
    .rec-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        padding: 1rem 1.25rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .rec-card.buy { border-left-color: #22c55e; }
    .rec-card.sell { border-left-color: #ef4444; }
    .rec-card.hold { border-left-color: #f59e0b; }

    .rec-ticker {
        font-size: 1.125rem;
        font-weight: 700;
        color: #f8fafc;
    }

    .rec-details {
        display: flex;
        gap: 1.5rem;
        color: #94a3b8;
        font-size: 0.875rem;
    }

    /* === BADGE STYLES === */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .badge-buy { background: rgba(34, 197, 94, 0.2); color: #22c55e; }
    .badge-sell { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
    .badge-hold { background: rgba(245, 158, 11, 0.2); color: #f59e0b; }
    .badge-positive { background: rgba(34, 197, 94, 0.15); color: #22c55e; }
    .badge-negative { background: rgba(239, 68, 68, 0.15); color: #ef4444; }
    .badge-neutral { background: rgba(148, 163, 184, 0.15); color: #94a3b8; }

    /* === SIDEBAR STYLES === */
    [data-testid="stSidebar"] {
        background: #0f172a;
    }

    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #f8fafc;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
    }

    /* === TAB STYLES === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #0f172a;
        padding: 0.5rem;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
        padding: 0.75rem 1.25rem;
    }

    .stTabs [aria-selected="true"] {
        background: #1e293b;
        color: #f8fafc;
    }

    /* === DATAFRAME STYLES === */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* === DIVIDER === */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.3), transparent);
        margin: 1.5rem 0;
    }

    /* === PROGRESS BAR === */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6, #22c55e);
        border-radius: 10px;
    }

    /* === INFO/WARNING BOXES === */
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        color: #93c5fd;
    }

    /* === RESPONSIVE === */
    @media (max-width: 768px) {
        .main-header { padding: 1.5rem; }
        .main-header h1 { font-size: 1.75rem; }
        .metric-card .metric-value { font-size: 1.5rem; }
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CHART THEME - Consistent colors across all visualizations
# =============================================================================
# Color palette for charts
CHART_COLORS = {
    'primary': '#3B82F6',      # Blue
    'secondary': '#8B5CF6',    # Purple
    'positive': '#22C55E',     # Green
    'negative': '#EF4444',     # Red
    'neutral': '#64748B',      # Gray
    'accent': '#F59E0B',       # Amber
    'background': '#0F172A',   # Dark
    'surface': '#1E293B',      # Dark surface
    'text': '#F8FAFC',         # Light text
    'text_muted': '#94A3B8',   # Muted text
}

# Plotly chart template
CHART_TEMPLATE = {
    'layout': {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': CHART_COLORS['text'], 'family': 'Inter, sans-serif'},
        'title': {'font': {'size': 18, 'color': CHART_COLORS['text']}},
        'xaxis': {
            'gridcolor': 'rgba(148, 163, 184, 0.1)',
            'linecolor': 'rgba(148, 163, 184, 0.2)',
            'tickfont': {'color': CHART_COLORS['text_muted']}
        },
        'yaxis': {
            'gridcolor': 'rgba(148, 163, 184, 0.1)',
            'linecolor': 'rgba(148, 163, 184, 0.2)',
            'tickfont': {'color': CHART_COLORS['text_muted']}
        },
        'legend': {'bgcolor': 'rgba(0,0,0,0)', 'font': {'color': CHART_COLORS['text_muted']}},
        'margin': {'t': 60, 'b': 40, 'l': 60, 'r': 20}
    }
}

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
def find_project_root():
    """Find project root by looking for data directory"""
    path1 = Path(__file__).parent.parent.parent / "data"
    if path1.exists():
        return path1.parent
    path2 = Path.cwd() / "data"
    if path2.exists():
        return Path.cwd()
    path3 = Path.cwd().parent / "data"
    if path3.exists():
        return Path.cwd().parent
    return Path(__file__).parent.parent.parent

PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'META']

# =============================================================================
# DATA LOADING (unchanged logic)
# =============================================================================
@st.cache_data(ttl=300)
def load_data():
    """Load all available data files"""
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

    # Load combined data
    combined_file = processed_dir / "stock_sentiment_combined.csv"
    if combined_file.exists():
        df = pd.read_csv(combined_file)
        df['date'] = pd.to_datetime(df['date']).dt.date
        data['combined'] = df
    else:
        data['combined'] = None

    # Load combined with Reddit
    reddit_combined_file = processed_dir / "stock_sentiment_reddit_combined.csv"
    if reddit_combined_file.exists():
        df = pd.read_csv(reddit_combined_file)
        df['date'] = pd.to_datetime(df['date']).dt.date
        data['reddit_combined'] = df
    else:
        data['reddit_combined'] = None

    # Load all content with sentiment
    all_content_file = processed_dir / "all_sentiment.csv"
    if all_content_file.exists():
        df = pd.read_csv(all_content_file)
        df['date'] = pd.to_datetime(df['date'])
        data['all_content'] = df
    else:
        all_content_file = processed_dir / "all_content_with_sentiment.csv"
        if all_content_file.exists():
            df = pd.read_csv(all_content_file)
            df['date'] = pd.to_datetime(df['date'])
            data['all_content'] = df
        else:
            data['all_content'] = None

    # Load news with sentiment
    news_sentiment_file = processed_dir / "news_with_sentiment.csv"
    if news_sentiment_file.exists():
        df = pd.read_csv(news_sentiment_file)
        df['time_published'] = pd.to_datetime(df['time_published'])
        df['date'] = df['time_published'].dt.date
        data['news_sentiment'] = df
    else:
        data['news_sentiment'] = None

    # Load recommendations
    recommendations_dir = PROJECT_ROOT / "data" / "recommendations"
    recommendations_file = recommendations_dir / "latest_recommendations.csv"
    if recommendations_file.exists():
        df = pd.read_csv(recommendations_file)
        data['recommendations'] = df
    else:
        data['recommendations'] = None

    return data

# =============================================================================
# HELPER FUNCTIONS (unchanged logic)
# =============================================================================
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
        'positive': CHART_COLORS['positive'],
        'negative': CHART_COLORS['negative'],
        'neutral': CHART_COLORS['neutral']
    }
    return colors.get(label, CHART_COLORS['neutral'])

# =============================================================================
# CHART FUNCTIONS (updated styling, same logic)
# =============================================================================
def create_price_chart(stock_df, selected_ticker):
    """Create stock price line chart with modern styling"""
    df = stock_df[stock_df['ticker'] == selected_ticker].copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['close'],
        mode='lines',
        name='Close Price',
        line=dict(color=CHART_COLORS['primary'], width=2.5),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))

    fig.update_layout(
        title=dict(text=f"{selected_ticker} Stock Price", font=dict(size=18)),
        xaxis_title="",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=400,
        **CHART_TEMPLATE['layout']
    )
    return fig

def create_candlestick_chart(stock_df, selected_ticker):
    """Create candlestick chart with modern styling"""
    df = stock_df[stock_df['ticker'] == selected_ticker].copy()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='OHLC',
        increasing_line_color=CHART_COLORS['positive'],
        decreasing_line_color=CHART_COLORS['negative']
    ))

    fig.update_layout(
        title=dict(text=f"{selected_ticker} - OHLC Chart", font=dict(size=18)),
        xaxis_title="",
        yaxis_title="Price ($)",
        height=450,
        xaxis_rangeslider_visible=False,
        **CHART_TEMPLATE['layout']
    )
    return fig

def create_volume_chart(stock_df, selected_ticker):
    """Create volume bar chart with modern styling"""
    df = stock_df[stock_df['ticker'] == selected_ticker].copy()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['date'],
        y=df['volume'],
        name='Volume',
        marker_color=CHART_COLORS['primary'],
        opacity=0.7
    ))

    fig.update_layout(
        title=dict(text=f"{selected_ticker} Trading Volume", font=dict(size=16)),
        xaxis_title="",
        yaxis_title="Volume",
        height=200,
        **CHART_TEMPLATE['layout']
    )
    return fig

def create_sentiment_timeline(news_df, selected_tickers):
    """Create sentiment timeline with modern styling"""
    df = news_df[news_df['ticker'].isin(selected_tickers)].copy()
    daily_sentiment = df.groupby(['date', 'ticker'])['sentiment_score'].mean().reset_index()

    # Custom color sequence for multiple tickers
    color_sequence = [CHART_COLORS['primary'], CHART_COLORS['secondary'],
                      CHART_COLORS['accent'], CHART_COLORS['positive'], '#EC4899', '#06B6D4']

    fig = px.line(
        daily_sentiment,
        x='date',
        y='sentiment_score',
        color='ticker',
        color_discrete_sequence=color_sequence
    )

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(148, 163, 184, 0.5)", line_width=1)
    fig.update_layout(
        title=dict(text="Sentiment Timeline", font=dict(size=18)),
        xaxis_title="",
        yaxis_title="Avg Sentiment",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **CHART_TEMPLATE['layout']
    )
    return fig

def create_correlation_scatter(combined_df, selected_ticker):
    """Create scatter plot with modern styling"""
    if 'daily_return' not in combined_df.columns and 'close' in combined_df.columns:
        combined_df = combined_df.sort_values(['ticker', 'date'])
        combined_df['daily_return'] = combined_df.groupby('ticker')['close'].pct_change() * 100

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
        trendline="ols",
        color_discrete_sequence=[CHART_COLORS['primary']]
    )

    fig.update_traces(marker=dict(size=10, opacity=0.7))
    fig.update_layout(
        title=dict(text=f'{selected_ticker} - Sentiment vs Daily Return', font=dict(size=18)),
        xaxis_title="Average Sentiment",
        yaxis_title="Daily Return (%)",
        height=400,
        **CHART_TEMPLATE['layout']
    )
    return fig

def create_sentiment_distribution(news_df):
    """Create donut chart for sentiment distribution with modern styling"""
    news_df['sentiment_label'] = news_df['sentiment_score'].apply(get_sentiment_label)
    sentiment_counts = news_df['sentiment_label'].value_counts()

    colors = [CHART_COLORS['positive'], CHART_COLORS['negative'], CHART_COLORS['neutral']]

    fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        hole=0.6,
        marker=dict(colors=[
            CHART_COLORS['positive'] if l == 'positive' else
            CHART_COLORS['negative'] if l == 'negative' else
            CHART_COLORS['neutral']
            for l in sentiment_counts.index
        ]),
        textinfo='label+percent',
        textfont=dict(size=14, color=CHART_COLORS['text']),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>"
    )])

    fig.update_layout(
        title=dict(text="Sentiment Distribution", font=dict(size=18)),
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        **CHART_TEMPLATE['layout']
    )
    return fig

# =============================================================================
# UI COMPONENTS
# =============================================================================
def render_metric_card(label, value, card_type="default"):
    """Render a styled metric card"""
    type_class = f" {card_type}" if card_type != "default" else ""
    st.markdown(f"""
    <div class="metric-card{type_class}">
        <p class="metric-value">{value}</p>
        <p class="metric-label">{label}</p>
    </div>
    """, unsafe_allow_html=True)

def render_recommendation_card(ticker, sentiment, confidence, rec_type):
    """Render a styled recommendation card"""
    badge_class = rec_type.lower()
    st.markdown(f"""
    <div class="rec-card {badge_class}">
        <div>
            <span class="rec-ticker">{ticker}</span>
            <span class="badge badge-{badge_class}" style="margin-left: 0.75rem;">{rec_type}</span>
        </div>
        <div class="rec-details">
            <span>Sentiment: <strong>{sentiment:+.3f}</strong></span>
            <span>Confidence: <strong>{confidence:.1%}</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_header():
    """Render the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>Stock Sentiment Analysis</h1>
        <p>AI-powered market sentiment insights using FinBERT NLP</p>
    </div>
    """, unsafe_allow_html=True)

def render_divider():
    """Render a styled divider"""
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    # Render header
    render_header()

    # Load data
    data = load_data()

    # Check if we have any data
    if data['news'] is None and data['stock'] is None and data['reddit'] is None and data['all_content'] is None:
        st.error("No data found! Please run the analysis script first.")
        with st.expander("Debug Info"):
            st.write(f"**PROJECT_ROOT:** {PROJECT_ROOT}")
            st.write(f"**DATA_DIR exists:** {DATA_DIR.exists()}")
            st.write(f"**PROCESSED_DIR exists:** {PROCESSED_DIR.exists()}")
            if PROCESSED_DIR.exists():
                st.write(f"**Files:** {list(PROCESSED_DIR.iterdir())}")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.markdown("## FILTERS")

        # Ticker selection
        if data['news'] is not None:
            available_tickers = sorted(data['news']['ticker'].unique())
        elif data['stock'] is not None:
            available_tickers = sorted(data['stock']['ticker'].unique())
        elif data['all_content'] is not None:
            available_tickers = sorted(data['all_content']['ticker'].unique())
        else:
            available_tickers = TICKERS

        selected_tickers = st.multiselect(
            "Select Tickers",
            options=available_tickers,
            default=available_tickers[:5] if len(available_tickers) >= 5 else available_tickers
        )

        # Content source filter
        if data['all_content'] is not None and 'content_type' in data['all_content'].columns:
            available_types = data['all_content']['content_type'].unique().tolist()
            type_labels = {
                'news': 'News', 'reddit': 'Reddit', 'sec_filing': 'SEC Filings',
                'earnings_call': 'Earnings Calls', 'earnings': 'Earnings'
            }
            content_sources = [type_labels.get(t, t.title()) for t in available_types]
            content_sources.append('All Sources')
            content_filter = st.multiselect("Content Sources", options=content_sources, default=['All Sources'])
        else:
            content_filter = None

        # Date range
        if data['news'] is not None:
            min_date, max_date = data['news']['date'].min(), data['news']['date'].max()
        elif data['stock'] is not None:
            min_date, max_date = data['stock']['date'].min(), data['stock']['date'].max()
        elif data['all_content'] is not None:
            min_date = pd.to_datetime(data['all_content']['date']).min().date()
            max_date = pd.to_datetime(data['all_content']['date']).max().date()
        else:
            min_date, max_date = datetime.now().date() - timedelta(days=30), datetime.now().date()

        date_range = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

        st.markdown("---")

        # Data sources info
        sources = []
        if data['news'] is not None: sources.append("News")
        if data['reddit'] is not None: sources.append("Reddit")
        if data['all_content'] is not None and 'content_type' in data['all_content'].columns:
            if 'sec_filing' in data['all_content']['content_type'].unique(): sources.append("SEC")
        if data['stock'] is not None: sources.append("Prices")
        sources.append("FinBERT")

        st.markdown(f"**Data Sources:** {' • '.join(sources)}")

        if data['all_content'] is not None:
            last_update = pd.to_datetime(data['all_content']['date']).max()
            st.caption(f"Last updated: {last_update.strftime('%Y-%m-%d')}")

    # Filter data
    start_date, end_date = (date_range[0], date_range[1]) if len(date_range) == 2 else (date_range[0], date_range[0])

    news_filtered = None
    if data['news'] is not None:
        news_filtered = data['news'][
            (data['news']['ticker'].isin(selected_tickers)) &
            (data['news']['date'] >= start_date) &
            (data['news']['date'] <= end_date)
        ].copy()

    stock_filtered = None
    if data['stock'] is not None:
        stock_filtered = data['stock'][
            (data['stock']['ticker'].isin(selected_tickers)) &
            (data['stock']['date'] >= start_date) &
            (data['stock']['date'] <= end_date)
        ].copy()

    combined_filtered = None
    if data['combined'] is not None:
        combined_filtered = data['combined'][
            (data['combined']['ticker'].isin(selected_tickers)) &
            (data['combined']['date'] >= start_date) &
            (data['combined']['date'] <= end_date)
        ].copy()

    # =============================================================================
    # TABS
    # =============================================================================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", "Recommendations", "Price & Sentiment", "News Analysis", "Ticker Deep Dive", "Data Export"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: OVERVIEW
    # -------------------------------------------------------------------------
    with tab1:
        overview_df = None
        if data['all_content'] is not None:
            overview_df = data['all_content'][data['all_content']['ticker'].isin(selected_tickers)].copy()
            if 'date' in overview_df.columns:
                overview_df['date'] = pd.to_datetime(overview_df['date']).dt.date
                overview_df = overview_df[(overview_df['date'] >= start_date) & (overview_df['date'] <= end_date)]
        elif news_filtered is not None:
            overview_df = news_filtered

        if overview_df is not None and len(overview_df) > 0:
            # KPI Row
            cols = st.columns(4)
            with cols[0]:
                render_metric_card("Tickers Analyzed", len(selected_tickers), "accent")
            with cols[1]:
                render_metric_card("Total Content", f"{len(overview_df):,}", "default")
            with cols[2]:
                avg_sentiment = overview_df['sentiment_score'].mean()
                card_type = "positive" if avg_sentiment > 0.1 else "negative" if avg_sentiment < -0.1 else "default"
                render_metric_card("Avg Sentiment", f"{avg_sentiment:+.3f}", card_type)
            with cols[3]:
                if 'sentiment_label' not in overview_df.columns:
                    overview_df['sentiment_label'] = overview_df['sentiment_score'].apply(get_sentiment_label)
                positive_pct = (overview_df['sentiment_label'] == 'positive').sum() / len(overview_df) * 100
                render_metric_card("Positive Rate", f"{positive_pct:.1f}%", "positive" if positive_pct > 50 else "default")

            render_divider()

            # Content breakdown
            if 'content_type' in overview_df.columns:
                st.markdown('<p class="section-title">Content by Source</p>', unsafe_allow_html=True)
                type_counts = overview_df['content_type'].value_counts()
                type_labels = {'news': 'News', 'reddit': 'Reddit', 'sec_filing': 'SEC Filings', 'earnings_call': 'Earnings', 'earnings': 'Earnings'}

                cols = st.columns(len(type_counts))
                for i, (ctype, count) in enumerate(type_counts.items()):
                    with cols[i]:
                        label = type_labels.get(ctype, ctype.title())
                        avg_sent = overview_df[overview_df['content_type'] == ctype]['sentiment_score'].mean()
                        st.metric(label, count, f"{avg_sent:+.2f}")

            # Charts row
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_sentiment_distribution(overview_df), use_container_width=True)
            with col2:
                # Sentiment by ticker bar chart
                ticker_sentiment = overview_df.groupby('ticker')['sentiment_score'].mean().sort_values(ascending=True)

                fig = go.Figure(go.Bar(
                    x=ticker_sentiment.values,
                    y=ticker_sentiment.index,
                    orientation='h',
                    marker=dict(
                        color=[CHART_COLORS['positive'] if v > 0 else CHART_COLORS['negative'] for v in ticker_sentiment.values],
                        opacity=0.8
                    )
                ))
                fig.update_layout(
                    title=dict(text="Sentiment by Ticker", font=dict(size=18)),
                    xaxis_title="Avg Sentiment",
                    yaxis_title="",
                    height=400,
                    **CHART_TEMPLATE['layout']
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select tickers from the sidebar to view analysis.")

    # -------------------------------------------------------------------------
    # TAB 2: RECOMMENDATIONS
    # -------------------------------------------------------------------------
    with tab2:
        if data['recommendations'] is not None:
            rec_df = data['recommendations'].copy()

            # Summary KPIs
            cols = st.columns(4)
            with cols[0]:
                buy_count = len(rec_df[rec_df['recommendation'] == 'BUY'])
                render_metric_card("BUY Signals", buy_count, "positive")
            with cols[1]:
                hold_count = len(rec_df[rec_df['recommendation'] == 'HOLD'])
                render_metric_card("HOLD Signals", hold_count, "default")
            with cols[2]:
                sell_count = len(rec_df[rec_df['recommendation'] == 'SELL'])
                render_metric_card("SELL Signals", sell_count, "negative")
            with cols[3]:
                avg_conf = rec_df['confidence'].mean()
                render_metric_card("Avg Confidence", f"{avg_conf:.1%}", "accent")

            render_divider()

            # Top Picks
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<p class="section-title">Top Buy Recommendations</p>', unsafe_allow_html=True)
                top_buys = rec_df[rec_df['recommendation'] == 'BUY'].sort_values('confidence', ascending=False).head(8)
                if len(top_buys) > 0:
                    for _, row in top_buys.iterrows():
                        render_recommendation_card(row['ticker'], row['sentiment_score'], row['confidence'], 'BUY')
                else:
                    st.info("No BUY recommendations available.")

            with col2:
                st.markdown('<p class="section-title">Stocks to Watch</p>', unsafe_allow_html=True)
                sells = rec_df[rec_df['recommendation'] == 'SELL'].sort_values('sentiment_score', ascending=True).head(5)
                holds = rec_df[rec_df['recommendation'] == 'HOLD'].sort_values('confidence', ascending=False).head(3)

                if len(sells) > 0:
                    for _, row in sells.iterrows():
                        render_recommendation_card(row['ticker'], row['sentiment_score'], row['confidence'], 'SELL')
                if len(holds) > 0:
                    for _, row in holds.iterrows():
                        render_recommendation_card(row['ticker'], row['sentiment_score'], row['confidence'], 'HOLD')
                if len(sells) == 0 and len(holds) == 0:
                    st.success("All stocks show positive signals!")

            render_divider()

            # Full table
            st.markdown('<p class="section-title">All Recommendations</p>', unsafe_allow_html=True)
            display_df = rec_df[['ticker', 'recommendation', 'sentiment_score', 'confidence', 'num_articles']].copy()
            display_df.columns = ['Ticker', 'Signal', 'Sentiment', 'Confidence', 'Articles']

            st.dataframe(
                display_df.style.format({'Sentiment': '{:+.3f}', 'Confidence': '{:.1%}'}),
                use_container_width=True,
                hide_index=True,
                height=400
            )
        else:
            st.info("No recommendations available. Run the analysis script first.")

    # -------------------------------------------------------------------------
    # TAB 3: PRICES & SENTIMENT
    # -------------------------------------------------------------------------
    with tab3:
        if stock_filtered is not None and len(stock_filtered) > 0:
            chart_ticker = st.selectbox("Select Ticker", options=selected_tickers, key='price_ticker')

            st.plotly_chart(create_price_chart(stock_filtered, chart_ticker), use_container_width=True)

            if news_filtered is not None and len(news_filtered) > 0:
                st.plotly_chart(create_sentiment_timeline(news_filtered, selected_tickers), use_container_width=True)

            if combined_filtered is not None and len(combined_filtered) > 0:
                render_divider()
                st.markdown('<p class="section-title">Correlation Analysis</p>', unsafe_allow_html=True)

                col1, col2 = st.columns([3, 1])
                with col1:
                    scatter_fig = create_correlation_scatter(combined_filtered, chart_ticker)
                    if scatter_fig:
                        st.plotly_chart(scatter_fig, use_container_width=True)
                    else:
                        st.info("Not enough data for correlation analysis.")
                with col2:
                    combined_temp = combined_filtered.copy()
                    if 'daily_return' not in combined_temp.columns and 'close' in combined_temp.columns:
                        combined_temp = combined_temp.sort_values(['ticker', 'date'])
                        combined_temp['daily_return'] = combined_temp.groupby('ticker')['close'].pct_change() * 100

                    if 'daily_return' in combined_temp.columns and 'avg_sentiment' in combined_temp.columns:
                        ticker_data = combined_temp[combined_temp['ticker'] == chart_ticker].dropna(subset=['avg_sentiment', 'daily_return'])
                        if len(ticker_data) > 1:
                            correlation = ticker_data['avg_sentiment'].corr(ticker_data['daily_return'])
                            render_metric_card("Correlation", f"{correlation:.3f}", "accent")

                            if abs(correlation) > 0.3:
                                st.success("Moderate correlation detected")
                            else:
                                st.info("Weak correlation")
        else:
            st.info("No stock price data available for selected tickers.")

    # -------------------------------------------------------------------------
    # TAB 4: NEWS ANALYSIS
    # -------------------------------------------------------------------------
    with tab4:
        if news_filtered is not None and len(news_filtered) > 0:
            news_filtered['sentiment_label'] = news_filtered['sentiment_score'].apply(get_sentiment_label)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<p class="section-title">Most Positive News</p>', unsafe_allow_html=True)
                top_positive = news_filtered.nlargest(5, 'sentiment_score')
                for _, row in top_positive.iterrows():
                    st.markdown(f"""
                    <div class="rec-card buy">
                        <div>
                            <span class="badge badge-positive">{row['ticker']}</span>
                            <span style="color: #f8fafc; margin-left: 0.5rem;">{row['title'][:60]}...</span>
                        </div>
                        <span style="color: #22c55e; font-weight: 600;">+{row['sentiment_score']:.3f}</span>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.markdown('<p class="section-title">Most Negative News</p>', unsafe_allow_html=True)
                top_negative = news_filtered.nsmallest(5, 'sentiment_score')
                for _, row in top_negative.iterrows():
                    st.markdown(f"""
                    <div class="rec-card sell">
                        <div>
                            <span class="badge badge-negative">{row['ticker']}</span>
                            <span style="color: #f8fafc; margin-left: 0.5rem;">{row['title'][:60]}...</span>
                        </div>
                        <span style="color: #ef4444; font-weight: 600;">{row['sentiment_score']:.3f}</span>
                    </div>
                    """, unsafe_allow_html=True)

            render_divider()

            st.markdown('<p class="section-title">All News Articles</p>', unsafe_allow_html=True)
            display_df = news_filtered[['date', 'ticker', 'title', 'sentiment_score', 'sentiment_label', 'source']].copy()
            display_df = display_df.sort_values('date', ascending=False)

            st.dataframe(
                display_df,
                use_container_width=True,
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
            st.info("No news data available for selected tickers.")

    # -------------------------------------------------------------------------
    # TAB 5: TICKER DEEP DIVE
    # -------------------------------------------------------------------------
    with tab5:
        if len(selected_tickers) > 0:
            ticker_choice = st.selectbox("Select Ticker for Analysis", options=selected_tickers, key='ticker_analysis')

            # Metrics row
            col1, col2, col3, col4 = st.columns(4)

            if stock_filtered is not None:
                ticker_stock = stock_filtered[stock_filtered['ticker'] == ticker_choice]
                if len(ticker_stock) > 0:
                    with col1:
                        latest_price = ticker_stock.iloc[-1]['close']
                        render_metric_card("Latest Price", f"${latest_price:.2f}", "accent")
                    with col2:
                        avg_volume = ticker_stock['volume'].mean()
                        render_metric_card("Avg Volume", f"{avg_volume:,.0f}", "default")

            if news_filtered is not None:
                ticker_news = news_filtered[news_filtered['ticker'] == ticker_choice]
                if len(ticker_news) > 0:
                    with col3:
                        avg_sent = ticker_news['sentiment_score'].mean()
                        card_type = "positive" if avg_sent > 0.1 else "negative" if avg_sent < -0.1 else "default"
                        render_metric_card("Avg Sentiment", f"{avg_sent:+.3f}", card_type)
                    with col4:
                        render_metric_card("Article Count", len(ticker_news), "default")

            render_divider()

            # Charts
            if stock_filtered is not None and len(stock_filtered[stock_filtered['ticker'] == ticker_choice]) > 0:
                st.plotly_chart(create_candlestick_chart(stock_filtered, ticker_choice), use_container_width=True)
                st.plotly_chart(create_volume_chart(stock_filtered, ticker_choice), use_container_width=True)

            # News table
            if news_filtered is not None:
                ticker_news = news_filtered[news_filtered['ticker'] == ticker_choice]
                if len(ticker_news) > 0:
                    st.markdown(f'<p class="section-title">Recent News for {ticker_choice}</p>', unsafe_allow_html=True)
                    news_display = ticker_news[['date', 'title', 'sentiment_score', 'source']].sort_values('date', ascending=False)
                    st.dataframe(news_display, use_container_width=True, hide_index=True)
        else:
            st.info("Select at least one ticker from the sidebar.")

    # -------------------------------------------------------------------------
    # TAB 6: DATA EXPORT
    # -------------------------------------------------------------------------
    with tab6:
        st.markdown('<p class="section-title">Export Data</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            if stock_filtered is not None and len(stock_filtered) > 0:
                csv = stock_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Stock Prices",
                    data=csv,
                    file_name=f"stock_prices_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            if data['recommendations'] is not None:
                csv = data['recommendations'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Recommendations",
                    data=csv,
                    file_name=f"recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with col2:
            if news_filtered is not None and len(news_filtered) > 0:
                csv = news_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download News Data",
                    data=csv,
                    file_name=f"news_articles_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            if combined_filtered is not None and len(combined_filtered) > 0:
                csv = combined_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Combined Data",
                    data=csv,
                    file_name=f"combined_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        render_divider()

        # Data preview
        st.markdown('<p class="section-title">Data Preview</p>', unsafe_allow_html=True)

        preview_option = st.selectbox("Select data to preview",
            ["Stock Prices", "News Articles", "Recommendations", "All Sentiment"])

        if preview_option == "Stock Prices" and stock_filtered is not None:
            st.dataframe(stock_filtered.head(100), use_container_width=True, hide_index=True)
        elif preview_option == "News Articles" and news_filtered is not None:
            st.dataframe(news_filtered.head(100), use_container_width=True, hide_index=True)
        elif preview_option == "Recommendations" and data['recommendations'] is not None:
            st.dataframe(data['recommendations'], use_container_width=True, hide_index=True)
        elif preview_option == "All Sentiment" and data['all_content'] is not None:
            st.dataframe(data['all_content'].head(100), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
