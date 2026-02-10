"""
Streamlit Dashboard for Stock Market Sentiment Analysis
Modern, Professional Data Science / ML Theme
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import recommendation tracker (v2.0)
try:
    from src.analysis.recommendation_tracker import RecommendationTracker
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False

# Import backtester
try:
    from src.backtesting.backtester import Backtester, get_benchmark_data
    BACKTESTER_AVAILABLE = True
except ImportError:
    BACKTESTER_AVAILABLE = False

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* === ANIMATED BACKGROUND GRADIENT === */
    .main-header {
        background: linear-gradient(135deg, #0c1929 0%, #1a365d 50%, #0f172a 100%);
        padding: 2.5rem 3rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(59, 130, 246, 0.15);
        position: relative;
        overflow: hidden;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(ellipse at top right, rgba(59, 130, 246, 0.15) 0%, transparent 50%),
                    radial-gradient(ellipse at bottom left, rgba(139, 92, 246, 0.1) 0%, transparent 50%);
        pointer-events: none;
    }

    .main-header h1 {
        color: #f8fafc;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff 0%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        position: relative;
        z-index: 1;
    }

    .main-header p {
        color: #64748b;
        font-size: 1.05rem;
        margin: 0.75rem 0 0 0;
        position: relative;
        z-index: 1;
    }

    .main-header .subtitle-highlight {
        color: #60a5fa;
        font-weight: 600;
    }

    /* === GLASSMORPHISM METRIC CARDS === */
    .metric-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 1.75rem;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.08);
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        border-color: rgba(59, 130, 246, 0.2);
    }

    .metric-card:hover::before {
        opacity: 1;
    }

    .metric-card .metric-value {
        font-size: 2.25rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 0;
        line-height: 1.2;
    }

    .metric-card .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }

    .metric-card.positive .metric-value {
        color: #4ade80;
        text-shadow: 0 0 30px rgba(74, 222, 128, 0.3);
    }
    .metric-card.positive::before {
        background: linear-gradient(90deg, #22c55e, #4ade80);
    }

    .metric-card.negative .metric-value {
        color: #f87171;
        text-shadow: 0 0 30px rgba(248, 113, 113, 0.3);
    }
    .metric-card.negative::before {
        background: linear-gradient(90deg, #ef4444, #f87171);
    }

    .metric-card.accent .metric-value {
        color: #60a5fa;
        text-shadow: 0 0 30px rgba(96, 165, 250, 0.3);
    }
    .metric-card.accent::before {
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
    }

    /* === SECTION STYLING === */
    .section-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(10px);
        padding: 1.75rem;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.08);
        margin-bottom: 1.25rem;
    }

    .section-title {
        color: #f1f5f9;
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        letter-spacing: -0.01em;
    }

    .section-title::before {
        content: '';
        width: 4px;
        height: 24px;
        background: linear-gradient(180deg, #3b82f6, #8b5cf6);
        border-radius: 2px;
    }

    /* === PREMIUM RECOMMENDATION CARDS === */
    .rec-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        backdrop-filter: blur(8px);
        padding: 1.25rem 1.5rem;
        border-radius: 14px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 0.875rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: all 0.25s ease;
        border: 1px solid rgba(148, 163, 184, 0.05);
        border-left-width: 4px;
    }

    .rec-card:hover {
        transform: translateX(4px);
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }

    .rec-card.buy {
        border-left-color: #4ade80;
        box-shadow: -4px 0 20px rgba(74, 222, 128, 0.1);
    }
    .rec-card.sell {
        border-left-color: #f87171;
        box-shadow: -4px 0 20px rgba(248, 113, 113, 0.1);
    }
    .rec-card.hold {
        border-left-color: #fbbf24;
        box-shadow: -4px 0 20px rgba(251, 191, 36, 0.1);
    }

    .rec-ticker {
        font-size: 1.2rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.01em;
    }

    .rec-details {
        display: flex;
        gap: 2rem;
        color: #94a3b8;
        font-size: 0.9rem;
    }

    /* === POLISHED BADGES === */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .badge-buy {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.25) 0%, rgba(74, 222, 128, 0.15) 100%);
        color: #4ade80;
        border: 1px solid rgba(74, 222, 128, 0.2);
    }
    .badge-sell {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.25) 0%, rgba(248, 113, 113, 0.15) 100%);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.2);
    }
    .badge-hold {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.25) 0%, rgba(251, 191, 36, 0.15) 100%);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.2);
    }
    .badge-positive {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
    }
    .badge-negative {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
    }
    .badge-neutral {
        background: rgba(148, 163, 184, 0.15);
        color: #94a3b8;
    }

    /* === SIDEBAR STYLING === */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1929 0%, #0f172a 100%);
    }

    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #e2e8f0;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 1.25rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.1);
    }

    /* === PREMIUM TAB STYLES === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.5) 100%);
        padding: 0.5rem;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.08);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: #64748b;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0.85rem 1.5rem;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #94a3b8;
        background: rgba(148, 163, 184, 0.05);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%);
        color: #f8fafc;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    /* === DATAFRAME STYLES === */
    .stDataFrame {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.1);
    }

    /* === ELEGANT DIVIDER === */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.3), rgba(139, 92, 246, 0.3), transparent);
        margin: 2rem 0;
    }

    /* === PROGRESS BAR === */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #22c55e);
        border-radius: 10px;
    }

    /* === INFO BOX === */
    .info-box {
        background: linear-gradient(145deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.05) 100%);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        color: #93c5fd;
    }

    /* === BUTTONS === */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
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
    'primary': '#60A5FA',      # Bright Blue
    'secondary': '#A78BFA',    # Bright Purple
    'positive': '#4ADE80',     # Bright Green
    'negative': '#F87171',     # Bright Red
    'neutral': '#94A3B8',      # Gray
    'accent': '#FBBF24',       # Bright Amber
    'cyan': '#22D3D1',         # Cyan
    'pink': '#F472B6',         # Pink
    'background': '#0F172A',   # Dark
    'surface': '#1E293B',      # Dark surface
    'text': '#F8FAFC',         # Light text
    'text_muted': '#94A3B8',   # Muted text
}

# Plotly chart template
# Base chart template - excludes title/legend to avoid conflicts with explicit settings
CHART_TEMPLATE = {
    'layout': {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': CHART_COLORS['text'], 'family': 'Inter, sans-serif'},
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
        title=dict(text=f"{selected_ticker} Stock Price", font=dict(size=18, color=CHART_COLORS['text'])),
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
        title=dict(text=f"{selected_ticker} - OHLC Chart", font=dict(size=18, color=CHART_COLORS['text'])),
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
        title=dict(text=f"{selected_ticker} Trading Volume", font=dict(size=16, color=CHART_COLORS['text'])),
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
        title=dict(text="Sentiment Timeline", font=dict(size=18, color=CHART_COLORS['text'])),
        xaxis_title="",
        yaxis_title="Avg Sentiment",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                   bgcolor='rgba(0,0,0,0)', font=dict(color=CHART_COLORS['text_muted'])),
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
        title=dict(text=f'{selected_ticker} - Sentiment vs Daily Return', font=dict(size=18, color=CHART_COLORS['text'])),
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
        title=dict(text="Sentiment Distribution", font=dict(size=18, color=CHART_COLORS['text'])),
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5,
                   font=dict(color=CHART_COLORS['text_muted'])),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=CHART_COLORS['text'], family='Inter, sans-serif'),
        margin=dict(t=60, b=60, l=20, r=20)
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
    """Render the main header with live stats"""
    current_date = datetime.now().strftime("%B %d, %Y")
    st.markdown(f"""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
            <div>
                <h1>Stock Sentiment Analysis</h1>
                <p>AI-powered market insights using <span class="subtitle-highlight">FinBERT NLP</span> + <span class="subtitle-highlight">Machine Learning</span></p>
            </div>
            <div style="text-align: right; color: #64748b; font-size: 0.85rem;">
                <div style="color: #94a3b8; font-weight: 600;">{current_date}</div>
                <div style="margin-top: 0.25rem;">S&P 100 Coverage</div>
            </div>
        </div>
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Overview", "Recommendations", "Performance", "Backtesting", "Price & Sentiment", "News Analysis", "Ticker Deep Dive", "Data Export"
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
                    title=dict(text="Sentiment by Ticker", font=dict(size=18, color=CHART_COLORS['text'])),
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
    # TAB 3: PERFORMANCE TRACKING
    # -------------------------------------------------------------------------
    with tab3:
        st.markdown('<p class="section-title">Investment Performance Simulator</p>', unsafe_allow_html=True)

        if TRACKER_AVAILABLE and data['recommendations'] is not None:
            # Initialize tracker
            tracker = RecommendationTracker()

            # Load recommendation history
            history_file = PROJECT_ROOT / "data" / "recommendations" / "history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history_data = json.load(f)

                # Handle both list format and dict format
                if isinstance(history_data, list):
                    rec_history = history_data
                else:
                    rec_history = history_data.get("recommendations", [])

                if rec_history:
                    # Summary stats
                    stats = tracker.get_summary_stats()

                    cols = st.columns(4)
                    with cols[0]:
                        render_metric_card("Total Sessions", stats.get('total_sessions', 0), "accent")
                    with cols[1]:
                        render_metric_card("Total Signals", stats.get('total_recommendations', 0), "default")
                    with cols[2]:
                        dist = stats.get('signal_distribution', {})
                        render_metric_card("BUY Signals", dist.get('BUY', 0), "positive")
                    with cols[3]:
                        render_metric_card("SELL Signals", dist.get('SELL', 0), "negative")

                    render_divider()

                    # Simulated Portfolio Performance
                    st.markdown('<p class="section-title">Simulated Portfolio Returns</p>', unsafe_allow_html=True)

                    col1, col2 = st.columns([1, 3])

                    with col1:
                        strategy = st.selectbox(
                            "Strategy",
                            options=["Follow BUY signals", "Follow all positive signals"],
                            key="perf_strategy"
                        )
                        holding_days = st.slider("Holding Period (days)", 1, 30, 5, key="perf_holding")
                        initial_capital = st.number_input("Initial Capital ($)", 10000, 1000000, 100000, step=10000)

                    with col2:
                        if stock_filtered is not None and len(stock_filtered) > 0:
                            # Calculate performance
                            strategy_type = "buy_signals" if "BUY" in strategy else "all_signals"

                            try:
                                perf = tracker.calculate_portfolio_performance(
                                    stock_filtered,
                                    strategy=strategy_type,
                                    holding_days=holding_days,
                                    position_size=0.1
                                )

                                if perf.get('num_trades', 0) > 0:
                                    # Performance metrics
                                    perf_cols = st.columns(4)
                                    with perf_cols[0]:
                                        total_return = perf.get('total_return_pct', 0)
                                        card_type = "positive" if total_return > 0 else "negative"
                                        render_metric_card("Total Return", f"{total_return:+.2f}%", card_type)
                                    with perf_cols[1]:
                                        render_metric_card("Win Rate", f"{perf.get('win_rate_pct', 0):.1f}%", "accent")
                                    with perf_cols[2]:
                                        render_metric_card("Trades", perf.get('num_trades', 0), "default")
                                    with perf_cols[3]:
                                        sharpe = perf.get('sharpe_ratio', 0)
                                        render_metric_card("Sharpe Ratio", f"{sharpe:.2f}", "accent")

                                    # Portfolio value chart
                                    if perf.get('portfolio_history'):
                                        portfolio_df = pd.DataFrame(perf['portfolio_history'])
                                        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])

                                        fig = go.Figure()
                                        fig.add_trace(go.Scatter(
                                            x=portfolio_df['date'],
                                            y=portfolio_df['value'],
                                            mode='lines',
                                            name='Portfolio Value',
                                            line=dict(color=CHART_COLORS['primary'], width=2.5),
                                            fill='tozeroy',
                                            fillcolor='rgba(59, 130, 246, 0.1)'
                                        ))
                                        fig.add_hline(y=initial_capital, line_dash="dash",
                                                    line_color="rgba(148, 163, 184, 0.5)")
                                        fig.update_layout(
                                            title=dict(text="Portfolio Value Over Time",
                                                      font=dict(size=18, color=CHART_COLORS['text'])),
                                            xaxis_title="",
                                            yaxis_title="Value ($)",
                                            height=350,
                                            **CHART_TEMPLATE['layout']
                                        )
                                        st.plotly_chart(fig, use_container_width=True)

                                    # Trade details
                                    if perf.get('trades'):
                                        with st.expander("View Trade Details"):
                                            trades_df = pd.DataFrame(perf['trades'])
                                            trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date']).dt.date
                                            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date']).dt.date
                                            st.dataframe(
                                                trades_df[['ticker', 'entry_date', 'exit_date', 'entry_price',
                                                          'exit_price', 'pnl_pct', 'confidence']],
                                                use_container_width=True,
                                                hide_index=True
                                            )
                                else:
                                    st.info("No trades executed with current settings. Try adjusting parameters.")

                            except Exception as e:
                                st.warning(f"Could not calculate performance: {str(e)}")
                        else:
                            st.info("No price data available for performance calculation.")

                    render_divider()

                    # Recommendation History Timeline
                    st.markdown('<p class="section-title">Recommendation History</p>', unsafe_allow_html=True)

                    # Create timeline data
                    timeline_data = []
                    for entry in rec_history[-30:]:  # Last 30 entries
                        summary = entry.get('summary', {})
                        timeline_data.append({
                            'date': entry.get('date'),
                            'BUY': summary.get('buy', 0),
                            'HOLD': summary.get('hold', 0),
                            'SELL': summary.get('sell', 0),
                            'total': summary.get('total', 0)
                        })

                    if timeline_data:
                        timeline_df = pd.DataFrame(timeline_data)
                        timeline_df['date'] = pd.to_datetime(timeline_df['date'])

                        fig = go.Figure()
                        fig.add_trace(go.Bar(name='BUY', x=timeline_df['date'], y=timeline_df['BUY'],
                                            marker_color=CHART_COLORS['positive']))
                        fig.add_trace(go.Bar(name='HOLD', x=timeline_df['date'], y=timeline_df['HOLD'],
                                            marker_color=CHART_COLORS['accent']))
                        fig.add_trace(go.Bar(name='SELL', x=timeline_df['date'], y=timeline_df['SELL'],
                                            marker_color=CHART_COLORS['negative']))

                        fig.update_layout(
                            barmode='stack',
                            title=dict(text="Signals by Date", font=dict(size=18, color=CHART_COLORS['text'])),
                            xaxis_title="",
                            yaxis_title="Count",
                            height=300,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                       bgcolor='rgba(0,0,0,0)', font=dict(color=CHART_COLORS['text_muted'])),
                            **CHART_TEMPLATE['layout']
                        )
                        st.plotly_chart(fig, use_container_width=True)

                else:
                    st.info("No recommendation history yet. Run analysis multiple times to build history.")
            else:
                st.info("No recommendation history file found. Run `python analyze_with_recommendations.py` to generate data.")
        else:
            st.info("Performance tracking requires recommendation history. Run analysis first.")

    # -------------------------------------------------------------------------
    # TAB 4: BACKTESTING WITH BENCHMARK
    # -------------------------------------------------------------------------
    with tab4:
        st.markdown('<p class="section-title">Backtesting with S&P 500 Benchmark</p>', unsafe_allow_html=True)

        if BACKTESTER_AVAILABLE:
            # Backtesting settings
            col1, col2, col3 = st.columns(3)

            with col1:
                bt_initial_capital = st.number_input("Initial Capital ($)", 10000, 1000000, 100000, step=10000, key="bt_capital")
            with col2:
                bt_position_size = st.slider("Position Size (%)", 5, 50, 10, key="bt_position") / 100
            with col3:
                bt_holding_period = st.slider("Holding Period (days)", 1, 30, 5, key="bt_holding")

            # Load backtest results if available
            backtest_file = PROJECT_ROOT / "data" / "backtest" / "backtest_summary.csv"
            portfolio_file = PROJECT_ROOT / "data" / "backtest" / "portfolio_history.csv"
            benchmark_file = PROJECT_ROOT / "data" / "backtest" / "benchmark_history.csv"

            run_backtest = st.button("Run Backtest", type="primary", key="run_bt")

            if run_backtest and data['all_content'] is not None and stock_filtered is not None:
                with st.spinner("Running backtest with benchmark comparison..."):
                    try:
                        backtester = Backtester(
                            initial_capital=bt_initial_capital,
                            position_size=bt_position_size,
                            sentiment_threshold_buy=0.2,
                            holding_period=bt_holding_period
                        )

                        # Prepare data
                        sentiment_df = data['all_content'][['ticker', 'date', 'sentiment_score']].copy()
                        price_df = stock_filtered[['ticker', 'date', 'close']].copy()

                        results = backtester.run_backtest(
                            sentiment_df,
                            price_df,
                            include_benchmark=True
                        )

                        backtester.save_results(results)
                        st.success("Backtest completed!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Backtest failed: {str(e)}")

            render_divider()

            # Display results
            if backtest_file.exists():
                summary_df = pd.read_csv(backtest_file)

                if len(summary_df) > 0:
                    summary = summary_df.iloc[0].to_dict()

                    # KPI Row - Portfolio Performance
                    cols = st.columns(4)
                    with cols[0]:
                        total_return = summary.get('total_return_pct', 0)
                        card_type = "positive" if total_return > 0 else "negative"
                        render_metric_card("Total Return", f"{total_return:+.2f}%", card_type)
                    with cols[1]:
                        render_metric_card("Sharpe Ratio", f"{summary.get('sharpe_ratio', 0):.2f}", "accent")
                    with cols[2]:
                        render_metric_card("Max Drawdown", f"{summary.get('max_drawdown_pct', 0):.2f}%", "negative")
                    with cols[3]:
                        render_metric_card("Win Rate", f"{summary.get('win_rate_pct', 0):.1f}%", "accent")

                    render_divider()

                    # Benchmark comparison metrics
                    if 'benchmark_alpha' in summary:
                        st.markdown('<p class="section-title">Benchmark Comparison (vs S&P 500)</p>', unsafe_allow_html=True)

                        cols = st.columns(4)
                        with cols[0]:
                            alpha = summary.get('benchmark_alpha', 0)
                            card_type = "positive" if alpha > 0 else "negative"
                            render_metric_card("Alpha", f"{alpha:+.2f}%", card_type)
                        with cols[1]:
                            render_metric_card("Beta", f"{summary.get('benchmark_beta', 0):.2f}", "accent")
                        with cols[2]:
                            ir = summary.get('benchmark_information_ratio', 0)
                            render_metric_card("Info Ratio", f"{ir:.2f}", "accent")
                        with cols[3]:
                            sortino = summary.get('benchmark_sortino_ratio', 0)
                            render_metric_card("Sortino", f"{sortino:.2f}", "accent")

                        # Additional benchmark info
                        cols2 = st.columns(3)
                        with cols2[0]:
                            bench_return = summary.get('benchmark_benchmark_return', 0)
                            render_metric_card("SPY Return", f"{bench_return:+.2f}%", "default")
                        with cols2[1]:
                            tracking_error = summary.get('benchmark_tracking_error', 0)
                            render_metric_card("Tracking Error", f"{tracking_error:.2f}%", "default")
                        with cols2[2]:
                            bench_sharpe = summary.get('benchmark_benchmark_sharpe', 0)
                            render_metric_card("SPY Sharpe", f"{bench_sharpe:.2f}", "default")

                        render_divider()

                    # Portfolio vs Benchmark Chart
                    if portfolio_file.exists():
                        portfolio_df = pd.read_csv(portfolio_file)
                        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])

                        # Calculate cumulative returns for portfolio
                        initial_value = portfolio_df['total_value'].iloc[0]
                        portfolio_df['cumulative_return'] = (portfolio_df['total_value'] / initial_value - 1) * 100

                        fig = go.Figure()

                        # Portfolio line
                        fig.add_trace(go.Scatter(
                            x=portfolio_df['date'],
                            y=portfolio_df['cumulative_return'],
                            mode='lines',
                            name='Strategy',
                            line=dict(color=CHART_COLORS['primary'], width=2.5)
                        ))

                        # Benchmark line
                        if benchmark_file.exists():
                            benchmark_df = pd.read_csv(benchmark_file)
                            benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
                            benchmark_df['cumulative_return'] = benchmark_df['cumulative_return'] * 100

                            fig.add_trace(go.Scatter(
                                x=benchmark_df['date'],
                                y=benchmark_df['cumulative_return'],
                                mode='lines',
                                name='S&P 500 (SPY)',
                                line=dict(color=CHART_COLORS['accent'], width=2, dash='dash')
                            ))

                        fig.add_hline(y=0, line_dash="dot", line_color="rgba(148, 163, 184, 0.5)")

                        fig.update_layout(
                            title=dict(text="Portfolio vs Benchmark Performance", font=dict(size=18, color=CHART_COLORS['text'])),
                            xaxis_title="",
                            yaxis_title="Cumulative Return (%)",
                            height=400,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                       bgcolor='rgba(0,0,0,0)', font=dict(color=CHART_COLORS['text_muted'])),
                            **CHART_TEMPLATE['layout']
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    # Trade statistics
                    st.markdown('<p class="section-title">Trade Statistics</p>', unsafe_allow_html=True)

                    cols = st.columns(4)
                    with cols[0]:
                        render_metric_card("Total Trades", int(summary.get('num_trades', 0)), "default")
                    with cols[1]:
                        render_metric_card("Winning Trades", int(summary.get('winning_trades', 0)), "positive")
                    with cols[2]:
                        render_metric_card("Losing Trades", int(summary.get('losing_trades', 0)), "negative")
                    with cols[3]:
                        profit_factor = summary.get('profit_factor', 0)
                        pf_display = f"{profit_factor:.2f}" if profit_factor < 100 else "∞"
                        render_metric_card("Profit Factor", pf_display, "accent")

                    # Trades table
                    trades_file = PROJECT_ROOT / "data" / "backtest" / "trades.csv"
                    if trades_file.exists():
                        with st.expander("View All Trades"):
                            trades_df = pd.read_csv(trades_file)
                            trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date']).dt.date
                            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date']).dt.date

                            display_cols = ['ticker', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'pnl', 'pnl_pct', 'sentiment']
                            available_cols = [c for c in display_cols if c in trades_df.columns]

                            st.dataframe(
                                trades_df[available_cols].style.format({
                                    'entry_price': '${:.2f}',
                                    'exit_price': '${:.2f}',
                                    'pnl': '${:+.2f}',
                                    'pnl_pct': '{:+.2f}%',
                                    'sentiment': '{:.3f}'
                                }),
                                use_container_width=True,
                                hide_index=True
                            )
            else:
                st.info("No backtest results available. Click 'Run Backtest' to generate results.")

        else:
            st.warning("Backtesting module not available. Ensure src/backtesting/backtester.py is accessible.")

    # -------------------------------------------------------------------------
    # TAB 5: PRICES & SENTIMENT
    # -------------------------------------------------------------------------
    with tab5:
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
    # TAB 6: NEWS ANALYSIS
    # -------------------------------------------------------------------------
    with tab6:
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
    # TAB 7: TICKER DEEP DIVE
    # -------------------------------------------------------------------------
    with tab7:
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
    # TAB 8: DATA EXPORT
    # -------------------------------------------------------------------------
    with tab8:
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

