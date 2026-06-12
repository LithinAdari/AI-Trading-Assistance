import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as obj
from plotly.subplots import make_subplots
import json
import os
import sys
import subprocess
import yfinance as yf
from datetime import datetime
import streamlit.components.v1 as components
import config
from config import TICKERS, SECTORS, PORTFOLIO_FILE, RECOMMENDATIONS_FILE, BROKER_CONFIGS, ALL_TICKERS, BASE_DIR
from broker_charges import calculate_portfolio_metrics, calculate_charges
from sentiment_analyzer import score_text_sentiment
from breakout_scanner import run_breakout_scan, save_scan_results, load_scan_results

def add_ticker_to_watchlist(t_formatted):
    """Adds a formatted ticker symbol (e.g. INFY.NS) to custom_tickers.json and auto-detects its industry sector."""
    CUSTOM_TICKERS_FILE = os.path.join(BASE_DIR, "custom_tickers.json")
    custom_data = {"tickers": [], "sectors": {}}
    if os.path.exists(CUSTOM_TICKERS_FILE):
        try:
            with open(CUSTOM_TICKERS_FILE, "r") as f:
                custom_data = json.load(f)
        except Exception:
            pass
            
    tickers_list = custom_data.setdefault("tickers", [])
    sectors_dict = custom_data.setdefault("sectors", {})
    
    if t_formatted in tickers_list:
        return True, "Already in watchlist."
        
    # Load Nifty 500 sectors config
    nifty_sectors = {}
    NIFTY500_CACHE_FILE = os.path.join(BASE_DIR, "nifty500_config.json")
    if os.path.exists(NIFTY500_CACHE_FILE):
        try:
            with open(NIFTY500_CACHE_FILE, "r") as f:
                nifty_data = json.load(f)
            nifty_sectors = nifty_data.get("sectors", {})
        except Exception:
            pass
            
    industry = "Custom Tickers"
    found_sector = False
    for s_name, s_tickers in nifty_sectors.items():
        if t_formatted in s_tickers:
            industry = s_name
            found_sector = True
            break
            
    if not found_sector:
        try:
            ticker_info = yf.Ticker(t_formatted).info
            fetched_ind = ticker_info.get("industry")
            if fetched_ind:
                industry = fetched_ind
        except Exception:
            industry = "Custom Tickers"
            
    tickers_list.append(t_formatted)
    sectors_dict.setdefault(industry, []).append(t_formatted)
    
    custom_data["tickers"] = sorted(list(set(tickers_list)))
    for s_name in sectors_dict:
        sectors_dict[s_name] = sorted(list(set(sectors_dict[s_name])))
    custom_data["sectors"] = sectors_dict
    
    try:
        with open(CUSTOM_TICKERS_FILE, "w") as f:
            json.dump(custom_data, f, indent=4)
        config.refresh_config()
        return True, f"Successfully added {t_formatted} to sector '{industry}'."
    except Exception as e:
        return False, f"Failed to save to database: {e}"

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Technical Trading Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark CSS Styling
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        /* Main Styling */
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        .stApp {
            background: radial-gradient(circle at 10% 20%, rgb(15, 23, 42) 0%, rgb(9, 13, 26) 90%);
            color: #E2E8F0;
        }
        
        /* Glassmorphic Metrics Card */
        .metric-card {
            background: rgba(30, 41, 59, 0.45);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            border-color: rgba(16, 185, 129, 0.3);
            transform: translateY(-2px);
        }
        .metric-title {
            color: #94A3B8;
            font-size: 0.9rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }
        .metric-val {
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1;
            margin-bottom: 6px;
        }
        .metric-sub {
            font-size: 0.85rem;
            color: #10B981;
            font-weight: 600;
        }
        .metric-sub.negative {
            color: #EF4444;
        }
        
        /* Alarm Badges */
        .badge-safe {
            background-color: rgba(16, 185, 129, 0.15);
            color: #10B981;
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 4px;
            padding: 3px 8px;
            font-weight: 600;
            font-size: 0.8rem;
            display: inline-block;
        }
        .badge-warning {
            background-color: rgba(245, 158, 11, 0.15);
            color: #F59E0B;
            border: 1px solid rgba(245, 158, 11, 0.3);
            border-radius: 4px;
            padding: 3px 8px;
            font-weight: 600;
            font-size: 0.8rem;
            display: inline-block;
        }
        .badge-danger {
            background-color: rgba(239, 68, 68, 0.15);
            color: #EF4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 4px;
            padding: 3px 8px;
            font-weight: 600;
            font-size: 0.8rem;
            display: inline-block;
        }
        
        /* Headers and subheaders */
        h1, h2, h3 {
            font-weight: 800 !important;
            letter-spacing: -0.02em;
        }
        .section-header {
            border-bottom: 2px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 8px;
            margin-bottom: 20px;
        }
        
        /* Custom tables */
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 8px;
            overflow: hidden;
        }
        
        /* 📱 Responsive UI for Mobile & Tablets */
        @media (max-width: 768px) {
            .metric-val { font-size: 1.6rem !important; }
            .metric-card { padding: 12px !important; }
            h1 { font-size: 1.8rem !important; }
            h2 { font-size: 1.5rem !important; }
            h3 { font-size: 1.2rem !important; }
            .section-header { margin-bottom: 12px !important; padding-bottom: 4px !important; }
            
            /* Make dataframe fonts readable on mobile */
            div[data-testid="stDataFrame"] { font-size: 0.85rem !important; }
            
            /* Streamlit specific layout overrides for mobile */
            .css-1544g2n { padding: 1rem 1rem 1.5rem !important; }
            .block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }
        }
        
        /* 💻 Responsive UI for Small Laptops */
        @media (min-width: 769px) and (max-width: 1024px) {
            .metric-val { font-size: 1.9rem !important; }
            .metric-card { padding: 16px !important; }
        }
    </style>
""", unsafe_allow_html=True)

# Helper functions for persisting portfolio
def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_portfolio(data):
    try:
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        st.error(f"Error saving holdings: {e}")

def load_recommendations():
    if os.path.exists(RECOMMENDATIONS_FILE):
        try:
            with open(RECOMMENDATIONS_FILE, "r") as f:
                return {item["ticker"]: item for item in json.load(f)}
        except Exception:
            return {}
    return {}

@st.cache_data(ttl=600)
def fetch_live_sentiment(ticker):
    """Fetch live sentiment for a ticker from Yahoo Finance news."""
    try:
        from sentiment_analyzer import get_ticker_news_sentiment
        res = get_ticker_news_sentiment(ticker)
        return float(res.get("average_sentiment", 0.0))
    except Exception:
        return 0.0

@st.cache_data(ttl=1800) # cache trending list for 30 minutes
def fetch_trending_tickers():
    """Scrapes trending/most active, gainer, and loser NSE tickers from Google Finance."""
    import requests
    import re
    urls = {
        "Most Active": "https://www.google.com/finance/markets/most-actives?hl=en&gl=IN",
        "Top Gainers": "https://www.google.com/finance/markets/gainers?hl=en&gl=IN",
        "Top Losers": "https://www.google.com/finance/markets/losers?hl=en&gl=IN"
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36"
    }
    trending = set()
    for name, url in urls.items():
        try:
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 200:
                quotes = re.findall(r'/quote/([A-Z0-9_]+):NSE', res.text)
                for q in quotes:
                    trending.add(f"{q}.NS")
        except Exception:
            pass
    return sorted(list(trending))
def group_tickers_by_sector(tickers):
    """Groups a list of tickers into their respective market sectors dynamically."""
    groups = {s: [] for s in SECTORS.keys()}
    groups["Custom Tickers"] = []
    
    for t in tickers:
        placed = False
        for sector_name, sector_list in SECTORS.items():
            if t in sector_list:
                groups[sector_name].append(t)
                placed = True
                break
        if not placed:
            groups["Custom Tickers"].append(t)
            
    # Return only groups that actually have tickers
    return {k: v for k, v in groups.items() if len(v) > 0}
# ----------------- SIDEBAR & CONTROL -----------------
st.sidebar.markdown("<h2 style='text-align: center; color: #10B981;'>⚡ AI Trading Assistant</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Watchlist presets by sector
st.sidebar.subheader("Watchlist Management")
st.sidebar.markdown("<small style='color: #64748B;'>Select presets from top industries or type custom symbols below.</small>", unsafe_allow_html=True)

# Sort sectors by size (number of tickers) for default selection
sorted_sectors = sorted(SECTORS.keys(), key=lambda k: len(SECTORS[k]), reverse=True)
default_selected_sectors = sorted_sectors[:5]

# Master multiselect for active industries/sectors
selected_sectors = st.sidebar.multiselect(
    "📁 Select Industries / Sectors",
    options=sorted(list(SECTORS.keys())),
    default=[s for s in default_selected_sectors if s in SECTORS],
    key="sidebar_active_sectors_multiselect",
    help="Select which industries to view. Custom/other added tickers will appear under their respective resolved sectors."
)

# Render dynamic multiselects for selected industries
selected_by_sector = []
for idx, sector_name in enumerate(selected_sectors):
    options = SECTORS[sector_name]
    selected = st.sidebar.multiselect(
        f"📂 {sector_name}",
        options=options,
        default=options,
        key=f"sidebar_sector_select_{sector_name}_{idx}"
    )
    selected_by_sector.extend(selected)

# Custom Symbols — staging area (always starts empty; merged tickers go into the main pool)
# Use session state to track the input value so we can clear it after merge
if "custom_symbol_input" not in st.session_state:
    st.session_state["custom_symbol_input"] = ""

CUSTOM_TICKERS_FILE = os.path.join(BASE_DIR, "custom_tickers.json")
NIFTY500_CACHE_FILE = os.path.join(BASE_DIR, "nifty500_config.json")

st.sidebar.markdown("""
<small style='color:#64748B;'>Enter NSE symbol(s) below and click <b>Add & Merge</b>.
They will be auto-classified by sector, permanently added to the main 500+ ticker pool, and included in all future pipeline runs automatically.</small>
""", unsafe_allow_html=True)

custom_watchlist_input = st.sidebar.text_input(
    "➕ Add Custom NSE Symbol",
    value=st.session_state["custom_symbol_input"],
    placeholder="e.g. INDOCO.NS or AEGISLOG",
    help="Type a single NSE symbol. It will be merged into the main ticker pool with sector info.",
    key="custom_symbol_text_input"
)

# Parse custom symbols (only for active watchlist display while typing)
custom_tickers_preview = [t.strip().upper() for t in custom_watchlist_input.split(",") if t.strip()]

# Compile complete active watchlist (sorted unique list)
active_watchlist = sorted(list(set(selected_by_sector + custom_tickers_preview)))

# ── Add & Merge button ────────────────────────────────────────────────────
save_custom_btn = st.sidebar.button(
    "✅ Add & Merge into Main Pool",
    help="Fetches sector details from NSE/yfinance and permanently merges this ticker into the main 500+ stock pool. The input box will be cleared after saving."
)

if save_custom_btn and custom_watchlist_input.strip():
    with st.spinner("Fetching sector info and merging into main ticker pool..."):
        # Normalise symbols
        new_syms = []
        for t in custom_tickers_preview:
            sym = t if t.endswith(".NS") else f"{t}.NS"
            new_syms.append(sym)
        new_syms = sorted(list(set(new_syms)))

        # ── Load existing Nifty 500 + extras config ────────────────────
        nifty_data = {"tickers": [], "sectors": {}}
        if os.path.exists(NIFTY500_CACHE_FILE):
            try:
                with open(NIFTY500_CACHE_FILE, "r") as f:
                    nifty_data = json.load(f)
            except Exception:
                pass

        nifty_tickers  = nifty_data.setdefault("tickers", [])
        nifty_sectors  = nifty_data.setdefault("sectors", {})

        resolved_details = []
        newly_added = []

        for sym in new_syms:
            # Skip if already in the main pool
            if sym in nifty_tickers:
                resolved_details.append(f"• `{sym}` — already in main pool ✓")
                continue

            # ── Resolve sector ─────────────────────────────────────────
            industry = "Custom Tickers"
            found_sector = False

            # 1. Check Nifty 500 sectors first (fast, offline)
            for s_name, s_list in nifty_sectors.items():
                if sym in s_list:
                    industry = s_name
                    found_sector = True
                    break

            # 2. Fallback → yfinance .info (live lookup)
            if not found_sector:
                try:
                    info = yf.Ticker(sym).info
                    fetched_sector   = info.get("sector", "")
                    fetched_industry = info.get("industry", "")
                    fetched_company  = info.get("longName", sym)
                    fetched_exchange = info.get("exchange", "")
                    industry = fetched_industry or fetched_sector or "Custom Tickers"
                    resolved_details.append(
                        f"• `{sym}` ({fetched_company}) → **{industry}** [{fetched_exchange}]"
                    )
                except Exception:
                    resolved_details.append(f"• `{sym}` → **{industry}** (sector lookup failed)")
            else:
                resolved_details.append(f"• `{sym}` → **{industry}** (matched from Nifty 500 index)")

            # ── Add to main pool ───────────────────────────────────────
            nifty_tickers.append(sym)
            nifty_sectors.setdefault(industry, [])
            if sym not in nifty_sectors[industry]:
                nifty_sectors[industry].append(sym)
            newly_added.append(sym)

        if newly_added:
            # Deduplicate & sort
            nifty_data["tickers"] = sorted(list(set(nifty_tickers)))
            for s in nifty_sectors:
                nifty_sectors[s] = sorted(list(set(nifty_sectors[s])))

            try:
                with open(NIFTY500_CACHE_FILE, "w") as f:
                    json.dump(nifty_data, f, indent=4)

                # Also clear custom_tickers.json staging area
                with open(CUSTOM_TICKERS_FILE, "w") as f:
                    json.dump({"tickers": [], "sectors": {}}, f, indent=4)

                # Refresh in-memory config so sidebar/tables update immediately
                config.refresh_config()

                st.session_state["custom_symbol_input"] = ""  # clear input box

                detail_str = "\n".join(resolved_details)
                st.sidebar.success(
                    f"✅ Merged {len(newly_added)} new ticker(s) into the main pool!\n\n{detail_str}"
                )
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Failed to save to database: {e}")
        else:
            detail_str = "\n".join(resolved_details)
            st.sidebar.info(f"No new tickers to add.\n{detail_str}")
elif save_custom_btn:
    st.sidebar.warning("Please enter at least one NSE symbol before clicking Add & Merge.")

custom_tickers = custom_tickers_preview  # keep compatibility alias

# Import Trending Tickers
import_trending_btn = st.sidebar.button("📥 Import Trending Market Stocks", help="Scrapes current trending, most active, top-gaining, and top-losing NSE stocks and merges them into your watchlist database.")

if import_trending_btn:
    with st.spinner("Fetching trending market-wide tickers..."):
        trending_list = fetch_trending_tickers()
        if not trending_list:
            st.sidebar.error("Failed to fetch trending symbols from the web. Please try again later.")
        else:
            # Load existing custom tickers
            CUSTOM_TICKERS_FILE = os.path.join(BASE_DIR, "custom_tickers.json")
            custom_data = {"tickers": [], "sectors": {}}
            if os.path.exists(CUSTOM_TICKERS_FILE):
                try:
                    with open(CUSTOM_TICKERS_FILE, "r") as f:
                        custom_data = json.load(f)
                except Exception:
                    pass
            
            tickers_list = custom_data.setdefault("tickers", [])
            sectors_dict = custom_data.setdefault("sectors", {})
            
            # Load Nifty 500 sectors config for resolving sectors fast
            nifty_sectors = {}
            NIFTY500_CACHE_FILE = os.path.join(BASE_DIR, "nifty500_config.json")
            if os.path.exists(NIFTY500_CACHE_FILE):
                try:
                    with open(NIFTY500_CACHE_FILE, "r") as f:
                        nifty_data = json.load(f)
                    nifty_sectors = nifty_data.get("sectors", {})
                except Exception:
                    pass
            
            added_tickers = []
            for t in trending_list:
                if t not in tickers_list:
                    # Resolve industry sector
                    industry = "Custom Tickers"
                    found_sector = False
                    for s_name, s_tickers in nifty_sectors.items():
                        if t in s_tickers:
                            industry = s_name
                            found_sector = True
                            break
                    
                    if not found_sector:
                        try:
                            ticker_info = yf.Ticker(t).info
                            fetched_ind = ticker_info.get("industry")
                            if fetched_ind:
                                industry = fetched_ind
                        except Exception:
                            industry = "Custom Tickers"
                            
                    tickers_list.append(t)
                    sectors_dict.setdefault(industry, []).append(t)
                    added_tickers.append(t)
            
            if added_tickers:
                custom_data["tickers"] = sorted(list(set(tickers_list)))
                for s_name in sectors_dict:
                    sectors_dict[s_name] = sorted(list(set(sectors_dict[s_name])))
                custom_data["sectors"] = sectors_dict
                
                try:
                    with open(CUSTOM_TICKERS_FILE, "w") as f:
                        json.dump(custom_data, f, indent=4)
                        
                    config.refresh_config()
                    st.sidebar.success(f"Added {len(added_tickers)} new trending stock(s):\n" + ", ".join(added_tickers))
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Failed to update database: {e}")
            else:
                st.sidebar.info("All current trending symbols are already in your watchlist database.")

# Action buttons
st.sidebar.subheader("Model Operations")
retrain_btn = st.sidebar.button("⚙️ Run Watchlist Pipeline", help="Re-downloads price history, crawls news headlines, and updates ML models for the active watchlist.")
scan_btn = st.sidebar.button("🔍 Run Full Market Scan", help="Re-downloads price history, crawls news headlines, and updates ML models for ALL sectors and tickers (takes ~2 minutes).")

# Display job log output container in sidebar if run
log_container = st.sidebar.empty()

if retrain_btn:
    with st.spinner("Executing Watchlist ML Pipeline..."):
        # Run scheduler as subprocess
        cmd = [sys.executable, "scheduler.py", "--tickers", ",".join(active_watchlist)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            st.sidebar.success("Pipeline executed successfully!")
            st.rerun()
        else:
            st.sidebar.error("Pipeline job failed!")
            st.sidebar.text(result.stderr)

if scan_btn:
    with st.spinner("Executing Full Market Scan (All Sectors)..."):
        # Run scheduler as subprocess with --all-sectors
        cmd = [sys.executable, "scheduler.py", "--all-sectors"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            st.sidebar.success("Full market scan completed successfully!")
            st.rerun()
        else:
            st.sidebar.error("Full market scan failed!")
            st.sidebar.text(result.stderr)

# Load stored recommendations (predictions)
predictions = load_recommendations()

# ----------------- LIVE MARKET CACHE -----------------
@st.cache_data(ttl=300) # cache live prices for 5 minutes
def fetch_live_market_data(tickers):
    """Downloads current session price details for watchlist tickers."""
    data = {}
    if not tickers:
        return data
        
    try:
        # Download historical prices (to get yesterday's close)
        history = yf.download(tickers, period="5d", progress=False)
        # Download today's live active session price
        today = yf.download(tickers, period="1d", progress=False)
        
        if history.empty or today.empty:
            return {}
            
        is_multi_hist = isinstance(history.columns, pd.MultiIndex)
        is_multi_today = isinstance(today.columns, pd.MultiIndex)
        
        for t in tickers:
            try:
                # Resolve today's close, high, and low prices
                if is_multi_today and t in today['Close']:
                    today_close = today['Close'][t].dropna()
                    today_high = today['High'][t].dropna()
                    today_low = today['Low'][t].dropna()
                elif not is_multi_today:
                    today_close = today['Close'].dropna()
                    today_high = today['High'].dropna()
                    today_low = today['Low'].dropna()
                else:
                    continue
                    
                # Resolve historical closes
                if is_multi_hist and t in history['Close']:
                    hist_close = history['Close'][t].dropna()
                elif not is_multi_hist:
                    hist_close = history['Close'].dropna()
                else:
                    continue
                
                # Exclude today's date from historical data if it exists there as NaN
                if not today_close.empty:
                    today_date = today_close.index[-1]
                    hist_close = hist_close[hist_close.index < today_date]
                    
                if not today_close.empty and not hist_close.empty:
                    data[t] = {
                        "live_price": float(today_close.iloc[-1]),
                        "prev_price": float(hist_close.iloc[-1]),
                        "high": float(today_high.iloc[-1]),
                        "low": float(today_low.iloc[-1])
                    }
            except Exception:
                pass
    except Exception:
        pass
    return data

# Compile all unique tickers that have active predictions or are in the active watchlist
all_price_tickers = sorted(list(set(active_watchlist + list(predictions.keys()))))
live_market = fetch_live_market_data(all_price_tickers)

# ----------------- DASHBOARD BODY -----------------
st.title("⚡ AI Technical Trading Assistant")
st.markdown("##### Tabular Machine Learning Predictions & Live Portfolio Monitoring Net of Indian Taxes")

# Check if model predictions are present, warning user if missing
if not predictions:
    st.warning("⚠️ Warning: No pre-calculated model recommendations found. Please click '⚙️ Run Model Pipeline' in the sidebar to fetch data and train the neural XGBoost models.")
else:
    missing_tickers = [t for t in active_watchlist if t not in predictions]
    if missing_tickers:
        st.warning(f"⚠️ Warning: The following ticker(s) in your watchlist do not have trained ML models yet: **{', '.join(missing_tickers)}**. Please click '⚙️ Run Model Pipeline' in the sidebar to fetch data and train models for them.")

# ----------------- PORTFOLIO SECTION -----------------
st.markdown("<div class='section-header'><h3>📈 Live Portfolio Holdings & Break-Even Alert Monitor</h3></div>", unsafe_allow_html=True)

portfolio = load_portfolio()

# Split layout for Portfolio: Inputs & Table
p_add_col, p_list_col = st.columns([1, 3.2])

with p_add_col:
    st.markdown("##### Add Stock Transaction")
    
    # Pre-fill Selector
    prefill_options = ["Custom (No pre-fill)"] + sorted(list(predictions.keys()))
    selected_prefill = st.selectbox("Pre-fill from Recommendations", prefill_options, key="prefill_selector")
    
    # Default values
    default_ticker = active_watchlist[0] if active_watchlist else ""
    default_price = 100.0
    default_tp = 110.0
    default_sl = 95.0
    
    if selected_prefill != "Custom (No pre-fill)" and selected_prefill in predictions:
        pred_item = predictions[selected_prefill]
        default_ticker = selected_prefill
        default_price = float(live_market.get(selected_prefill, {}).get("live_price", pred_item.get("current_price", 100.0)))
        pred_return = pred_item.get("predicted_return", 0.0)
        default_tp = float(default_price * (1.0 + pred_return))
        default_sl = float(pred_item.get("stop_loss_price", default_price * 0.95))
        
    # Search filter text input for Ticker selectbox
    ticker_search_query = st.text_input("🔍 Search Ticker Symbol (filters dropdown below)", "", key="ticker_search_filter_input")
    filtered_tickers = [t for t in all_price_tickers if ticker_search_query.upper() in t]
    if not filtered_tickers:
        filtered_tickers = all_price_tickers
        
    add_ticker = st.selectbox(
        "Ticker Symbol", 
        filtered_tickers, 
        index=filtered_tickers.index(default_ticker) if default_ticker in filtered_tickers else 0,
        key="ticker_selector_widget"
    )
    
    # If the user changes ticker directly, try to load defaults for it
    if add_ticker != default_ticker and add_ticker in predictions:
        pred_item = predictions[add_ticker]
        default_price = float(live_market.get(add_ticker, {}).get("live_price", pred_item.get("current_price", 100.0)))
        pred_return = pred_item.get("predicted_return", 0.0)
        default_tp = float(default_price * (1.0 + pred_return))
        default_sl = float(pred_item.get("stop_loss_price", default_price * 0.95))
        
    add_qty = st.number_input("Quantity (Shares)", min_value=1, value=10, step=1)
    add_price = st.number_input("Purchase Price (INR)", min_value=0.1, value=default_price, step=1.0, key="purchase_price_input")
    add_tp = st.number_input("Target Take-Profit / Upper Limit (INR)", min_value=0.1, value=default_tp, step=1.0, key="tp_input")
    add_sl = st.number_input("Target Stop-Loss / Lower Limit (INR)", min_value=0.1, value=default_sl, step=1.0, key="sl_input")
    add_date = st.date_input("Purchase Date", value=datetime.today())
    add_broker = st.selectbox("Trading Broker", list(BROKER_CONFIGS.keys()))
    
    submitted = st.button("💼 Record Purchase", width='stretch')
    if submitted:
        # Append transaction
        new_tx = {
            "id": str(int(datetime.now().timestamp())),
            "ticker": add_ticker,
            "quantity": float(add_qty),
            "purchase_price": float(add_price),
            "purchase_date": str(add_date),
            "broker": add_broker,
            "target_take_profit": float(add_tp),
            "target_stop_loss": float(add_sl)
        }
        portfolio.append(new_tx)
        save_portfolio(portfolio)
        st.success(f"Holding added with locked limits: TP ₹{add_tp:.2f} | SL ₹{add_sl:.2f}!")
        st.rerun()

with p_list_col:
    if not portfolio:
        st.info("No active holdings recorded. Add your stock transactions in the form to track net P&L and monitor thresholds in real-time.")
    else:
        # Build portfolio DataFrame
        rows = []
        tot_buy_cost_net = 0.0
        tot_curr_value_net = 0.0
        
        for item in portfolio:
            t = item["ticker"]
            qty = item["quantity"]
            buy_p = item["purchase_price"]
            broker = item["broker"]
            
            # Fetch live current price
            curr_p = buy_p
            if t in live_market:
                curr_p = live_market[t]["live_price"]
                
            # Get metrics
            metrics = calculate_portfolio_metrics(qty, buy_p, curr_p, broker)
            
            tot_buy_cost_net += metrics["buy_cost_net"]
            tot_curr_value_net += metrics["current_value_net"]
            
            # Resolve predicted value from model
            pred_5d_p = "-"
            stop_loss_p = "-"
            if t in predictions:
                pred_return = predictions[t].get("predicted_return", 0.0)
                pred_5d_p = f"₹{round(curr_p * (1.0 + pred_return), 2)}"
                # Load stop loss
                stop_loss_p = f"₹{predictions[t].get('stop_loss_price', 0.0)}"

            # Resolve Alarm status
            alert_badge = "<span class='badge-safe'>✅ SAFE</span>"
            if curr_p < metrics["break_even_price"]:
                alert_badge = "<span class='badge-danger'>🚨 LOSS (BELOW B.E.)</span>"
            elif t in predictions and curr_p < predictions[t].get('stop_loss_price', 0.0):
                alert_badge = "<span class='badge-warning'>⚠️ STOP LOSS BREACHED</span>"
                
            # Load locked target boundaries
            target_take_profit = item.get("target_take_profit", buy_p * 1.10)
            target_stop_loss = item.get("target_stop_loss", buy_p * 0.95)
            
            # Resolve Target limit alarm status
            target_status_badge = "<span class='badge-warning'>⏳ HOLDING</span>"
            if curr_p >= target_take_profit:
                target_status_badge = "<span class='badge-safe'>🟢 TP TARGET HIT</span>"
            elif curr_p <= target_stop_loss:
                target_status_badge = "<span class='badge-danger'>🔴 SL BREACHED</span>"
                
            rows.append({
                "ID": item["id"],
                "Ticker": t,
                "Qty": int(qty),
                "Buy Price": f"₹{buy_p:.2f}",
                "Live Price": f"₹{curr_p:.2f}",
                "Broker": broker,
                "Break-Even Price": f"₹{metrics['break_even_price']:.2f}",
                "Locked TP (Upper)": f"₹{target_take_profit:.2f}",
                "Locked SL (Lower)": f"₹{target_stop_loss:.2f}",
                "Est. Value (Net)": f"₹{metrics['current_value_net']:.2f}",
                "Net P&L": f"₹{metrics['net_pnl']:.2f}",
                "Net P&L %": f"{metrics['net_pnl_percent']:.2f}%",
                "Model Forecast (5d)": pred_5d_p,
                "Target Status": target_status_badge,
                "Alert Status": alert_badge
            })
            
        df_portfolio = pd.DataFrame(rows)
        
        # Display aggregate indicators
        tot_net_pnl = tot_curr_value_net - tot_buy_cost_net
        tot_net_pnl_percent = (tot_net_pnl / tot_buy_cost_net) * 100 if tot_buy_cost_net > 0 else 0.0
        
        c_inv, c_val, c_pnl = st.columns(3)
        with c_inv:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>Total Net Investment</div>
                    <div class='metric-val'>₹{tot_buy_cost_net:,.2f}</div>
                    <div class='metric-sub' style='color:#94A3B8;'>Including buy-side fees</div>
                </div>
            """, unsafe_allow_html=True)
        with c_val:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>Current Value (Net)</div>
                    <div class='metric-val'>₹{tot_curr_value_net:,.2f}</div>
                    <div class='metric-sub' style='color:#94A3B8;'>Deducting projected sell-side fees</div>
                </div>
            """, unsafe_allow_html=True)
        with c_pnl:
            pnl_class = "negative" if tot_net_pnl < 0 else ""
            pnl_sign = "+" if tot_net_pnl >= 0 else ""
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>Net Portfolio P&L</div>
                    <div class='metric-val {pnl_class}'>{pnl_sign}₹{tot_net_pnl:,.2f}</div>
                    <div class='metric-sub {pnl_class}'>{pnl_sign}{tot_net_pnl_percent:,.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Group holdings rows by sector
        rows_by_sector = {}
        for r in rows:
            t_symbol = r["Ticker"]
            placed = False
            for sector_name, sector_list in SECTORS.items():
                if t_symbol in sector_list:
                    rows_by_sector.setdefault(sector_name, []).append(r)
                    placed = True
                    break
            if not placed:
                rows_by_sector.setdefault("Custom Tickers", []).append(r)
                
        # Display sector-wise holdings tables inside a single closed expander using a sector selectbox
        with st.expander("📁 View Sector-wise Holdings Breakdown", expanded=False):
            selected_sector = st.selectbox("Select Sector to View Holdings", options=list(rows_by_sector.keys()), key="portfolio_sector_select")
            sector_rows = rows_by_sector[selected_sector]
            
            # Calculate sector subtotals
            sect_val = sum(float(row["Est. Value (Net)"].replace("₹", "").replace(",", "")) for row in sector_rows)
            sect_pnl = sum(float(row["Net P&L"].replace("₹", "").replace(",", "")) for row in sector_rows)
            pnl_color = "#10B981" if sect_pnl >= 0 else "#EF4444"
            pnl_sign = "+" if sect_pnl >= 0 else ""
            
            st.markdown(f"##### 📁 {selected_sector} — Net Value: ₹{sect_val:,.2f} | P&L: <span style='color:{pnl_color};font-weight:700;'>{pnl_sign}₹{sect_pnl:,.2f}</span>", unsafe_allow_html=True)
            df_sect = pd.DataFrame(sector_rows)
            st.markdown(df_sect.drop(columns=["ID"]).to_html(escape=False, index=False), unsafe_allow_html=True)
        
        # Quick transaction deletion
        st.markdown("<br>", unsafe_allow_html=True)
        del_cols = st.columns([1, 4])
        with del_cols[0]:
            del_id = st.selectbox("Remove Record", options=[r["id"] for r in portfolio], format_func=lambda x: next(f"{r['ticker']} ({int(r['quantity'])} sh @ ₹{r['purchase_price']})" for r in portfolio if r["id"] == x))
            del_btn = st.button("🗑️ Delete Transaction", width='stretch')
            if del_btn:
                portfolio = [r for r in portfolio if r["id"] != del_id]
                save_portfolio(portfolio)
                st.success("Holding deleted!")
                st.rerun()

# ----------------- TOP 10 RECOMMENDATIONS PANEL -----------------
st.markdown("<div class='section-header'><h3>🏆 Top 10 Buy Recommendations Matrix</h3></div>", unsafe_allow_html=True)

if not predictions:
    st.info("No recommendation data found. Run a 'Full Market Scan' in the sidebar to generate recommendations.")
else:
    # Compile all BUY/highly bullish recommendations across all loaded predictions
    bullish_recs = []
    for t, pred in predictions.items():
        sig = pred.get("signal", "NEUTRAL")
        curr_p = live_market.get(t, {}).get("live_price", pred.get("current_price", 0.0))
        if curr_p == 0.0:
            continue
        
        pred_return = pred.get("predicted_return", 0.0)
        up_prob = pred.get("upward_probability", 0.0)
        val_mae = pred.get("val_mae", 0.0)
        rrr = pred.get("rrr", 0.0)
        take_profit = curr_p * (1.0 + pred_return)
        stop_loss_p = pred.get("stop_loss_price", 0.0)
        
        # Resolve sector name
        sector_name = "Custom / Other"
        for s_name, s_tickers in SECTORS.items():
            if t in s_tickers:
                sector_name = s_name
                break
                
        bullish_recs.append({
            "Ticker": t,
            "Sector": sector_name,
            "Signal": sig,
            "Live Price": curr_p,
            "Upward Prob": up_prob,
            "Predicted Return": pred_return,
            "Val. MAE (Unseen)": val_mae,
            "RRR": rrr,
            "Target Price (5d)": take_profit,
            "Stop-Loss Price": stop_loss_p,
            "Reasoning": pred.get("signal_reason", "")
        })
        
    # Sort: BUY signals first, then NEUTRAL, then SELL
    # Within each signal, sort by upward_probability descending, then predicted_return descending, then RRR descending
    def ranking_key(item):
        sig_priority = 0
        if item["Signal"] == "BUY":
            sig_priority = 2
        elif item["Signal"] == "NEUTRAL":
            sig_priority = 1
        else:
            sig_priority = 0
            
        return (sig_priority, item["Upward Prob"], item["Predicted Return"], item["RRR"])
        
    bullish_recs.sort(key=ranking_key, reverse=True)
    top_10 = bullish_recs[:10]
    
    if not top_10:
        st.info("No recommendations found. Please run a market scan to train models.")
    else:
        st.markdown("<small style='color: #64748B;'>The following are the top 10 ranked buy recommendations across all sectors based on classifier confidence, expected return, and validation truth (Val MAE).</small>", unsafe_allow_html=True)
        
        # Format the table beautifully
        top_rows = []
        for item in top_10:
            sig_html = f"<span class='badge-safe'>🟢 BUY</span>" if item["Signal"] == "BUY" else f"<span class='badge-warning'>🟡 NEUTRAL</span>"
            
            top_rows.append({
                "Ticker": f"<b>{item['Ticker']}</b>",
                "Sector": item["Sector"],
                "Signal": sig_html,
                "Live Price": f"₹{item['Live Price']:.2f}",
                "Upward Prob": f"{item['Upward Prob']:.2%}",
                "Predicted Return (5d)": f"{item['Predicted Return']:+.2%}",
                "Val. MAE (Unseen)": f"{item['Val. MAE (Unseen)']:.4f}",
                "RRR": f"{item['RRR']:.2f}",
                "Target Price (5d)": f"₹{item['Target Price (5d)']:.2f}",
                "Stop-Loss Price": f"₹{item['Stop-Loss Price']:.2f}",
                "Signal Reasoning": f"<small>{item['Reasoning']}</small>"
            })
            
        df_top = pd.DataFrame(top_rows)
        st.markdown(df_top.to_html(escape=False, index=False), unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True)

# ----------------- RANKED RECOMMENDATIONS SECTION -----------------
st.markdown("<div class='section-header'><h3>🧠 Neural Analyst — BUY / SELL / NEUTRAL Signals & Execution Brackets</h3></div>", unsafe_allow_html=True)

if not predictions:
    st.info("No recommendation summaries loaded. Run the model pipeline in the sidebar to generate data.")
else:
    # ── Model Training Metadata expander ──────────────────────────────────
    with st.expander("📐 Model Training & Validation Dataset Details (click to expand)", expanded=False):
        # Search filter text input for model metadata
        meta_search_query = st.text_input("🔍 Search Ticker Symbol (filters metadata rows below)", "", key="model_metadata_ticker_search")
        
        meta_rows = []
        for t in active_watchlist:
            if meta_search_query and meta_search_query.upper() not in t.upper():
                continue
            if t in predictions:
                p = predictions[t]
                meta_rows.append({
                    "Ticker":                    t,
                    "Algorithm":                 p.get("model_algorithm", "—"),
                    "Total Trading Days":        p.get("total_trading_days", "—"),
                    "Train Days (80%)":          p.get("train_trading_days", "—"),
                    "Train Period":              f"{p.get('train_start_date','—')}  →  {p.get('train_end_date','—')}",
                    "Val. Days (20%)":           p.get("validation_trading_days", "—"),
                    "Val. Period":               f"{p.get('validation_start_date','—')}  →  {p.get('validation_end_date','—')}",
                    "Train Accuracy":            f"{p.get('train_accuracy', 0):.2%}",
                    "Val. Accuracy (Unseen)":    f"{p.get('val_accuracy', 0):.2%}",
                    "Train MAE":                 f"{p.get('train_mae', 0):.4f}",
                    "Val. MAE (Unseen)":         f"{p.get('val_mae', 0):.4f}",
                    "Forecast Horizon":          f"{p.get('forecast_horizon_days', 5)} trading days",
                })
        if meta_rows:
            st.markdown(pd.DataFrame(meta_rows).to_html(escape=False, index=False), unsafe_allow_html=True)
        elif meta_search_query:
            st.info("No tickers in active watchlist match your search query.")
        st.markdown("""
            <br>
            <small style='color:#64748B'>
            ℹ️ <b>Train Accuracy / Train MAE</b> = performance on the 80% of historical data the model was trained on.<br>
            ℹ️ <b>Val. Accuracy / Val. MAE (Unseen)</b> = performance on the most recent 20% of data the model has <i>never seen</i> — this is the true real-world accuracy estimate.<br>
            ℹ️ <b>Forecast Horizon</b> = number of <i>market trading days</i> ahead the model predicts (not calendar days).
            </small>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Signal Filter Tabs ────────────────────────────────────────────────
    tab_all, tab_buy, tab_sell, tab_neutral = st.tabs(["All Signals", "🟢 BUY Only", "🔴 SELL Only", "🟡 NEUTRAL Only"])

    def build_rec_rows(filter_signal=None):
        rows = []
        for t in active_watchlist:
            if t not in predictions:
                if filter_signal is not None:
                    continue
                
                curr_p     = live_market.get(t, {}).get("live_price", 0.0)
                prev_p     = live_market.get(t, {}).get("prev_price", curr_p)
                day_change = curr_p - prev_p
                day_pct    = (day_change / prev_p * 100) if prev_p else 0.0
                day_arrow  = "▲" if day_change >= 0 else "▼"
                day_color  = "#10B981" if day_change >= 0 else "#EF4444"
                
                day_badge = (
                    f"<span style='color:{day_color};font-weight:700;'>"
                    f"{day_arrow} ₹{abs(day_change):.2f} ({day_pct:+.2f}%)</span>"
                ) if curr_p > 0 else "—"
                
                rows.append({
                    "Ticker":                t,
                    "Live Price":            f"₹{curr_p:.2f}" if curr_p > 0 else "—",
                    "Day Change":            day_badge,
                    "Signal":                "<span class='badge-warning'>⏳ PENDING</span>",
                    "↑ Prob (Up)":           "—",
                    "↓ Prob (Down)":         "—",
                    "Predicted Δ (5d)":      "—",
                    "Entry / Buy At":        "—",
                    "Take-Profit":           "—",
                    "Stop-Loss":             "—",
                    "Suggested Exit (Sell)": "—",
                    "RRR":                   "—",
                    "Action":                "<span class='badge-warning'>⏳ PENDING</span>",
                    "Signal Reasoning":      "No model data. Click 'Run Model Pipeline' in the sidebar to train models for this ticker.",
                })
                continue

            pred       = predictions[t]
            sig        = pred.get("signal", "NEUTRAL")
            if filter_signal and sig != filter_signal:
                continue

            curr_p     = live_market.get(t, {}).get("live_price", pred["current_price"])
            prev_p     = live_market.get(t, {}).get("prev_price", curr_p)
            day_change = curr_p - prev_p
            day_pct    = (day_change / prev_p * 100) if prev_p else 0.0
            day_arrow  = "▲" if day_change >= 0 else "▼"
            day_color  = "#10B981" if day_change >= 0 else "#EF4444"

            pred_return  = pred.get("predicted_return", 0.0)
            up_prob      = pred.get("upward_probability", 0.0)
            down_prob    = pred.get("downward_probability", 1 - up_prob)
            take_profit  = curr_p * (1.0 + pred_return)
            stop_loss_p  = pred.get("stop_loss_price", 0.0)
            exit_p       = pred.get("suggested_exit_price")  # only populated for SELL
            rrr          = pred.get("rrr", 0.0)

            # Signal badge
            if sig == "BUY":
                sig_badge = "<span class='badge-safe'>🟢 BUY</span>"
            elif sig == "SELL":
                sig_badge = "<span class='badge-danger'>🔴 SELL</span>"
            else:
                sig_badge = "<span class='badge-warning'>🟡 NEUTRAL</span>"

            # Day change badge
            day_badge = (
                f"<span style='color:{day_color};font-weight:700;'>"
                f"{day_arrow} ₹{abs(day_change):.2f} ({day_pct:+.2f}%)</span>"
            )

            # Build price bracket cells depending on signal
            if sig == "BUY":
                entry_cell      = f"₹{curr_p:.2f}"
                target_cell     = f"<span style='color:#10B981;font-weight:700;'>₹{take_profit:.2f}</span>"
                stop_cell       = f"<span style='color:#EF4444;'>₹{stop_loss_p:.2f}</span>"
                exit_cell       = "—"
            elif sig == "SELL":
                entry_cell      = "—"
                target_cell     = "—"
                stop_cell       = f"<span style='color:#EF4444;'>₹{stop_loss_p:.2f}</span>"
                exit_cell       = (
                    f"<span style='color:#F59E0B;font-weight:700;'>₹{exit_p:.2f}</span>"
                    if exit_p else "—"
                )
            else:
                entry_cell  = f"₹{curr_p:.2f}"
                target_cell = f"₹{take_profit:.2f}"
                stop_cell   = f"₹{stop_loss_p:.2f}"
                exit_cell   = "—"

            rows.append({
                "Ticker":                t,
                "Live Price":            f"₹{curr_p:.2f}",
                "Day Change":            day_badge,
                "Signal":                sig_badge,
                "↑ Prob (Up)":           f"{up_prob:.2%}",
                "↓ Prob (Down)":         f"{down_prob:.2%}",
                "Predicted Δ (5d)":      f"{pred_return:+.2%}",
                "Entry / Buy At":        entry_cell,
                "Take-Profit":           target_cell,
                "Stop-Loss":             stop_cell,
                "Suggested Exit (Sell)": exit_cell,
                "RRR":                   f"{rrr:.2f}",
                "Action":                sig_badge,
                "Signal Reasoning":      pred.get("signal_reason", "—"),
            })
            
        def get_signal_type(ticker):
            if ticker not in predictions:
                return "PENDING"
            return predictions[ticker].get("signal", "NEUTRAL")
            
        def get_upward_prob(ticker):
            if ticker not in predictions:
                return -1.0
            return predictions[ticker].get("upward_probability", 0.0)

        # Sort: BUY first by up_prob desc, then NEUTRAL, then SELL, then PENDING
        order = {"BUY": 0, "NEUTRAL": 1, "SELL": 2, "PENDING": 3}
        rows.sort(key=lambda r: (
            order.get(get_signal_type(r["Ticker"]), 3),
            -get_upward_prob(r["Ticker"])
        ))
        return rows

    def render_table_sectorwise(rows, signal_type="All"):
        if not rows:
            st.info("No signals match this filter.")
            return
            
        with st.expander(f"📁 View {signal_type} Signals by Sector", expanded=True if signal_type in ["BUY", "SELL"] else False):
            # Ticker search query
            ticker_q = st.text_input("🔍 Search Ticker Symbol (filters table rows below)", "", key=f"signals_ticker_search_{signal_type}")
            
            # Filter rows by ticker
            filtered_rows = rows
            if ticker_q:
                filtered_rows = []
                for r in rows:
                    t_clean = r["Ticker"].replace("<b>", "").replace("</b>", "").strip().upper()
                    if ticker_q.upper() in t_clean:
                        filtered_rows.append(r)
            
            if not filtered_rows:
                st.info("No tickers match your search query.")
                return
                
            # Group rows by sector
            rows_by_sector = {}
            for r in filtered_rows:
                t_symbol = r["Ticker"]
                t_clean = t_symbol.replace("<b>", "").replace("</b>", "").strip()
                placed = False
                for sector_name, sector_list in SECTORS.items():
                    if t_clean in sector_list:
                        rows_by_sector.setdefault(sector_name, []).append(r)
                        placed = True
                        break
                if not placed:
                    rows_by_sector.setdefault("Custom Tickers", []).append(r)
                    
            # Sector search query
            sector_q = st.text_input("🔍 Filter Sector/Industry names (filters dropdown below)", "", key=f"signals_sector_search_{signal_type}")
            matching_sectors = sorted(list(rows_by_sector.keys()))
            filtered_sectors = [s for s in matching_sectors if sector_q.upper() in s.upper()]
            if not filtered_sectors:
                filtered_sectors = matching_sectors
                
            selected_sector = st.selectbox(
                "Select Sector to View Signals", 
                options=filtered_sectors, 
                key=f"signals_sector_select_{signal_type}"
            )
            
            if selected_sector and selected_sector in rows_by_sector:
                sector_rows = rows_by_sector[selected_sector]
                
                # Count the signals in this sector
                buys = sum(1 for row in sector_rows if "BUY" in row["Signal"])
                sells = sum(1 for row in sector_rows if "SELL" in row["Signal"])
                neutrals = sum(1 for row in sector_rows if "NEUTRAL" in row["Signal"])
                pendings = sum(1 for row in sector_rows if "PENDING" in row["Signal"])
                
                pills = []
                if buys > 0: pills.append(f"🟢 {buys} BUY")
                if sells > 0: pills.append(f"🔴 {sells} SELL")
                if neutrals > 0: pills.append(f"🟡 {neutrals} NEUTRAL")
                if pendings > 0: pills.append(f"⏳ {pendings} PENDING")
                
                pills_str = " | ".join(pills)
                st.markdown(f"##### 📁 {selected_sector} Signals ({len(sector_rows)} stock{'s' if len(sector_rows) > 1 else ''}) — {pills_str}")
                
                df_sect = pd.DataFrame(sector_rows).drop(columns=["Action"])
                st.markdown(df_sect.to_html(escape=False, index=False), unsafe_allow_html=True)

    with tab_all:
        render_table_sectorwise(build_rec_rows(), "All")
    with tab_buy:
        render_table_sectorwise(build_rec_rows("BUY"), "BUY")
    with tab_sell:
        render_table_sectorwise(build_rec_rows("SELL"), "SELL")
    with tab_neutral:
        render_table_sectorwise(build_rec_rows("NEUTRAL"), "NEUTRAL")

# ----------------- MARKET-WIDE BREAKOUT SCANNER SECTION -----------------
st.markdown("<div class='section-header'><h3>⚡ Market-Wide Daily Breakout Scanner</h3></div>", unsafe_allow_html=True)

with st.expander("⚡ View Daily Breakout Stocks (≥ +10% Gainers)", expanded=False):
    st.markdown("""
        This tool scans the entire National Stock Exchange (NSE) of India (over 2,100+ listed symbols) for stocks trading with 
        a day change of **+10% or more**. Daily breakout results can be permanently saved to your watchlist database.
    """)
    
    # Load cached breakout scan results
    scan_data = load_scan_results()
    
    if scan_data:
        st.markdown(f"⏱️ **Last Full Market Scan**: `{scan_data['last_scan_time']}`")
        stocks = scan_data.get("stocks", [])
        if not stocks:
            st.info("No stocks with a daily gain of ≥ +10% were detected in the last scan.")
        else:
            # Build list of dicts for presentation
            table_rows = []
            for s in stocks:
                vol_formatted = f"{s['Volume']:,}" if s['Volume'] > 0 else "—"
                table_rows.append({
                    "Ticker Symbol": f"<b>{s['Ticker']}</b>",
                    "Previous Close": f"₹{s['Prev Close']:.2f}",
                    "Live/Close Price": f"₹{s['Live Price']:.2f}",
                    "Daily Return": f"<span style='color:#10B981;font-weight:700;'>+{s['Change %']:.2f}%</span>",
                    "Today's Volume": vol_formatted
                })
            df_breakout = pd.DataFrame(table_rows)
            st.markdown(df_breakout.to_html(escape=False, index=False), unsafe_allow_html=True)
            
            # Action: select and add to watchlist
            st.markdown("<br>", unsafe_allow_html=True)
            col_add_sel, col_add_btn = st.columns([2.0, 1.0])
            with col_add_sel:
                ticker_to_add = st.selectbox(
                    "Select a Breakout Stock to Monitor",
                    options=[s["Ticker"] for s in stocks],
                    key="breakout_watchlist_selectbox"
                )
            with col_add_btn:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                add_btn = st.button("➕ Add to Watchlist Database", width='stretch', key="breakout_watchlist_add_btn")
                if add_btn:
                    success, msg = add_ticker_to_watchlist(ticker_to_add)
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
    else:
        st.info("No scan history found. Click the button below to execute a live full market scan.")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Manual scan trigger
    if st.button("🔍 Run New Live Market Scan (2,100+ Stocks)", key="run_live_breakout_scan_btn"):
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        def update_progress(curr, tot, msg):
            frac = float(curr) / float(tot)
            progress_bar.progress(min(frac, 1.0))
            status_text.text(msg)
            
        with st.spinner("Scanning all NSE listed stocks in batch..."):
            results = run_breakout_scan(progress_callback=update_progress)
            save_scan_results(results)
        st.success(f"Scan finished! Detected {len(results)} breakout stocks.")
        st.rerun()

# ----------------- TECHNICAL CHARTS SECTION -----------------
st.markdown("<div class='section-header'><h3>📊 Interactive Technical Analysis Charts</h3></div>", unsafe_allow_html=True)

@st.cache_data(ttl=600)
def load_chart_history(ticker):
    """Fetch historical daily price rows for charting."""
    try:
        df = yf.download(ticker, period="6mo", progress=False)
        if df.empty:
            return pd.DataFrame()
        # Handle multiindex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Patch today's live session data if history download returns NaNs
        try:
            if not df.empty:
                last_idx = df.index[-1]
                if df.loc[last_idx].isna().any():
                    today_df = yf.download(ticker, period="1d", progress=False)
                    if isinstance(today_df.columns, pd.MultiIndex):
                        today_df.columns = today_df.columns.get_level_values(0)
                    if not today_df.empty:
                        today_idx = today_df.index[-1]
                        if last_idx == today_idx:
                            for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
                                if col in df.columns and col in today_df.columns:
                                    val = today_df.loc[today_idx, col]
                                    if pd.notna(val):
                                        df.loc[last_idx, col] = float(val)
        except Exception:
            pass
        
        # SMAs
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        
        # Technical indices
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0.0)).copy()
        loss = (-delta.where(delta < 0, 0.0)).copy()
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, 1e-9)
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR_14'] = true_range.rolling(14).mean()
        
        return df
    except Exception:
        return pd.DataFrame()

# Group active watchlist by sector
grouped_watchlist = group_tickers_by_sector(active_watchlist)

with st.expander("📊 View Technical Analysis Charts", expanded=False):
    chart_search_query = st.text_input("🔍 Search Ticker for Charting (filters dropdown below)", "", key="charts_search_filter_input")
    filtered_chart_tickers = [t for t in all_price_tickers if chart_search_query.upper() in t]
    if not filtered_chart_tickers:
        filtered_chart_tickers = all_price_tickers
        
    chart_ticker = st.selectbox(
        "Select Symbol for Charting", 
        options=filtered_chart_tickers, 
        index=filtered_chart_tickers.index("TATAPOWER.NS") if "TATAPOWER.NS" in filtered_chart_tickers else 0,
        key="charts_global_select"
    )
    
    df_chart = load_chart_history(chart_ticker)
    if df_chart.empty:
        st.error(f"Could not load chart history for {chart_ticker}.")
    else:
        # Build 3-row subplots
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03, 
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Row 1: Candlesticks
        fig.add_trace(
            obj.Candlestick(
                x=df_chart.index,
                open=df_chart['Open'],
                high=df_chart['High'],
                low=df_chart['Low'],
                close=df_chart['Close'],
                name="Candlesticks"
            ),
            row=1, col=1
        )
        
        # Overlay SMA lines
        fig.add_trace(
            obj.Scatter(x=df_chart.index, y=df_chart['SMA_50'], name="50-day SMA", line=dict(color="#10B981", width=1.5)),
            row=1, col=1
        )
        fig.add_trace(
            obj.Scatter(x=df_chart.index, y=df_chart['SMA_200'], name="200-day SMA", line=dict(color="#3B82F6", width=1.5)),
            row=1, col=1
        )
        
        # Row 2: RSI
        fig.add_trace(
            obj.Scatter(x=df_chart.index, y=df_chart['RSI_14'], name="RSI (14)", line=dict(color="#F59E0B", width=1.5)),
            row=2, col=1
        )
        # Highlight lines for RSI overbought / oversold
        fig.add_hline(y=70, line_dash="dash", line_color="#EF4444", line_width=1, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#10B981", line_width=1, row=2, col=1)
        
        # Row 3: ATR
        fig.add_trace(
            obj.Scatter(x=df_chart.index, y=df_chart['ATR_14'], name="ATR (14)", line=dict(color="#EC4899", width=1.5)),
            row=3, col=1
        )
        
        fig.update_layout(
            title=f"📊 6-Month Historical Technical Analysis Chart — {chart_ticker}",
            template="plotly_dark",
            height=500,
            xaxis_rangeslider_visible=False,
            margin=dict(t=50, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, width='stretch', key=f"plotly_hist_{chart_ticker}")
    
        # ── Future Projection Analysis Chart ──
        if chart_ticker in predictions:
            st.markdown("<br>", unsafe_allow_html=True)
            pred = predictions[chart_ticker]
            last_date = df_chart.index[-1]
            last_close = float(df_chart['Close'].iloc[-1])
            predicted_p = float(pred.get("predicted_price", last_close))
            take_profit_p = float(pred.get("take_profit_price", last_close))
            stop_loss_p = float(pred.get("stop_loss_price", last_close))
            signal = pred.get("signal", "NEUTRAL")
            
            # Get recent context (last 15 trading days for visual context)
            df_recent = df_chart.tail(15)
            recent_dates = df_recent.index.tolist()
            recent_closes = df_recent['Close'].tolist()
            
            # Generate next 5 trading days
            future_dates = []
            curr_d = pd.to_datetime(last_date)
            while len(future_dates) < 5:
                curr_d += pd.Timedelta(days=1)
                if curr_d.weekday() < 5:  # Mon-Fri
                    future_dates.append(curr_d)
                    
            # Projection lines
            proj_dates = [pd.to_datetime(last_date)] + future_dates
            proj_prices = np.linspace(last_close, predicted_p, len(proj_dates))
            
            # Select color based on signal
            if signal == "BUY":
                pred_color = "#10B981"  # Vibrant green
            elif signal == "SELL":
                pred_color = "#EF4444"  # Vibrant red
            else:
                pred_color = "#F59E0B"  # Vibrant amber/yellow
                
            fig_proj = obj.Figure()
            
            # 1. Plot historical context close price
            fig_proj.add_trace(
                obj.Scatter(
                    x=recent_dates,
                    y=recent_closes,
                    name="Recent Close Price",
                    line=dict(color="#64748B", width=2, dash="solid"),
                    mode="lines+markers"
                )
            )
            
            # 2. Projected path
            fig_proj.add_trace(
                obj.Scatter(
                    x=proj_dates,
                    y=proj_prices,
                    name=f"Expected Forecast ({signal})",
                    line=dict(color=pred_color, width=3, dash="dash"),
                    mode="lines+markers",
                    marker=dict(size=8, symbol="circle")
                )
            )
            
            # 3. Take-profit horizontal line
            fig_proj.add_trace(
                obj.Scatter(
                    x=proj_dates,
                    y=[take_profit_p] * len(proj_dates),
                    name=f"Take-Profit Target (₹{take_profit_p:.2f})",
                    line=dict(color="rgba(16, 185, 129, 0.8)", width=1.5, dash="dot"),
                    mode="lines"
                )
            )
            
            # 4. Stop-loss horizontal line
            fig_proj.add_trace(
                obj.Scatter(
                    x=proj_dates,
                    y=[stop_loss_p] * len(proj_dates),
                    name=f"Stop-Loss Limit (₹{stop_loss_p:.2f})",
                    line=dict(color="rgba(239, 68, 68, 0.8)", width=1.5, dash="dot"),
                    mode="lines"
                )
            )
            
            fig_proj.update_layout(
                title=f"🔮 5-Day Machine Learning Future Projection — {chart_ticker}",
                template="plotly_dark",
                height=350,
                xaxis_title="Timeline (Trading Sessions)",
                yaxis_title="Stock Price (INR)",
                margin=dict(t=50, b=40, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_proj, width='stretch', key=f"plotly_proj_{chart_ticker}")
            
            # ── Historical Prediction Error Margin Analysis ──
            val_errors_all = pred.get("validation_errors", [])
            if val_errors_all:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"##### 📉 Prediction Error Margin Analysis — {chart_ticker}")
                st.markdown("""
                    This table tracks the historical predictions from the backtesting dataset, comparing the stock price on the prediction day against what was predicted and what actually occurred 5 trading days later.
                """)
                
                col_inspect_count, _ = st.columns([2.0, 3.0])
                with col_inspect_count:
                    num_rows = st.selectbox(
                        "🔍 Show Last N Predictions",
                        options=[5, 10, 15, 20, 30, len(val_errors_all)],
                        index=2 if len(val_errors_all) >= 15 else (1 if len(val_errors_all) >= 10 else 0),
                        format_func=lambda x: "All Predictions" if x == len(val_errors_all) else f"Last {x} Predictions",
                        key=f"val_inspect_count_{chart_ticker}"
                    )
                
                # Render history rows (most recent first)
                recent_recs = val_errors_all[-num_rows:]
                err_rows = []
                for rec in reversed(recent_recs):
                    actual_ret = rec["actual_return_pct"]
                    pred_ret = rec["predicted_return_pct"]
                    err_margin = rec["error_margin_pct"]
                    
                    act_color = "#10B981" if actual_ret >= 0 else "#EF4444"
                    pred_color = "#10B981" if pred_ret >= 0 else "#EF4444"
                    
                    # Highlight error margin magnitude
                    abs_err = abs(err_margin)
                    if abs_err < 2.0:
                        err_badge = f"<span class='badge-safe'>{err_margin:+.2f}%</span>"
                    elif abs_err < 5.0:
                        err_badge = f"<span class='badge-warning'>{err_margin:+.2f}%</span>"
                    else:
                        err_badge = f"<span class='badge-danger'>{err_margin:+.2f}%</span>"
                        
                    # Calculate predicted price 5d later in INR
                    predicted_future_close = rec['actual_close'] * (1.0 + pred_ret / 100)
                    
                    err_rows.append({
                        "Date Predicted": rec["date"],
                        "Price at Prediction Date": f"₹{rec['actual_close']:.2f}",
                        "Predicted Price (5d Later)": f"₹{predicted_future_close:.2f}",
                        "Actual Price (5d Later)": f"₹{rec['actual_future_close']:.2f}",
                        "Predicted Return": f"<span style='color:{pred_color};font-weight:600;'>{pred_ret:+.2f}%</span>",
                        "Actual Return": f"<span style='color:{act_color};font-weight:600;'>{actual_ret:+.2f}%</span>",
                        "Error Margin (Diff)": err_badge
                    })
                
                df_err = pd.DataFrame(err_rows)
                st.markdown(df_err.to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.markdown("<br>", unsafe_allow_html=True)
                st.info("📉 No prediction error history available. Retrain the model pipeline to compile backtesting error margins.")
        else:
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"🔮 Future prediction data is not available for {chart_ticker}. Run the model training pipeline to generate projections.")

# ----------------- SENTIMENT & PUBLIC INDICES SECTION -----------------
st.markdown("<div class='section-header'><h3>📰 Live Financial Sentiment Newsboard</h3></div>", unsafe_allow_html=True)

with st.expander("📰 View Sentiment & Newsboard", expanded=False):
    # Search filter text input for News selectbox
    news_search_query = st.text_input("🔍 Search Ticker for News & Sentiment (filters dropdown below)", "", key="news_search_filter_input")
    filtered_news_tickers = [t for t in all_price_tickers if news_search_query.upper() in t]
    if not filtered_news_tickers:
        filtered_news_tickers = all_price_tickers
        
    news_ticker = st.selectbox(
        "Select Ticker Symbol for News & Sentiment", 
        options=filtered_news_tickers, 
        index=filtered_news_tickers.index("TATAPOWER.NS") if "TATAPOWER.NS" in filtered_news_tickers else 0,
        key="news_global_select"
    )
    
    # Resolve the selected ticker's industry sector
    news_industry = "Other"
    for ind_name, ind_tickers in SECTORS.items():
        if news_ticker in ind_tickers:
            news_industry = ind_name
            break
            
    industry_tickers = SECTORS.get(news_industry, [news_ticker])
    
    s_select_col, s_board_col = st.columns([1.5, 3.5])
    
    with s_select_col:
        st.markdown(f"##### Peer Group: {news_industry}")
        # Limit peer group displays to avoid overflow
        peer_display = sorted(list(set(industry_tickers)))[:8]
        for t in peer_display:
            sentiment_score = 0.0
            if t in predictions:
                sentiment_score = predictions[t].get("sentiment_score", 0.0)
            else:
                sentiment_score = fetch_live_sentiment(t)
                
            color = "#10B981" if sentiment_score > 0.1 else ("#EF4444" if sentiment_score < -0.1 else "#94A3B8")
            st.markdown(f"""
                <div style='background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 12px; margin-bottom: 10px;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span style='font-size: 0.95rem; font-weight: 600; color: #E2E8F0;'>{t}</span>
                        <span style='font-size: 1.3rem; font-weight: 800; color: {color};'>{sentiment_score:+.2f}</span>
                    </div>
                    <div style='font-size: 0.75rem; color: #64748B; margin-top: 4px;'>Yahoo Finance news score</div>
                </div>
            """, unsafe_allow_html=True)
            
    with s_board_col:
        
        st.markdown(f"##### News Headlines for {news_ticker}")
        
        # Retrieve news using yfinance directly (live) to display
        try:
            yf_tick = yf.Ticker(news_ticker)
            news_articles = yf_tick.news
            if news_articles:
                for article in news_articles[:5]:
                    content = article.get("content", {})
                    title = content.get("title", "")
                    summary = content.get("summary", "")
                    pub_date = content.get("pubDate", "")
                    link = content.get("canonicalUrl", content.get("clickThroughUrl", ""))
                    
                    # Analyze sentiment
                    score = score_text_sentiment(f"{title} {summary}")
                    badge_html = "<span class='badge-warning'>NEUTRAL</span>"
                    if score > 0.1:
                        badge_html = f"<span class='badge-safe'>POS (+{score:.2f})</span>"
                    elif score < -0.1:
                        badge_html = f"<span class='badge-danger'>NEG ({score:.2f})</span>"
                        
                    st.markdown(f"""
                        <div style='background: rgba(30, 41, 59, 0.25); border-left: 4px solid #10B981; border-radius: 4px; padding: 12px; margin-bottom: 12px;'>
                            <div style='display: flex; justify-content: space-between; margin-bottom: 6px;'>
                                <span style='font-size: 0.8rem; color: #64748B;'>{pub_date}</span>
                                {badge_html}
                            </div>
                            <h6 style='margin: 0; color: #F8FAFC;'><a href='{link}' target='_blank' style='color:#3B82F6; text-decoration:none;'>{title}</a></h6>
                            <p style='font-size: 0.85rem; color: #94A3B8; margin-top: 6px; margin-bottom: 0;'>{summary[:200]}...</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info(f"No recent news articles found for {news_ticker}.")
        except Exception as e:
            st.error(f"Error fetching news for {news_ticker}: {e}")

# ----------------- SIDEBAR AUTO-CLOSE ON OUTSIDE CLICK -----------------
st.html("""
<script>
    const doc = window.parent.document;
    
    function closeSidebar() {
        const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
        if (sidebar) {
            // Check if the sidebar is collapsed
            const isCollapsed = sidebar.getAttribute('data-collapsed') === 'true' || 
                                sidebar.offsetWidth === 0 || 
                                sidebar.clientWidth === 0;
            if (!isCollapsed) {
                // Find the close button and specifically avoid clicking the expand button
                let collapseBtn = sidebar.querySelector('button[data-testid="stSidebarCollapseButton"]') || 
                                  sidebar.querySelector('button[aria-label="Collapse sidebar"]') ||
                                  sidebar.querySelector('button[aria-label="Close sidebar"]');
                
                if (collapseBtn) {
                    const label = (collapseBtn.getAttribute('aria-label') || '').toLowerCase();
                    // Prevent triggering if it has an "expand" or "open" action
                    if (!label.includes('expand') && !label.includes('open')) {
                        collapseBtn.click();
                    }
                }
            }
        }
    }
    
    const mainContent = doc.querySelector('[data-testid="stMain"]') || doc.querySelector('.main');
    if (mainContent) {
        if (!mainContent.dataset.sidebarListenerAttached) {
            mainContent.addEventListener('click', function(e) {
                // Safeguard 1: Check if click is on the sidebar expand/toggle button to prevent instant re-closing
                if (e.target.closest('[data-testid="collapsedSidebarTab"]') || 
                    e.target.closest('button[aria-label="Expand sidebar"]') ||
                    e.target.closest('.stSidebarCollapseButton') ||
                    e.target.closest('[data-testid="stSidebarCollapseButton"]')) {
                    return;
                }
                
                // Safeguard 2: Ignore clicks on dropdown menus, popovers, or listboxes (which render at body level in portals)
                if (e.target.closest('[data-baseweb="popover"]') || 
                    e.target.closest('[data-baseweb="menu"]') || 
                    e.target.closest('[role="listbox"]') ||
                    e.target.closest('.virtual-listbox') ||
                    e.target.closest('[id^="bui-"]') || 
                    e.target.closest('.stSelectbox')) {
                    return;
                }
                
                const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
                if (sidebar && !sidebar.contains(e.target)) {
                    closeSidebar();
                }
            }, true); // Use capture phase to intercept early
            mainContent.dataset.sidebarListenerAttached = 'true';
        }
    }
</script>
""", height=0, width=0)
