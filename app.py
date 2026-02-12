"""
TheFlowBoard - Main Streamlit Dashboard
HTML table-based options flow heatmap matching the reference screenshot.
"""
import time
import html as html_mod
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import streamlit as st
import pandas as pd

from config import (
    SYMBOLS, DEFAULT_SYMBOL, DisplayConfig, ThresholdConfig, ColorConfig,
    MOCK_PRICES, DATA_SOURCES, DEFAULT_DATA_SOURCE,
)
from schwab_api import create_api_client, MockSchwabAPI, SchwabAPIClient
from utils.data_processor import DataProcessor
from utils.flow_detector import FlowDetector

import asyncio

# ib_insync (via eventkit) requires an asyncio event loop at import time.
# Streamlit's ScriptRunner thread may not have one, so create it first.
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

try:
    from ibkr_api import IBKRClient
    HAS_IBKR = True
except ImportError:
    HAS_IBKR = False


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TheFlowBoard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .stSidebar { background-color: #1a1a2e; }
    [data-testid="stMetricValue"] { font-size: 1.1rem; }
    [data-testid="stMetricDelta"] { font-size: 0.8rem; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }

    /* Force the HTML block to be full-width */
    .flow-table-wrap {
        width: 100%;
        overflow-x: auto;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 12px;
    }
    .flow-table {
        border-collapse: collapse;
        width: 100%;
        white-space: nowrap;
    }
    .flow-table th {
        background: #1a1a2e;
        color: #b0b0b0;
        padding: 4px 8px;
        text-align: center;
        border-bottom: 2px solid #333;
        position: sticky;
        top: 0;
        z-index: 2;
        font-weight: 600;
        font-size: 11px;
    }
    .flow-table td {
        padding: 2px 8px;
        text-align: center;
        border-bottom: 1px solid #1a1a2e;
        color: #e0e0e0;
        font-size: 12px;
        height: 22px;
    }
    .flow-table .strike-col {
        text-align: right;
        font-weight: 600;
        color: #ccc;
        background: #0e1117;
        position: sticky;
        left: 0;
        z-index: 1;
        padding-right: 12px;
    }
    .flow-table .atm-row td {
        border-top: 2px solid #ff00ff !important;
        border-bottom: 2px solid #ff00ff !important;
    }
    .flow-table .atm-row .strike-col {
        color: #ffeb3b;
        font-weight: 800;
    }
    .flow-table .dte-col {
        font-weight: 700;
    }

    /* Bar cell for net contracts / premium */
    .bar-cell {
        position: relative;
        width: 100%;
        height: 18px;
        display: flex;
        align-items: center;
    }
    .bar-pos {
        height: 14px;
        background: #00c853;
        border-radius: 0 2px 2px 0;
        min-width: 1px;
    }
    .bar-neg {
        height: 14px;
        background: #ff1744;
        border-radius: 2px 0 0 2px;
        min-width: 1px;
        margin-left: auto;
    }
    .bar-label {
        font-size: 10px;
        color: #ccc;
        padding: 0 3px;
        white-space: nowrap;
    }

    /* Snapshot columns - thin colored bars */
    .snap-bar {
        display: inline-block;
        width: 4px;
        border-radius: 1px;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ────────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "api_client": None,
        "processor": DataProcessor(),
        "flow_detector": FlowDetector(),
        "last_update": None,
        "processed_data": None,
        "quote_data": None,
        "auto_refresh": True,
        "symbol": DEFAULT_SYMBOL,
        "data_source": DEFAULT_DATA_SOURCE,
        "display_mode": "Volume",  # "Volume", "GEX", "OI"
        "display_config": DisplayConfig(),
        "threshold_config": ThresholdConfig(),
        "color_config": ColorConfig(),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    if st.session_state.api_client is None:
        st.session_state.api_client = create_api_client(st.session_state.data_source)


init_session_state()


# ── Color helpers ────────────────────────────────────────────────────────────
def get_cell_bg(call_pct: float, total_volume: int, is_atm: bool = False) -> str:
    """
    Cell background color based on call_volume / (call_volume + put_volume).
    Colors are muted/dark to match the reference screenshot.

    call_pct > 0.5 → green (calls dominant), brighter = stronger
    call_pct < 0.5 → red   (puts dominant),  brighter = stronger
    call_pct = 0.5 → neutral dark
    """
    if total_volume == 0:
        if is_atm:
            return "#2e2e16"
        return "#161616"

    # How far from 50/50 — ranges 0.0 (even) to 1.0 (100% one side)
    strength = abs(call_pct - 0.5) * 2.0

    if is_atm:
        # ATM row: yellow-tinted base, shift green/red from there
        if call_pct > 0.5:
            # yellow-green: base (46,46,22) → brightest (58,80,22)
            r = int(46 + 12 * strength)
            g = int(46 + 34 * strength)
            b = int(22)
            return f"rgb({r},{g},{b})"
        elif call_pct < 0.5:
            # yellow-red: base (46,46,22) → brightest (80,36,22)
            r = int(46 + 34 * strength)
            g = int(46 - 10 * strength)
            b = int(22)
            return f"rgb({r},{g},{b})"
        return "#2e2e16"

    if call_pct > 0.5:
        # Green shades: dark olive → medium green
        # Weakest (just above 50%): rgb(24, 34, 20)  — barely tinted
        # Strongest (100% calls):    rgb(38, 72, 30)  — clearly green but still muted
        r = int(24 + 14 * strength)
        g = int(34 + 38 * strength)
        b = int(20 + 10 * strength)
        return f"rgb({r},{g},{b})"
    elif call_pct < 0.5:
        # Red shades: dark maroon → medium red
        # Weakest (just below 50%): rgb(34, 20, 22)  — barely tinted
        # Strongest (100% puts):     rgb(72, 26, 28)  — clearly red but still muted
        r = int(34 + 38 * strength)
        g = int(20 - 0 * strength)
        b = int(22 + 6 * strength)
        return f"rgb({r},{g},{b})"

    return "#1e1e1e"


def get_dte_cell_bg(net_vol: int, total_vol: int) -> str:
    """0-30 DTE aggregate column color. Same muted palette."""
    if total_vol == 0:
        return "#161616"
    call_pct = (net_vol / total_vol + 1) / 2  # map -1..1 → 0..1
    strength = abs(call_pct - 0.5) * 2.0
    if call_pct > 0.5:
        r = int(24 + 14 * strength)
        g = int(34 + 38 * strength)
        b = int(20 + 10 * strength)
        return f"rgb({r},{g},{b})"
    elif call_pct < 0.5:
        r = int(34 + 38 * strength)
        g = int(20)
        b = int(22 + 6 * strength)
        return f"rgb({r},{g},{b})"
    return "#1e1e1e"


def format_vol(val: int) -> str:
    """Compact volume formatting like the screenshot."""
    if val == 0:
        return "-"
    if abs(val) >= 1000:
        return f"{val/1000:.1f}k"
    return str(val)


def format_premium_short(val: float) -> str:
    if abs(val) >= 1_000_000:
        return f"{val/1_000_000:.1f}M"
    if abs(val) >= 1_000:
        return f"{val/1_000:.0f}K"
    return f"{val:.0f}"


def format_gex(val: float) -> str:
    """Format GEX value for compact display (in dollars)."""
    if val == 0:
        return "-"
    abs_val = abs(val)
    if abs_val >= 1_000_000_000:
        return f"{val/1_000_000_000:.1f}B"
    if abs_val >= 1_000_000:
        return f"{val/1_000_000:.1f}M"
    if abs_val >= 1_000:
        return f"{val/1_000:.0f}K"
    return f"{val:.0f}"


def get_gex_cell_bg(gex_val: float, max_gex: float, is_atm: bool = False) -> str:
    """
    Cell background color based on GEX value.
    Positive GEX → green (dealer long gamma, market pinned)
    Negative GEX → red (dealer short gamma, volatile)
    """
    if max_gex == 0 or gex_val == 0:
        if is_atm:
            return "#2e2e16"
        return "#161616"

    strength = min(abs(gex_val) / max_gex, 1.0)

    if is_atm:
        if gex_val > 0:
            r = int(46 + 12 * strength)
            g = int(46 + 34 * strength)
            b = int(22)
            return f"rgb({r},{g},{b})"
        elif gex_val < 0:
            r = int(46 + 34 * strength)
            g = int(46 - 10 * strength)
            b = int(22)
            return f"rgb({r},{g},{b})"
        return "#2e2e16"

    if gex_val > 0:
        r = int(24 + 14 * strength)
        g = int(34 + 38 * strength)
        b = int(20 + 10 * strength)
        return f"rgb({r},{g},{b})"
    elif gex_val < 0:
        r = int(34 + 38 * strength)
        g = int(20)
        b = int(22 + 6 * strength)
        return f"rgb({r},{g},{b})"

    return "#1e1e1e"


# ── Data fetch ───────────────────────────────────────────────────────────────
def fetch_data():
    client = st.session_state.api_client
    symbol = st.session_state.symbol
    cfg = st.session_state.display_config

    # Identify actual data source for debugging
    source_name = type(client).__name__
    try:
        quote = client.get_quote(symbol)
        chain = client.get_option_chain(
            symbol,
            strike_count=cfg.strikes_above_atm + cfg.strikes_below_atm,
            days_to_expiration=cfg.max_dte,
        )

        # Debug: log what we got
        n_calls = sum(len(v) for v in chain.get("callExpDateMap", {}).values())
        n_puts = sum(len(v) for v in chain.get("putExpDateMap", {}).values())
        spot = chain.get("underlying", {}).get("last", 0)
        st.session_state._debug_info = (
            f"Source: {source_name} | Spot: {spot} | "
            f"Call strikes: {n_calls} | Put strikes: {n_puts}"
        )

        processor = st.session_state.processor
        processor.config = cfg
        processed = processor.process_chain(chain)
        st.session_state.quote_data = quote
        st.session_state.processed_data = processed
        st.session_state.last_update = datetime.now()
        processor.save_snapshot(processed, symbol)
        processor.cleanup_old_snapshots(cfg.snapshot_retention_hours)
    except Exception as e:
        import traceback
        st.error(f"Data fetch error ({source_name}): {e}")
        st.code(traceback.format_exc(), language="text")


# ── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## TheFlowBoard")

        # ── Data Source selector ──
        st.markdown("**Data Source**")
        current_idx = DATA_SOURCES.index(st.session_state.data_source) \
            if st.session_state.data_source in DATA_SOURCES else 0
        data_source = st.selectbox(
            "source", DATA_SOURCES,
            index=current_idx,
            key="source_select",
            label_visibility="collapsed",
        )
        if data_source != st.session_state.data_source:
            # Disconnect old IBKR client if switching away
            old_client = st.session_state.api_client
            if HAS_IBKR and isinstance(old_client, IBKRClient):
                try:
                    old_client.disconnect()
                except Exception:
                    pass

            st.session_state.data_source = data_source
            st.session_state.processed_data = None
            st.session_state.quote_data = None

            # Create new client
            try:
                st.session_state.api_client = create_api_client(data_source)
            except Exception as e:
                st.error(f"Connection failed: {e}")
                st.session_state.data_source = DEFAULT_DATA_SOURCE
                st.session_state.api_client = create_api_client(DEFAULT_DATA_SOURCE)
            st.rerun()

        # Connection status indicator
        client = st.session_state.api_client
        if isinstance(client, MockSchwabAPI):
            st.markdown(
                "<div style='background:#1b3a1b;color:#81c784;padding:6px 12px;"
                "border-radius:6px;font-size:0.8rem'>Demo Mode (Mock Data)</div>",
                unsafe_allow_html=True,
            )
        elif HAS_IBKR and isinstance(client, IBKRClient):
            if client.is_connected:
                port = client.config.port
                mode = "Paper" if port == 7497 else "Live"
                st.markdown(
                    f"<div style='background:#1b3a1b;color:#81c784;padding:6px 12px;"
                    f"border-radius:6px;font-size:0.8rem'>"
                    f"IBKR Connected ({mode} : {port})</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='background:#3a1b1b;color:#e57373;padding:6px 12px;"
                    "border-radius:6px;font-size:0.8rem'>"
                    "IBKR Disconnected</div>",
                    unsafe_allow_html=True,
                )
        elif isinstance(client, SchwabAPIClient):
            if client.is_authenticated:
                st.markdown(
                    "<div style='background:#1b3a1b;color:#81c784;padding:6px 12px;"
                    "border-radius:6px;font-size:0.8rem'>Schwab API Connected</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='background:#3a2a1b;color:#ffb74d;padding:6px 12px;"
                    "border-radius:6px;font-size:0.8rem'>Schwab — Login Required</div>",
                    unsafe_allow_html=True,
                )
                # Show OAuth login flow
                if client.is_configured:
                    auth_url = client.get_auth_url()
                    st.markdown(
                        f"**Step 1:** [Click here to login with Schwab]({auth_url})",
                    )
                    st.markdown(
                        "**Step 2:** After login, Schwab redirects to a URL. "
                        "Paste the **full redirect URL** below:"
                    )
                    redirect_url = st.text_input(
                        "Redirect URL", key="schwab_redirect_url",
                        placeholder="https://127.0.0.1:8182/callback?code=...",
                    )
                    if redirect_url and "code=" in redirect_url:
                        from urllib.parse import urlparse, parse_qs
                        parsed = urlparse(redirect_url)
                        code = parse_qs(parsed.query).get("code", [None])[0]
                        if code:
                            with st.spinner("Exchanging auth code..."):
                                if client.exchange_code(code):
                                    st.success("Authenticated!")
                                    st.rerun()
                                else:
                                    st.error("Token exchange failed. Check credentials.")
                else:
                    st.warning(
                        "Set SCHWAB_CLIENT_ID and SCHWAB_CLIENT_SECRET "
                        "environment variables or in a .env file."
                    )
        else:
            st.markdown(
                "<div style='background:#1b2a3a;color:#90caf9;padding:6px 12px;"
                "border-radius:6px;font-size:0.8rem'>Schwab API</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Symbol selector ──
        symbol = st.selectbox(
            "Symbol", SYMBOLS,
            index=SYMBOLS.index(st.session_state.symbol),
            key="symbol_select",
        )
        if symbol != st.session_state.symbol:
            st.session_state.symbol = symbol
            st.session_state.processed_data = None
            st.rerun()

        # Price display
        quote = st.session_state.quote_data
        if quote:
            price = quote.get("lastPrice", 0)
            change = quote.get("netChange", 0)
            change_pct = quote.get("netPercentChange", 0)
            st.markdown(
                f"<div style='font-size:1.4rem;font-weight:800;color:#e0e0e0'>"
                f"{symbol} &nbsp; ${price:,.2f}</div>"
                f"<div style='font-size:0.8rem;color:#888'>LIVE LAST</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Display Mode selector ──
        st.markdown("**Display Mode**")
        display_modes = ["Volume", "GEX", "OI"]
        current_mode_idx = display_modes.index(st.session_state.display_mode) \
            if st.session_state.display_mode in display_modes else 0
        display_mode = st.selectbox(
            "mode", display_modes,
            index=current_mode_idx,
            key="mode_select",
            label_visibility="collapsed",
        )
        st.session_state.display_mode = display_mode

        st.markdown("---")

        # ── Strikes sliders ──
        st.markdown("**Strikes Above ATM**")
        strikes_above = st.slider(
            "above", 5, 40,
            st.session_state.display_config.strikes_above_atm,
            key="s_above", label_visibility="collapsed",
        )
        st.markdown("**Strikes Below ATM**")
        strikes_below = st.slider(
            "below", 5, 40,
            st.session_state.display_config.strikes_below_atm,
            key="s_below", label_visibility="collapsed",
        )
        st.session_state.display_config.strikes_above_atm = strikes_above
        st.session_state.display_config.strikes_below_atm = strikes_below

        # Info
        if st.session_state.processed_data:
            n_strikes = len(st.session_state.processed_data.get("strikes", []))
            max_dte = st.session_state.display_config.max_dte
            st.markdown(
                f"<div style='background:#1b3a1b;color:#81c784;padding:8px 12px;"
                f"border-radius:6px;font-size:0.85rem'>"
                f"Total: {n_strikes} strikes | Up to {max_dte} days</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Refresh controls ──
        auto_refresh = st.checkbox(
            "Auto-refresh every 30s",
            value=st.session_state.auto_refresh,
            key="auto_ref",
        )
        st.session_state.auto_refresh = auto_refresh

        if st.button("Manual Refresh Now", use_container_width=True):
            fetch_data()
            st.rerun()

        if st.session_state.last_update:
            st.caption(f"Updated: {st.session_state.last_update.strftime('%H:%M:%S')}")


# ── Build the HTML heatmap table ─────────────────────────────────────────────
def build_heatmap_html(processed_data: dict) -> str:
    """Build the full HTML table matching the screenshot layout."""
    strikes = processed_data["strikes"]
    expirations = processed_data["expirations"]
    grid = processed_data["grid"]
    atm_strike = processed_data["atm_strike"]
    spot = processed_data.get("spot_price", 0)
    contracts = processed_data.get("contracts", [])
    display_mode = st.session_state.get("display_mode", "GEX")

    if not strikes or not expirations:
        return "<p style='color:#888'>No data to display.</p>"

    processor = st.session_state.processor
    detector = st.session_state.flow_detector

    # Compute aggregates per strike across all expirations (0-30 DTE column)
    dte_agg: Dict[float, Dict[str, Any]] = {}
    for c in contracts:
        s = c["strike"]
        if s not in dte_agg:
            dte_agg[s] = {
                "call_vol": 0, "put_vol": 0,
                "call_oi": 0, "put_oi": 0,
                "call_gamma_oi": 0.0, "put_gamma_oi": 0.0,
            }
        if c["type"] == "CALL":
            dte_agg[s]["call_vol"] += c["volume"]
            dte_agg[s]["call_oi"] += c["openInterest"]
            dte_agg[s]["call_gamma_oi"] += c["gamma"] * c["openInterest"]
        else:
            dte_agg[s]["put_vol"] += c["volume"]
            dte_agg[s]["put_oi"] += c["openInterest"]
            dte_agg[s]["put_gamma_oi"] += c["gamma"] * c["openInterest"]

    # Compute per-strike aggregates
    net_contracts = processor.compute_net_contracts(processed_data)
    net_premium = processor.compute_net_premium(processed_data)
    gex_per_strike = processor.compute_gex(processed_data)

    # Find max values for bar scaling
    max_net_c = max((abs(v) for v in net_contracts.values()), default=1)
    max_net_p = max((abs(v) for v in net_premium.values()), default=1)
    max_gex = max((abs(v) for v in gex_per_strike.values()), default=1)

    # Max GEX per cell for color scaling
    max_cell_gex = 1.0
    if display_mode == "GEX":
        cell_gex_values = [abs(cell.get("gex", 0)) for cell in grid.values() if cell.get("gex", 0) != 0]
        max_cell_gex = max(cell_gex_values) if cell_gex_values else 1.0

    # Format expiration headers
    exp_headers = []
    for exp in expirations:
        try:
            dt = datetime.strptime(exp, "%Y-%m-%d")
            exp_headers.append(dt.strftime("%Y-%m-%d"))
        except ValueError:
            exp_headers.append(exp)

    # Determine strike format — integer for SPX-like, 2 decimal for stocks
    symbol = st.session_state.symbol
    use_int_strikes = all(s == int(s) for s in strikes)

    # Mode-specific labels
    if display_mode == "Volume":
        dte_label = "0-30 DTE"
        bar_label = "Net Contracts"
    elif display_mode == "GEX":
        dte_label = "0-30 DTE GEX"
        bar_label = "Net GEX"
    else:  # OI
        dte_label = "0-30 DTE OI"
        bar_label = "Net OI"

    # ── Header row ───────────────────────────────────────────────────────
    header_html = "<tr>"
    header_html += "<th style='text-align:right;min-width:80px'>Strike</th>"
    for eh in exp_headers:
        header_html += f"<th style='min-width:75px'>{eh}</th>"
    header_html += f"<th style='min-width:70px;background:#1a2a1a;color:#ffeb3b'>{dte_label}</th>"
    header_html += (
        f"<th style='min-width:100px'>{bar_label}</th>"
        "<th style='min-width:30px'>Live Flow</th>"
        "<th style='min-width:80px'>Net Premium<br>"
        "<span style='font-size:9px;color:#888'>60m Ago</span></th>"
        "<th style='min-width:80px'>Net Premium<br>"
        "<span style='font-size:9px;color:#888'>30m Ago</span></th>"
        "<th style='min-width:80px'>Net Premium<br>"
        "<span style='font-size:9px;color:#888'>15m Ago</span></th>"
    )
    header_html += "</tr>"

    # ── Data rows (highest strike on top) ────────────────────────────────
    rows_html = []
    strikes_desc = list(reversed(strikes))

    for strike in strikes_desc:
        is_atm = abs(strike - atm_strike) < 0.01
        row_class = ' class="atm-row"' if is_atm else ""

        row = f"<tr{row_class}>"

        # Strike column — integer format for clean strikes
        if use_int_strikes:
            strike_fmt = f"{int(strike):,}"
        else:
            strike_fmt = f"{strike:,.2f}"

        if is_atm:
            strike_color = "#ffeb3b"
            strike_bg = "#3a3a1b"
        else:
            strike_color = "#ccc"
            strike_bg = "#0e1117"
        row += (
            f'<td class="strike-col" style="color:{strike_color};'
            f'background:{strike_bg}">{strike_fmt}</td>'
        )

        # Expiration columns — value and color depend on display_mode
        for exp in expirations:
            cell = grid.get((strike, exp), {})

            if display_mode == "GEX":
                cell_gex = cell.get("gex", 0)
                bg = get_gex_cell_bg(cell_gex, max_cell_gex, is_atm)
                val = format_gex(cell_gex)
            elif display_mode == "OI":
                call_oi = cell.get("call_oi", 0)
                put_oi = cell.get("put_oi", 0)
                total_oi = call_oi + put_oi
                oi_pct = call_oi / total_oi if total_oi > 0 else 0.5
                bg = get_cell_bg(oi_pct, total_oi, is_atm)
                val = format_vol(total_oi)  # show TOTAL OI, color shows direction
            else:  # Volume
                total = cell.get("total_volume", 0)
                call_pct = cell.get("call_pct", 0.5)
                bg = get_cell_bg(call_pct, total, is_atm)
                val = format_vol(total)  # show TOTAL volume, color shows direction

            txt_color = "#e0e0e0" if val != "-" else "#555"
            row += f'<td style="background:{bg};color:{txt_color}">{val}</td>'

        # 0-30 DTE aggregate column
        agg = dte_agg.get(strike, {
            "call_vol": 0, "put_vol": 0,
            "call_oi": 0, "put_oi": 0,
            "call_gamma_oi": 0.0, "put_gamma_oi": 0.0,
        })

        if display_mode == "GEX":
            dte_gex = (agg["call_gamma_oi"] - agg["put_gamma_oi"]) * spot * spot * 0.01 * 100
            dte_val = format_gex(dte_gex)
            dte_bg = get_gex_cell_bg(dte_gex, max_gex, is_atm)
            dte_txt_color = "#6fbf73" if dte_gex > 0 else "#e57373" if dte_gex < 0 else "#555"
        elif display_mode == "OI":
            dte_total_oi = agg["call_oi"] + agg["put_oi"]
            dte_call_pct = agg["call_oi"] / dte_total_oi if dte_total_oi > 0 else 0.5
            dte_val = format_vol(dte_total_oi)
            dte_bg = get_cell_bg(dte_call_pct, dte_total_oi, is_atm)
            dte_txt_color = "#e0e0e0" if dte_total_oi > 0 else "#555"
        else:  # Volume
            dte_total = agg["call_vol"] + agg["put_vol"]
            dte_call_pct = agg["call_vol"] / dte_total if dte_total > 0 else 0.5
            dte_bg = get_cell_bg(dte_call_pct, dte_total, is_atm)
            dte_val = format_vol(dte_total)
            dte_txt_color = "#e0e0e0" if dte_total > 0 else "#555"

        if is_atm:
            dte_bg = "#2e2e16"

        row += (
            f'<td class="dte-col" style="background:{dte_bg};'
            f'color:{dte_txt_color};font-weight:700">{dte_val}</td>'
        )

        # Bar column — GEX bar or Net Contracts bar based on mode
        if display_mode == "GEX":
            gex_val = gex_per_strike.get(strike, 0)
            bar_html = _make_bar_cell_float(gex_val, max_gex, format_gex)
        elif display_mode == "OI":
            oi_agg = dte_agg.get(strike, {"call_oi": 0, "put_oi": 0})
            net_oi = oi_agg["call_oi"] - oi_agg["put_oi"]
            max_oi = max((abs(a["call_oi"] - a["put_oi"]) for a in dte_agg.values()), default=1)
            bar_html = _make_bar_cell(net_oi, max_oi)
        else:
            nc = net_contracts.get(strike, 0)
            bar_html = _make_bar_cell(nc, max_net_c)
        row += f"<td style='background:#111;padding:2px 4px'>{bar_html}</td>"

        # Live Flow column
        flow_bar = _make_flow_indicator(strike, contracts, detector)
        row += f"<td style='background:#111;padding:2px'>{flow_bar}</td>"

        # Net Premium columns (60m, 30m, 15m) - show current as placeholder
        np_val = net_premium.get(strike, 0)
        for _ in range(3):
            prem_bar = _make_premium_bar(np_val, max_net_p)
            row += f"<td style='background:#111;padding:2px'>{prem_bar}</td>"

        row += "</tr>"
        rows_html.append(row)

    # Assemble table
    table = (
        '<div class="flow-table-wrap">'
        '<table class="flow-table">'
        f"<thead>{header_html}</thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table></div>"
    )
    return table


def _make_bar_cell(value: int, max_val: int) -> str:
    """Create an inline horizontal bar for net contracts."""
    if value == 0 or max_val == 0:
        return '<span style="color:#555;font-size:10px">0</span>'

    pct = min(abs(value) / max_val, 1.0) * 100
    label = format_vol(value)

    if value > 0:
        return (
            f'<div style="display:flex;align-items:center;height:18px">'
            f'<div style="width:{pct:.0f}%;height:14px;background:#4caf50;'
            f'border-radius:0 2px 2px 0;min-width:2px"></div>'
            f'<span class="bar-label" style="color:#6fbf73">{label}</span>'
            f'</div>'
        )
    else:
        return (
            f'<div style="display:flex;align-items:center;justify-content:flex-end;height:18px">'
            f'<span class="bar-label" style="color:#e57373">{label}</span>'
            f'<div style="width:{pct:.0f}%;height:14px;background:#c62828;'
            f'border-radius:2px 0 0 2px;min-width:2px"></div>'
            f'</div>'
        )


def _make_bar_cell_float(value: float, max_val: float, formatter=None) -> str:
    """Create an inline horizontal bar for float values (GEX, etc.)."""
    if value == 0 or max_val == 0:
        return '<span style="color:#555;font-size:10px">0</span>'

    pct = min(abs(value) / max_val, 1.0) * 100
    label = formatter(value) if formatter else f"{value:.0f}"

    if value > 0:
        return (
            f'<div style="display:flex;align-items:center;height:18px">'
            f'<div style="width:{pct:.0f}%;height:14px;background:#4caf50;'
            f'border-radius:0 2px 2px 0;min-width:2px"></div>'
            f'<span class="bar-label" style="color:#6fbf73">{label}</span>'
            f'</div>'
        )
    else:
        return (
            f'<div style="display:flex;align-items:center;justify-content:flex-end;height:18px">'
            f'<span class="bar-label" style="color:#e57373">{label}</span>'
            f'<div style="width:{pct:.0f}%;height:14px;background:#c62828;'
            f'border-radius:2px 0 0 2px;min-width:2px"></div>'
            f'</div>'
        )


def _make_flow_indicator(
    strike: float, contracts: list, detector: FlowDetector
) -> str:
    """Create a thin flow indicator bar for a strike."""
    # Find matching contracts for this strike
    strike_contracts = [c for c in contracts if c["strike"] == strike]
    if not strike_contracts:
        return ""

    total_vol = sum(c["volume"] for c in strike_contracts)
    call_vol = sum(c["volume"] for c in strike_contracts if c["type"] == "CALL")

    if total_vol == 0:
        return ""

    call_pct = call_vol / total_vol
    height = min(18, max(2, int(total_vol / 50)))

    if call_pct > 0.55:
        color = "#4caf50"
    elif call_pct < 0.45:
        color = "#c62828"
    else:
        color = "#bfaa30"

    return (
        f'<div style="display:flex;justify-content:center;align-items:center;height:18px">'
        f'<div style="width:6px;height:{height}px;background:{color};'
        f'border-radius:1px"></div></div>'
    )


def _make_premium_bar(value: float, max_val: float) -> str:
    """Create thin premium bar for snapshot columns."""
    if value == 0 or max_val == 0:
        return ""

    pct = min(abs(value) / max_val, 1.0)
    height = max(2, int(18 * pct))
    color = "#4caf50" if value > 0 else "#c62828"

    # Slightly brighter for large values
    if pct > 0.5:
        if value > 0:
            color = "#5cb860"
        else:
            color = "#d32f2f"

    return (
        f'<div style="display:flex;justify-content:center;align-items:center;height:18px">'
        f'<div style="width:4px;height:{height}px;background:{color};'
        f'border-radius:1px"></div></div>'
    )


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    if st.session_state.processed_data is None:
        fetch_data()

    render_sidebar()

    # Timestamp header + data source debug info
    now = datetime.now()
    client = st.session_state.api_client
    source_type = type(client).__name__
    debug_info = st.session_state.get("_debug_info", "")
    st.markdown(
        f"<div style='color:#888;font-family:monospace;font-size:0.85rem;"
        f"padding:4px 0'>{now.strftime('%a %b %d, %H:%M:%S')} &nbsp;|&nbsp; "
        f"<span style='color:#ffeb3b'>{source_type}</span>"
        f" &nbsp;|&nbsp; {debug_info}</div>",
        unsafe_allow_html=True,
    )

    processed = st.session_state.processed_data
    if processed is None or not processed.get("strikes"):
        st.warning("No data available. Click 'Manual Refresh Now' in the sidebar.")
        return

    # Render the HTML heatmap table
    table_html = build_heatmap_html(processed)
    st.markdown(table_html, unsafe_allow_html=True)

    # Auto-refresh
    if st.session_state.auto_refresh:
        time.sleep(st.session_state.display_config.refresh_interval)
        fetch_data()
        st.rerun()


if __name__ == "__main__":
    main()
