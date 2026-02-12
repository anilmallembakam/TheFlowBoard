"""
TheFlowBoard - Configuration
Centralized settings for thresholds, API credentials, and display options.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Load .env file from project root if it exists
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass


@dataclass
class APIConfig:
    """Schwab API configuration."""
    client_id: str = ""
    client_secret: str = ""
    redirect_uri: str = "https://127.0.0.1:8182/callback"
    token_url: str = "https://api.schwabapi.com/v1/oauth/token"
    auth_url: str = "https://api.schwabapi.com/v1/oauth/authorize"
    base_url: str = "https://api.schwabapi.com/marketdata/v1"
    rate_limit: int = 120  # requests per minute


@dataclass
class IBKRConfig:
    """Interactive Brokers TWS configuration."""
    host: str = "127.0.0.1"
    port: int = 7496          # TWS live=7496, paper=7497
    client_id: int = 10       # unique client ID for this connection (avoid 0/1 which are commonly used)
    timeout: int = 30         # seconds to wait for data
    readonly: bool = True     # read-only mode (no order placement)


@dataclass
class ThresholdConfig:
    """Flow detection thresholds."""
    large_volume: int = 100          # contracts
    large_premium: float = 50_000.0  # dollars
    unusual_volume_ratio: float = 3.0  # volume / OI ratio
    block_trade_min: int = 50        # minimum for block trade


@dataclass
class DisplayConfig:
    """UI display settings."""
    strikes_above_atm: int = 20
    strikes_below_atm: int = 20
    max_expirations: int = 6
    max_dte: int = 30
    refresh_interval: int = 30       # seconds
    snapshot_retention_hours: int = 24
    top_flow_count: int = 20


@dataclass
class ColorConfig:
    """Color scheme for the dashboard."""
    # Background / theme
    bg_primary: str = "#0e1117"
    bg_secondary: str = "#1a1a2e"
    bg_card: str = "#16213e"
    text_primary: str = "#e0e0e0"
    text_secondary: str = "#a0a0a0"

    # Call colors (greens)
    call_strong: str = "#00c853"
    call_medium: str = "#4caf50"
    call_light: str = "#81c784"
    call_bg: str = "#1b3a1b"

    # Put colors (reds)
    put_strong: str = "#ff1744"
    put_medium: str = "#f44336"
    put_light: str = "#ef9a9a"
    put_bg: str = "#3a1b1b"

    # ATM line
    atm_line: str = "#ffeb3b"
    atm_bg: str = "#3a3a1b"

    # Neutral
    neutral: str = "#424242"


SYMBOLS: List[str] = ["SPX", "SPY", "QQQ", "IWM"]
DEFAULT_SYMBOL: str = "SPX"

# Approximate spot prices for mock data
MOCK_PRICES = {
    "SPX": 6932.30,
    "SPY": 693.20,
    "QQQ": 530.50,
    "IWM": 225.80,
}

# Strike intervals per symbol
STRIKE_INTERVALS = {
    "SPX": 5.0,
    "SPY": 1.0,
    "QQQ": 1.0,
    "IWM": 1.0,
}

SNAPSHOT_DIR = "snapshots"

# Data source options
DATA_SOURCES = ["Mock (Demo)", "IBKR (TWS)", "Schwab API"]
DEFAULT_DATA_SOURCE = "Mock (Demo)"

# IBKR contract mapping — IBKR uses specific contract definitions
IBKR_CONTRACTS = {
    "SPX": {"symbol": "SPX", "exchange": "CBOE", "currency": "USD", "secType": "IND"},
    "SPY": {"symbol": "SPY", "exchange": "SMART", "currency": "USD", "secType": "STK"},
    "QQQ": {"symbol": "QQQ", "exchange": "SMART", "currency": "USD", "secType": "STK"},
    "IWM": {"symbol": "IWM", "exchange": "SMART", "currency": "USD", "secType": "STK"},
}


def load_api_config() -> APIConfig:
    """Load Schwab API config from environment variables."""
    return APIConfig(
        client_id=os.getenv("SCHWAB_CLIENT_ID", ""),
        client_secret=os.getenv("SCHWAB_CLIENT_SECRET", ""),
        redirect_uri=os.getenv("SCHWAB_REDIRECT_URI", "https://127.0.0.1:8182/callback"),
    )


def load_ibkr_config() -> IBKRConfig:
    """Load IBKR config from environment variables."""
    return IBKRConfig(
        host=os.getenv("IBKR_HOST", "127.0.0.1"),
        port=int(os.getenv("IBKR_PORT", "7496")),
        client_id=int(os.getenv("IBKR_CLIENT_ID", "10")),
    )
