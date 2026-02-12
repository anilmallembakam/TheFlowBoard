"""
TheFlowBoard - Schwab API Client
Provides both real Schwab API integration and a MockSchwabAPI for demo mode.

Schwab OAuth2 flow:
  1) User visits auth URL → logs in → Schwab redirects to callback with ?code=...
  2) App exchanges code for access_token (30 min) + refresh_token (7 days)
  3) Access token used as Bearer header for all API calls
  4) When access token expires, refresh it using the refresh token
"""
import base64
import json
import os
import random
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse, parse_qs

import requests

from config import (
    APIConfig, MOCK_PRICES, STRIKE_INTERVALS,
    load_api_config,
)

# Where we persist tokens so user doesn't have to re-auth every run
TOKEN_FILE = Path(__file__).parent / ".schwab_token.json"


class SchwabAPIClient:
    """Real Schwab API client with OAuth2 authentication."""

    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or load_api_config()
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expiry: float = 0
        self.session = requests.Session()
        self._request_times: List[float] = []
        # Try to load saved tokens
        self._load_tokens()

    @property
    def is_configured(self) -> bool:
        return bool(self.config.client_id and self.config.client_secret)

    @property
    def is_authenticated(self) -> bool:
        return bool(self.access_token)

    def get_auth_url(self) -> str:
        """Generate Schwab OAuth2 authorization URL."""
        return (
            f"{self.config.auth_url}"
            f"?response_type=code"
            f"&client_id={self.config.client_id}"
            f"&redirect_uri={self.config.redirect_uri}"
        )

    def _basic_auth_header(self) -> str:
        """Base64-encode client_id:client_secret for Authorization header."""
        credentials = f"{self.config.client_id}:{self.config.client_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"

    def exchange_code(self, auth_code: str) -> bool:
        """Exchange authorization code for access + refresh tokens."""
        headers = {
            "Authorization": self._basic_auth_header(),
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": self.config.redirect_uri,
        }
        try:
            resp = self.session.post(
                self.config.token_url, headers=headers, data=data, timeout=10
            )
            resp.raise_for_status()
            token_data = resp.json()
            self.access_token = token_data["access_token"]
            self.refresh_token = token_data.get("refresh_token")
            self.token_expiry = time.time() + token_data.get("expires_in", 1800)
            self._save_tokens()
            return True
        except Exception as e:
            print(f"Schwab token exchange failed: {e}")
            return False

    def refresh_access_token(self) -> bool:
        """Use refresh token to get a new access token (called when access token expires)."""
        if not self.refresh_token:
            return False
        headers = {
            "Authorization": self._basic_auth_header(),
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
        }
        try:
            resp = self.session.post(
                self.config.token_url, headers=headers, data=data, timeout=10
            )
            resp.raise_for_status()
            token_data = resp.json()
            self.access_token = token_data["access_token"]
            self.refresh_token = token_data.get("refresh_token", self.refresh_token)
            self.token_expiry = time.time() + token_data.get("expires_in", 1800)
            self._save_tokens()
            return True
        except Exception as e:
            print(f"Schwab token refresh failed: {e}")
            self.access_token = None
            return False

    def _ensure_token(self):
        """Make sure we have a valid access token, refreshing if needed."""
        if not self.access_token:
            raise RuntimeError(
                "Not authenticated. Complete the Schwab OAuth flow first."
            )
        # Refresh 60 seconds before expiry
        if time.time() > (self.token_expiry - 60):
            if not self.refresh_access_token():
                raise RuntimeError(
                    "Access token expired and refresh failed. Re-authenticate."
                )

    def _save_tokens(self):
        """Persist tokens to disk so user doesn't re-auth every restart."""
        data = {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_expiry": self.token_expiry,
            "saved_at": time.time(),
        }
        try:
            TOKEN_FILE.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _load_tokens(self):
        """Load previously saved tokens."""
        try:
            if TOKEN_FILE.exists():
                data = json.loads(TOKEN_FILE.read_text())
                self.access_token = data.get("access_token")
                self.refresh_token = data.get("refresh_token")
                self.token_expiry = data.get("token_expiry", 0)
        except Exception:
            pass

    def _check_rate_limit(self):
        """Enforce rate limiting."""
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 60]
        if len(self._request_times) >= self.config.rate_limit:
            raise RuntimeError("Rate limit exceeded. Wait before making more requests.")
        self._request_times.append(now)

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
        }

    # Schwab uses $-prefixed symbols for indices
    SCHWAB_SYMBOL_MAP = {
        "SPX": "$SPX",
        "DJX": "$DJX",
        "NDX": "$NDX",
        "RUT": "$RUT",
    }

    def _schwab_symbol(self, symbol: str) -> str:
        """Convert our symbol to Schwab's format (e.g. SPX → $SPX)."""
        return self.SCHWAB_SYMBOL_MAP.get(symbol, symbol)

    def _api_get(self, url: str, params: dict = None) -> dict:
        """Make an authenticated GET request, auto-refreshing token if needed."""
        self._ensure_token()
        self._check_rate_limit()
        resp = self.session.get(url, headers=self._get_headers(), params=params, timeout=15)
        # If 401, try refreshing token once
        if resp.status_code == 401 and self.refresh_access_token():
            resp = self.session.get(url, headers=self._get_headers(), params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get current quote for a symbol."""
        schwab_sym = self._schwab_symbol(symbol)
        # Use the /quotes endpoint with query param (supports special chars like $SPX)
        url = f"{self.config.base_url}/quotes"
        data = self._api_get(url, params={"symbols": schwab_sym, "indicative": "false"})

        # Debug: log raw response structure
        print(f"[Schwab quote] keys={list(data.keys()) if isinstance(data, dict) else type(data)}")

        # Schwab wraps the response — extract the inner quote
        # Response format: { "$SPX": { "quote": {...}, "reference": {...} } }
        q = {}
        if isinstance(data, dict):
            # Try schwab symbol key first, then our symbol, then first key
            inner = data.get(schwab_sym) or data.get(symbol)
            if inner is None:
                # Try first key in response
                for key in data:
                    inner = data[key]
                    break
            if isinstance(inner, dict):
                q = inner.get("quote", inner)
            elif inner is not None:
                q = inner

        if not isinstance(q, dict):
            print(f"[Schwab quote] WARNING: unexpected quote format: {type(q)} = {q}")
            q = {}

        last = q.get("lastPrice", q.get("mark", 0))
        return {
            "symbol": symbol,
            "lastPrice": last,
            "netChange": q.get("netChange", 0),
            "netPercentChange": q.get("netPercentChangeInDouble", 0),
            "high": q.get("highPrice", last),
            "low": q.get("lowPrice", last),
            "volume": q.get("totalVolume", 0),
            "timestamp": datetime.now().isoformat(),
        }

    def get_option_chain(
        self,
        symbol: str,
        strike_count: int = 40,
        days_to_expiration: int = 30,
    ) -> Dict[str, Any]:
        """Get option chain from Schwab API."""
        schwab_sym = self._schwab_symbol(symbol)
        url = f"{self.config.base_url}/chains"
        params = {
            "symbol": schwab_sym,
            "contractType": "ALL",
            "strikeCount": strike_count,
            "range": "ALL",
            "toDate": (datetime.now() + timedelta(days=days_to_expiration)).strftime("%Y-%m-%d"),
        }
        data = self._api_get(url, params)

        # Debug: log what we got
        print(f"[Schwab chain] keys={list(data.keys()) if isinstance(data, dict) else type(data)}")

        # Normalize: ensure expected keys exist
        if isinstance(data, dict):
            # Schwab returns "underlying" which might be None for indices
            if data.get("underlying") is None:
                spot = data.get("underlyingPrice", 0)
                data["underlying"] = {
                    "symbol": symbol,
                    "last": spot,
                    "mark": spot,
                }
        return data


class MockSchwabAPI:
    """Mock API that generates realistic simulated options data for demo mode."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._last_gen: float = 0
        self._rng = random.Random(42)

    @property
    def is_configured(self) -> bool:
        return True

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        base = MOCK_PRICES.get(symbol, 500.0)
        jitter = base * random.uniform(-0.003, 0.003)
        price = round(base + jitter, 2)
        return {
            "symbol": symbol,
            "lastPrice": price,
            "netChange": round(jitter, 2),
            "netPercentChange": round(jitter / base * 100, 2),
            "high": round(price + base * 0.005, 2),
            "low": round(price - base * 0.005, 2),
            "volume": random.randint(500_000, 5_000_000),
            "timestamp": datetime.now().isoformat(),
        }

    def get_option_chain(
        self,
        symbol: str,
        strike_count: int = 40,
        days_to_expiration: int = 30,
    ) -> Dict[str, Any]:
        """Generate realistic mock option chain data."""
        now = time.time()
        cache_key = f"{symbol}_{strike_count}_{days_to_expiration}"

        # Regenerate every 25 seconds to simulate fresh data
        if cache_key in self._cache and (now - self._last_gen) < 25:
            return self._cache[cache_key]

        spot = MOCK_PRICES.get(symbol, 500.0)
        spot += spot * random.uniform(-0.002, 0.002)
        interval = STRIKE_INTERVALS.get(symbol, 5.0)

        # Generate expirations (next 5-6 business-ish days)
        expirations = self._generate_expirations(days_to_expiration)

        calls_by_exp: Dict[str, Dict[str, List[Dict]]] = {}
        puts_by_exp: Dict[str, Dict[str, List[Dict]]] = {}

        half = strike_count // 2
        # Round ATM to nearest clean strike interval, then generate range
        atm_strike = round(spot / interval) * interval
        strikes = [round(atm_strike - half * interval + i * interval, 2) for i in range(strike_count)]

        for exp_date in expirations:
            dte = (exp_date - datetime.now()).days
            if dte < 0:
                dte = 0
            exp_key = exp_date.strftime("%Y-%m-%d")
            calls_by_exp[exp_key] = {}
            puts_by_exp[exp_key] = {}

            for strike in strikes:
                moneyness = (strike - spot) / spot
                call_contract = self._make_contract(
                    symbol, strike, spot, dte, "CALL", moneyness
                )
                put_contract = self._make_contract(
                    symbol, strike, spot, dte, "PUT", -moneyness
                )
                strike_key = f"{strike:.1f}"
                calls_by_exp[exp_key][strike_key] = [call_contract]
                puts_by_exp[exp_key][strike_key] = [put_contract]

        chain = {
            "symbol": symbol,
            "status": "SUCCESS",
            "underlying": {
                "symbol": symbol,
                "last": round(spot, 2),
                "mark": round(spot, 2),
            },
            "callExpDateMap": calls_by_exp,
            "putExpDateMap": puts_by_exp,
        }
        self._cache[cache_key] = chain
        self._last_gen = now
        return chain

    def _generate_expirations(self, max_dte: int) -> List[datetime]:
        """Generate realistic expiration dates."""
        exps = []
        today = datetime.now()
        # Daily / weekly expirations for near-term
        for d in range(0, min(max_dte, 35)):
            candidate = today + timedelta(days=d)
            # SPX has daily expirations; others mostly weekly (Fridays)
            if d <= 7 or candidate.weekday() == 4:  # Friday
                exps.append(candidate)
            if len(exps) >= 6:
                break
        return exps

    def _make_contract(
        self,
        symbol: str,
        strike: float,
        spot: float,
        dte: int,
        option_type: str,
        moneyness: float,
    ) -> Dict[str, Any]:
        """Generate a single realistic option contract."""
        t = max(dte / 365.0, 1 / 365.0)
        iv = 0.15 + abs(moneyness) * 0.5 + random.uniform(-0.02, 0.02)

        # Simplified Black-Scholes-ish pricing
        intrinsic = max(0, spot - strike) if option_type == "CALL" else max(0, strike - spot)
        time_value = spot * iv * math.sqrt(t) * 0.4
        mid = max(0.05, intrinsic + time_value + random.uniform(-0.5, 0.5))
        spread = max(0.05, mid * random.uniform(0.02, 0.08))

        bid = round(max(0.01, mid - spread / 2), 2)
        ask = round(mid + spread / 2, 2)
        mid = round((bid + ask) / 2, 2)

        # Volume and OI - higher near ATM
        atm_factor = max(0.05, 1.0 - abs(moneyness) * 8)
        base_oi = int(random.uniform(50, 2000) * atm_factor)
        base_vol = int(random.uniform(0, base_oi * 0.6) * atm_factor)

        # Occasional large volume spikes
        if random.random() < 0.08:
            base_vol = int(base_vol * random.uniform(3, 15))

        # Delta approximation
        if option_type == "CALL":
            delta = max(0.01, min(0.99, 0.5 - moneyness * 3))
        else:
            delta = -max(0.01, min(0.99, 0.5 + moneyness * 3))

        gamma = max(0.0001, 0.05 * atm_factor)
        theta = -mid * 0.02 * (1 / max(t, 0.003))
        vega = spot * math.sqrt(t) * 0.004 * atm_factor

        return {
            "putCall": option_type,
            "symbol": f"{symbol}  {datetime.now().strftime('%y%m%d')}{option_type[0]}{strike:.0f}",
            "description": f"{symbol} {option_type}",
            "bid": bid,
            "ask": ask,
            "last": mid,
            "mark": mid,
            "totalVolume": base_vol,
            "openInterest": base_oi,
            "strikePrice": strike,
            "daysToExpiration": dte,
            "volatility": round(iv * 100, 2),
            "delta": round(delta, 4),
            "gamma": round(gamma, 4),
            "theta": round(theta, 4),
            "vega": round(vega, 4),
            "multiplier": 100,
            "inTheMoney": (option_type == "CALL" and strike < spot)
            or (option_type == "PUT" and strike > spot),
        }


def create_api_client(source: str = "Mock (Demo)") -> Any:
    """
    Factory function to create the appropriate API client.

    Args:
        source: One of "Mock (Demo)", "IBKR (TWS)", "Schwab API"
    """
    if source == "IBKR (TWS)":
        import asyncio
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        from ibkr_api import create_ibkr_client
        client = create_ibkr_client()
        client.connect()  # raises ConnectionError if TWS not available
        return client

    elif source == "Schwab API":
        config = load_api_config()
        client = SchwabAPIClient(config)
        if not client.is_configured:
            print("Schwab API not configured. Falling back to mock data.")
            return MockSchwabAPI()
        return client

    # Default: Mock
    return MockSchwabAPI()
