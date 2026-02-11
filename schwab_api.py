"""
TheFlowBoard - Schwab API Client
Provides both real Schwab API integration and a MockSchwabAPI for demo mode.
"""
import random
import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import requests

from config import (
    APIConfig, MOCK_PRICES, STRIKE_INTERVALS,
    load_api_config,
)


class SchwabAPIClient:
    """Real Schwab API client with OAuth2 authentication."""

    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or load_api_config()
        self.access_token: Optional[str] = None
        self.token_expiry: float = 0
        self.session = requests.Session()
        self._request_times: List[float] = []

    @property
    def is_configured(self) -> bool:
        return bool(self.config.client_id and self.config.client_secret)

    def get_auth_url(self) -> str:
        """Generate OAuth2 authorization URL."""
        params = {
            "response_type": "code",
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "scope": "readonly",
        }
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.config.auth_url}?{qs}"

    def exchange_code(self, auth_code: str) -> bool:
        """Exchange authorization code for access token."""
        data = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": self.config.redirect_uri,
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
        }
        try:
            resp = self.session.post(self.config.token_url, data=data, timeout=10)
            resp.raise_for_status()
            token_data = resp.json()
            self.access_token = token_data["access_token"]
            self.token_expiry = time.time() + token_data.get("expires_in", 1800)
            return True
        except Exception:
            return False

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

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get current quote for a symbol."""
        if not self.access_token:
            raise RuntimeError("Not authenticated. Call exchange_code() first.")
        self._check_rate_limit()
        url = f"{self.config.base_url}/quotes/{symbol}"
        resp = self.session.get(url, headers=self._get_headers(), timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_option_chain(
        self,
        symbol: str,
        strike_count: int = 40,
        days_to_expiration: int = 30,
    ) -> Dict[str, Any]:
        """Get option chain from Schwab API."""
        if not self.access_token:
            raise RuntimeError("Not authenticated. Call exchange_code() first.")
        self._check_rate_limit()
        url = f"{self.config.base_url}/chains"
        params = {
            "symbol": symbol,
            "contractType": "ALL",
            "strikeCount": strike_count,
            "range": "ALL",
            "toDate": (datetime.now() + timedelta(days=days_to_expiration)).strftime("%Y-%m-%d"),
        }
        resp = self.session.get(url, headers=self._get_headers(), params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()


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
        try:
            import asyncio
            try:
                asyncio.get_event_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())
            from ibkr_api import create_ibkr_client
            client = create_ibkr_client()
            client.connect()
            return client
        except ImportError:
            print("ib_insync not installed. Run: pip install ib_insync")
            print("Falling back to mock data.")
            return MockSchwabAPI()
        except Exception as e:
            print(f"IBKR connection failed: {e}")
            print("Falling back to mock data.")
            return MockSchwabAPI()

    elif source == "Schwab API":
        config = load_api_config()
        client = SchwabAPIClient(config)
        if not client.is_configured:
            print("Schwab API not configured. Falling back to mock data.")
            return MockSchwabAPI()
        return client

    # Default: Mock
    return MockSchwabAPI()
