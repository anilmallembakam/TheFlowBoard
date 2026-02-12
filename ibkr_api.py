"""
TheFlowBoard - Interactive Brokers TWS API Client
Connects to TWS via ib_insync to fetch real-time options chain data.

Requirements:
    - TWS or IB Gateway running with API enabled
    - TWS: Edit > Global Configuration > API > Settings
        - Enable ActiveX and Socket Clients
        - Socket port: 7497 (paper) or 7496 (live)
        - Allow connections from localhost
    - pip install ib_insync
"""
import asyncio
import math
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from config import IBKRConfig, IBKR_CONTRACTS, STRIKE_INTERVALS, load_ibkr_config

# ib_insync (via eventkit) calls asyncio.get_event_loop() at import time.
# Streamlit runs scripts in a non-main thread that may lack an event loop,
# so we ensure one exists before importing.
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

try:
    from ib_insync import IB, Stock, Index, Option, Contract, util
    HAS_IB_INSYNC = True
except ImportError:
    HAS_IB_INSYNC = False


def _safe_float(val, default: float = 0.0) -> float:
    """Safely convert a value to float, returning default if NaN, None, or invalid."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if math.isnan(f) or math.isinf(f) else f
    except (ValueError, TypeError):
        return default


def _safe_int(val, default: int = 0) -> int:
    """Safely convert a value to int, returning default if NaN, None, or invalid."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if math.isnan(f) or math.isinf(f) else int(f)
    except (ValueError, TypeError):
        return default


class IBKRClient:
    """
    Interactive Brokers API client using ib_insync.

    Connects to TWS desktop, fetches quotes and option chains,
    and returns data in the same format as MockSchwabAPI/SchwabAPIClient
    so it plugs into the existing DataProcessor seamlessly.
    """

    def __init__(self, config: Optional[IBKRConfig] = None):
        self.config = config or load_ibkr_config()
        self._ib: Optional[Any] = None
        self._connected = False

    @property
    def is_configured(self) -> bool:
        return HAS_IB_INSYNC

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ib is not None and self._ib.isConnected()

    def connect(self) -> bool:
        """Connect to TWS. Returns True if successful."""
        if not HAS_IB_INSYNC:
            raise ImportError(
                "ib_insync is not installed. Run: pip install ib_insync"
            )

        if self.is_connected:
            return True

        import random

        # Try up to 3 different client IDs — TWS rejects duplicate client IDs
        # with Error 326 which surfaces as a TimeoutError in ib_insync
        client_ids = [self.config.client_id] + [random.randint(100, 999) for _ in range(2)]
        last_error = None

        for cid in client_ids:
            try:
                self._ib = IB()
                self._ib.connect(
                    host=self.config.host,
                    port=self.config.port,
                    clientId=cid,
                    readonly=self.config.readonly,
                    timeout=self.config.timeout,
                )
                self._connected = True
                return True
            except Exception as e:
                last_error = e
                # Clean up failed connection before retrying
                try:
                    if self._ib:
                        self._ib.disconnect()
                except Exception:
                    pass
                self._ib = None
                continue

        self._connected = False
        raise ConnectionError(
            f"Cannot connect to TWS at {self.config.host}:{self.config.port}. "
            f"Tried client IDs {client_ids}. "
            f"Make sure TWS is running with API enabled. Error: {last_error}"
        )

    def disconnect(self):
        """Disconnect from TWS."""
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
        self._connected = False

    def _ensure_connected(self):
        if not self.is_connected:
            self.connect()

    def _make_underlying(self, symbol: str) -> Contract:
        """Create the underlying contract for a symbol."""
        info = IBKR_CONTRACTS.get(symbol, {})
        sec_type = info.get("secType", "STK")
        exchange = info.get("exchange", "SMART")
        currency = info.get("currency", "USD")

        if sec_type == "IND":
            contract = Index(symbol, exchange, currency)
        else:
            contract = Stock(symbol, exchange, currency)
        return contract

    def _wait_for_data(self, ticker, timeout: float = 4.0) -> None:
        """Wait for streaming data to arrive on a ticker, up to timeout seconds."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            self._ib.sleep(0.2)
            # Check if we got a valid price
            last = _safe_float(ticker.last, 0)
            close = _safe_float(ticker.close, 0)
            bid = _safe_float(ticker.bid, 0)
            ask = _safe_float(ticker.ask, 0)
            if last > 0 or close > 0 or (bid > 0 and ask > 0):
                return
        # Final sleep to collect any remaining data
        self._ib.sleep(0.5)

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get current quote for a symbol. Returns same format as MockSchwabAPI."""
        self._ensure_connected()

        contract = self._make_underlying(symbol)
        self._ib.qualifyContracts(contract)

        # Use streaming mode — more reliable across paper/live accounts
        ticker = self._ib.reqMktData(contract, genericTickList="", snapshot=False)
        self._wait_for_data(ticker, timeout=4.0)

        last = _safe_float(ticker.last, _safe_float(ticker.close, 0.0))
        if last == 0:
            mp = ticker.marketPrice()
            last = _safe_float(mp, 0.0)

        prev_close = _safe_float(ticker.close, last)
        change = last - prev_close if prev_close > 0 else 0
        change_pct = (change / prev_close * 100) if prev_close > 0 else 0

        high = _safe_float(ticker.high, last)
        low = _safe_float(ticker.low, last)
        volume = _safe_int(ticker.volume, 0)

        self._ib.cancelMktData(contract)

        return {
            "symbol": symbol,
            "lastPrice": round(last, 2),
            "netChange": round(change, 2),
            "netPercentChange": round(change_pct, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "volume": volume,
            "timestamp": datetime.now().isoformat(),
        }

    def get_option_chain(
        self,
        symbol: str,
        strike_count: int = 40,
        days_to_expiration: int = 30,
    ) -> Dict[str, Any]:
        """
        Fetch option chain from IBKR.
        Returns data in the same format as MockSchwabAPI for compatibility.
        """
        self._ensure_connected()

        contract = self._make_underlying(symbol)
        self._ib.qualifyContracts(contract)

        # Get underlying price — use streaming for reliability on paper accounts
        ticker = self._ib.reqMktData(contract, genericTickList="", snapshot=False)
        self._wait_for_data(ticker, timeout=5.0)

        spot = _safe_float(ticker.last, _safe_float(ticker.close, 0.0))
        if spot == 0:
            spot = _safe_float(ticker.marketPrice(), 0.0)

        self._ib.cancelMktData(contract)

        if spot == 0:
            return self._empty_chain(symbol)

        # Request option chain definitions
        chains = self._ib.reqSecDefOptParams(
            contract.symbol,
            "",  # empty for all exchanges
            contract.secType,
            contract.conId,
        )

        if not chains:
            return self._empty_chain(symbol, spot)

        # Pick the best exchange — prefer SMART for stocks, CBOE for indices
        # For SPX: multiple trading classes exist (SPX standard, SPXW weeklies).
        # We merge expirations from all chain defs on the target exchange, but
        # only keep strikes that fall on clean intervals (5 for SPX, 1 for SPY).
        info = IBKR_CONTRACTS.get(symbol, {})
        preferred_exchange = info.get("exchange", "SMART")

        chain_def = chains[0]
        all_expirations = set()
        all_raw_strikes = set()

        for c in chains:
            if c.exchange == preferred_exchange:
                chain_def = c  # keep last match for exchange field
                all_expirations.update(c.expirations)
                all_raw_strikes.update(c.strikes)

        # If no match on preferred exchange, fall back to first chain
        if not all_expirations:
            all_expirations = set(chain_def.expirations)
            all_raw_strikes = set(chain_def.strikes)

        # Filter expirations within our DTE range
        today = datetime.now().date()
        max_date = today + timedelta(days=days_to_expiration)

        valid_expirations = []
        for exp_str in sorted(all_expirations):
            try:
                exp_date = datetime.strptime(exp_str, "%Y%m%d").date()
                if today <= exp_date <= max_date:
                    valid_expirations.append(exp_str)
            except ValueError:
                continue

        # Limit to 4 nearest expirations to stay within TWS data line limits
        valid_expirations = valid_expirations[:4]

        # Filter strikes to clean multiples of the symbol's strike interval
        # e.g. SPX → 5-point (6880, 6885, 6890...), SPY → 1-point
        strike_interval = STRIKE_INTERVALS.get(symbol, 5.0)
        clean_strikes = sorted([
            s for s in all_raw_strikes
            if abs(s % strike_interval) < 0.01
            or abs(s % strike_interval - strike_interval) < 0.01
        ])

        if not clean_strikes:
            # Fallback: use all strikes if none match the interval filter
            clean_strikes = sorted(all_raw_strikes)

        half = strike_count // 2

        # Find closest strike to spot
        atm_idx = min(range(len(clean_strikes)), key=lambda i: abs(clean_strikes[i] - spot))
        lo = max(0, atm_idx - half)
        hi = min(len(clean_strikes), atm_idx + half)
        selected_strikes = clean_strikes[lo:hi]

        # Build option contracts and request data
        # Process ONE expiration at a time, requesting at most 90 market data
        # lines (45 strikes × call + put) to stay within TWS's 100-line limit.
        # Cancel ALL lines and wait before moving to the next expiration.
        MAX_STRIKES_PER_BATCH = 45  # 45 strikes × 2 sides = 90 lines < 100

        calls_by_exp: Dict[str, Dict[str, List[Dict]]] = {}
        puts_by_exp: Dict[str, Dict[str, List[Dict]]] = {}

        # If we have more strikes than the batch limit, trim from both ends
        if len(selected_strikes) > MAX_STRIKES_PER_BATCH:
            trim = len(selected_strikes) - MAX_STRIKES_PER_BATCH
            trim_lo = trim // 2
            trim_hi = trim - trim_lo
            selected_strikes = selected_strikes[trim_lo: len(selected_strikes) - trim_hi]

        for exp_str in valid_expirations:
            exp_key = f"{exp_str[:4]}-{exp_str[4:6]}-{exp_str[6:8]}"
            calls_by_exp[exp_key] = {}
            puts_by_exp[exp_key] = {}

            # Build option contracts for this expiration
            opt_contracts = []
            for strike in selected_strikes:
                for right in ("C", "P"):
                    opt = Option(
                        symbol=contract.symbol,
                        lastTradeDateOrContractMonth=exp_str,
                        strike=strike,
                        right=right,
                        exchange=preferred_exchange,
                    )
                    opt_contracts.append(opt)

            # Qualify in batches
            qualified = []
            batch_size = 50
            for i in range(0, len(opt_contracts), batch_size):
                batch = opt_contracts[i:i + batch_size]
                try:
                    self._ib.qualifyContracts(*batch)
                    qualified.extend([c for c in batch if c.conId > 0])
                except Exception:
                    continue

            if not qualified:
                continue

            # Request market data for ALL options in this expiration at once
            # (max 90 lines which is under the 100 limit)
            # Generic ticks: 100=OI, 106=IV (which provides computed greeks)
            tickers = []
            for opt in qualified:
                t = self._ib.reqMktData(opt, genericTickList="100,106", snapshot=False)
                tickers.append((opt, t))

            # Wait for streaming data — pump the event loop in small steps
            # so TWS can deliver updates incrementally
            for _ in range(12):
                self._ib.sleep(0.5)

            # Process results and cancel ALL market data for this expiration
            for opt, tkr in tickers:
                self._process_option_ticker(
                    opt, tkr, exp_str, today, spot, symbol,
                    calls_by_exp, puts_by_exp,
                )
                self._ib.cancelMktData(opt)

            # Wait for cancellations to fully process before next expiration
            self._ib.sleep(1.0)

        return {
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

    def _process_option_ticker(
        self, opt, ticker, exp_str: str, today, spot: float, symbol: str,
        calls_by_exp: Dict, puts_by_exp: Dict,
    ):
        """Extract data from a single option ticker and add to result dicts."""
        exp_key = f"{exp_str[:4]}-{exp_str[4:6]}-{exp_str[6:8]}"
        exp_date = datetime.strptime(
            opt.lastTradeDateOrContractMonth, "%Y%m%d"
        ).date()
        dte = (exp_date - today).days
        strike = opt.strike
        strike_key = f"{strike:.1f}"

        bid = _safe_float(ticker.bid, 0)
        ask = _safe_float(ticker.ask, 0)
        last = _safe_float(ticker.last, 0)
        mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else last
        volume = _safe_int(ticker.volume, 0)

        # Model greeks — modelGreeks are computed by TWS, lastGreeks are from exchange
        greeks = ticker.modelGreeks or ticker.lastGreeks
        delta = _safe_float(getattr(greeks, 'delta', None), 0) if greeks else 0
        gamma = _safe_float(getattr(greeks, 'gamma', None), 0) if greeks else 0
        theta = _safe_float(getattr(greeks, 'theta', None), 0) if greeks else 0
        vega = _safe_float(getattr(greeks, 'vega', None), 0) if greeks else 0
        iv = _safe_float(getattr(greeks, 'impliedVol', None), 0) if greeks else 0

        # Open interest from generic tick 100
        if opt.right == "C":
            oi = _safe_int(getattr(ticker, 'callOpenInterest', 0), 0)
        else:
            oi = _safe_int(getattr(ticker, 'putOpenInterest', 0), 0)

        option_type = "CALL" if opt.right == "C" else "PUT"
        multiplier = _safe_int(opt.multiplier, 100) if opt.multiplier else 100

        contract_data = {
            "putCall": option_type,
            "symbol": f"{symbol} {exp_str}{opt.right}{strike:.0f}",
            "description": f"{symbol} {option_type}",
            "bid": round(bid, 2),
            "ask": round(ask, 2),
            "last": round(last, 2),
            "mark": round(mid, 2),
            "totalVolume": volume,
            "openInterest": int(oi),
            "strikePrice": strike,
            "daysToExpiration": dte,
            "volatility": round(iv * 100, 2),
            "delta": round(delta, 4),
            "gamma": round(gamma, 4),
            "theta": round(theta, 4),
            "vega": round(vega, 4),
            "multiplier": multiplier,
            "inTheMoney": (option_type == "CALL" and strike < spot)
                          or (option_type == "PUT" and strike > spot),
        }

        if opt.right == "C":
            calls_by_exp[exp_key][strike_key] = [contract_data]
        else:
            puts_by_exp[exp_key][strike_key] = [contract_data]

    def _empty_chain(self, symbol: str, spot: float = 0) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "status": "SUCCESS",
            "underlying": {"symbol": symbol, "last": spot, "mark": spot},
            "callExpDateMap": {},
            "putExpDateMap": {},
        }


def create_ibkr_client(config: Optional[IBKRConfig] = None) -> IBKRClient:
    """Factory function to create an IBKR client."""
    if not HAS_IB_INSYNC:
        raise ImportError(
            "ib_insync is required for IBKR integration.\n"
            "Install it with: pip install ib_insync"
        )
    return IBKRClient(config)
