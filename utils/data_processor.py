"""
TheFlowBoard - Data Processor
Processes raw option chain data into structured grids for the heatmap display.
"""
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from config import DisplayConfig, SNAPSHOT_DIR


class DataProcessor:
    """Processes option chain data into display-ready structures."""

    def __init__(self, display_config: Optional[DisplayConfig] = None):
        self.config = display_config or DisplayConfig()

    def process_chain(
        self, chain_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process raw option chain into structured data.

        Returns dict with:
            - spot_price: float
            - expirations: list of expiration date strings
            - strikes: list of strike prices
            - grid: dict mapping (strike, exp) -> cell data
            - contracts: list of all individual contract records
            - atm_strike: closest strike to spot
        """
        underlying = chain_data.get("underlying", {})
        spot = underlying.get("last", underlying.get("mark", 0))
        if spot == 0:
            return self._empty_result()

        call_map = chain_data.get("callExpDateMap", {})
        put_map = chain_data.get("putExpDateMap", {})

        contracts: List[Dict[str, Any]] = []
        exp_set = set()
        strike_set = set()

        # Parse calls
        for exp_key, strikes_map in call_map.items():
            exp_date = exp_key.split(":")[0] if ":" in exp_key else exp_key
            exp_set.add(exp_date)
            for strike_key, contract_list in strikes_map.items():
                strike = float(strike_key)
                strike_set.add(strike)
                for c in contract_list:
                    contracts.append(self._parse_contract(c, exp_date, strike, "CALL"))

        # Parse puts
        for exp_key, strikes_map in put_map.items():
            exp_date = exp_key.split(":")[0] if ":" in exp_key else exp_key
            exp_set.add(exp_date)
            for strike_key, contract_list in strikes_map.items():
                strike = float(strike_key)
                strike_set.add(strike)
                for c in contract_list:
                    contracts.append(self._parse_contract(c, exp_date, strike, "PUT"))

        # Sort expirations and strikes
        expirations = sorted(exp_set)[:self.config.max_expirations]
        all_strikes = sorted(strike_set)

        # Find ATM strike
        atm_strike = min(all_strikes, key=lambda s: abs(s - spot)) if all_strikes else spot

        # Filter strikes around ATM
        atm_idx = all_strikes.index(atm_strike) if atm_strike in all_strikes else len(all_strikes) // 2
        lo = max(0, atm_idx - self.config.strikes_below_atm)
        hi = min(len(all_strikes), atm_idx + self.config.strikes_above_atm + 1)
        strikes = all_strikes[lo:hi]

        # Build grid
        grid = self._build_grid(contracts, strikes, expirations, spot)

        return {
            "spot_price": spot,
            "expirations": expirations,
            "strikes": strikes,
            "grid": grid,
            "contracts": contracts,
            "atm_strike": atm_strike,
            "timestamp": datetime.now().isoformat(),
        }

    @staticmethod
    def _safe_num(val, default=0):
        """Convert to number safely, handling NaN/None/inf."""
        if val is None:
            return default
        try:
            f = float(val)
            if np.isnan(f) or np.isinf(f):
                return default
            return f
        except (ValueError, TypeError):
            return default

    def _parse_contract(
        self, c: Dict, exp_date: str, strike: float, option_type: str
    ) -> Dict[str, Any]:
        """Parse a single contract into a normalized record."""
        volume = int(self._safe_num(c.get("totalVolume", 0)))
        oi = int(self._safe_num(c.get("openInterest", 0)))
        mid = self._safe_num(c.get("mark", c.get("last", 0)))
        multiplier = int(self._safe_num(c.get("multiplier", 100), 100))
        premium = volume * mid * multiplier

        return {
            "expiration": exp_date,
            "strike": strike,
            "type": option_type,
            "volume": volume,
            "openInterest": oi,
            "bid": self._safe_num(c.get("bid", 0)),
            "ask": self._safe_num(c.get("ask", 0)),
            "mid": mid,
            "last": self._safe_num(c.get("last", 0)),
            "premium": premium,
            "iv": self._safe_num(c.get("volatility", 0)),
            "delta": self._safe_num(c.get("delta", 0)),
            "gamma": self._safe_num(c.get("gamma", 0)),
            "theta": self._safe_num(c.get("theta", 0)),
            "vega": self._safe_num(c.get("vega", 0)),
            "dte": int(self._safe_num(c.get("daysToExpiration", 0))),
            "itm": c.get("inTheMoney", False),
            "volumeOIRatio": round(volume / oi, 2) if oi > 0 else 0,
        }

    def _build_grid(
        self,
        contracts: List[Dict],
        strikes: List[float],
        expirations: List[str],
        spot: float = 0.0,
    ) -> Dict[Tuple[float, str], Dict[str, Any]]:
        """Build the heatmap grid aggregating call/put data per (strike, expiration)."""
        grid: Dict[Tuple[float, str], Dict[str, Any]] = {}

        for strike in strikes:
            for exp in expirations:
                grid[(strike, exp)] = {
                    "call_volume": 0,
                    "put_volume": 0,
                    "call_oi": 0,
                    "put_oi": 0,
                    "call_premium": 0,
                    "put_premium": 0,
                    "call_gamma": 0.0,
                    "put_gamma": 0.0,
                    "net_volume": 0,
                    "total_volume": 0,
                    "call_pct": 0.5,
                    "gex": 0.0,
                    "dominant_type": "neutral",
                    "display_value": "",
                }

        for c in contracts:
            key = (c["strike"], c["expiration"])
            if key not in grid:
                continue
            cell = grid[key]
            if c["type"] == "CALL":
                cell["call_volume"] += c["volume"]
                cell["call_oi"] += c["openInterest"]
                cell["call_premium"] += c["premium"]
                cell["call_gamma"] += c["gamma"] * c["openInterest"]
            else:
                cell["put_volume"] += c["volume"]
                cell["put_oi"] += c["openInterest"]
                cell["put_premium"] += c["premium"]
                cell["put_gamma"] += c["gamma"] * c["openInterest"]

        # Compute derived fields
        for key, cell in grid.items():
            total = cell["call_volume"] + cell["put_volume"]
            cell["total_volume"] = total
            cell["net_volume"] = cell["call_volume"] - cell["put_volume"]
            if total > 0:
                cell["call_pct"] = cell["call_volume"] / total
            else:
                cell["call_pct"] = 0.5
            cell["dominant_type"] = (
                "call" if cell["call_pct"] > 0.55
                else "put" if cell["call_pct"] < 0.45
                else "neutral"
            )

            # GEX per cell: OI * Gamma * Spot^2 * 0.01 * multiplier
            # Calls have positive gamma exposure, puts have negative
            # Formula: (call_OI * call_gamma - put_OI * put_gamma) * spot^2 * 0.01 * 100
            cell["gex"] = (cell["call_gamma"] - cell["put_gamma"]) * spot * spot * 0.01 * 100

            cell["display_value"] = self._format_volume(cell["net_volume"])

        return grid

    @staticmethod
    def _format_volume(value: int) -> str:
        """Format volume for compact display."""
        if value == 0:
            return "-"
        abs_val = abs(value)
        if abs_val >= 1000:
            return f"{value / 1000:.1f}k"
        return str(value)

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "spot_price": 0,
            "expirations": [],
            "strikes": [],
            "grid": {},
            "contracts": [],
            "atm_strike": 0,
            "timestamp": datetime.now().isoformat(),
        }

    # --- Snapshot management ---

    def save_snapshot(self, processed_data: Dict[str, Any], symbol: str):
        """Save a snapshot of processed data to parquet."""
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(SNAPSHOT_DIR, f"{symbol}_{ts}.parquet")
        if processed_data["contracts"]:
            df = pd.DataFrame(processed_data["contracts"])
            df["snapshot_time"] = processed_data["timestamp"]
            df["spot_price"] = processed_data["spot_price"]
            df.to_parquet(filepath, index=False)

    def load_snapshot(self, symbol: str, minutes_ago: int) -> Optional[Dict[str, Any]]:
        """Load the closest snapshot to N minutes ago."""
        if not os.path.exists(SNAPSHOT_DIR):
            return None

        target_time = datetime.now() - pd.Timedelta(minutes=minutes_ago)
        best_file = None
        best_diff = float("inf")

        for fname in os.listdir(SNAPSHOT_DIR):
            if not fname.startswith(symbol) or not fname.endswith(".parquet"):
                continue
            try:
                parts = fname.replace(f"{symbol}_", "").replace(".parquet", "")
                file_time = datetime.strptime(parts, "%Y%m%d_%H%M%S")
                diff = abs((file_time - target_time).total_seconds())
                if diff < best_diff:
                    best_diff = diff
                    best_file = os.path.join(SNAPSHOT_DIR, fname)
            except ValueError:
                continue

        if best_file and best_diff < 600:  # within 10 minutes
            df = pd.read_parquet(best_file)
            return {
                "contracts": df.to_dict("records"),
                "spot_price": df["spot_price"].iloc[0] if "spot_price" in df.columns else 0,
                "timestamp": df["snapshot_time"].iloc[0] if "snapshot_time" in df.columns else "",
            }
        return None

    def cleanup_old_snapshots(self, max_age_hours: int = 24):
        """Delete snapshots older than max_age_hours."""
        if not os.path.exists(SNAPSHOT_DIR):
            return
        cutoff = datetime.now() - pd.Timedelta(hours=max_age_hours)
        for fname in os.listdir(SNAPSHOT_DIR):
            if not fname.endswith(".parquet"):
                continue
            try:
                parts = fname.rsplit("_", 2)
                date_str = parts[-2] + "_" + parts[-1].replace(".parquet", "")
                file_time = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                if file_time < cutoff:
                    os.remove(os.path.join(SNAPSHOT_DIR, fname))
            except (ValueError, IndexError):
                continue

    def compute_net_contracts(
        self, processed_data: Dict[str, Any]
    ) -> Dict[float, int]:
        """Compute net contracts (calls - puts) per strike for the bar chart."""
        net: Dict[float, int] = {}
        for c in processed_data.get("contracts", []):
            strike = c["strike"]
            vol = c["volume"]
            if c["type"] == "CALL":
                net[strike] = net.get(strike, 0) + vol
            else:
                net[strike] = net.get(strike, 0) - vol
        return net

    def compute_net_premium(
        self, processed_data: Dict[str, Any]
    ) -> Dict[float, float]:
        """Compute net premium (calls - puts) per strike."""
        net: Dict[float, float] = {}
        for c in processed_data.get("contracts", []):
            strike = c["strike"]
            prem = c["premium"]
            if c["type"] == "CALL":
                net[strike] = net.get(strike, 0) + prem
            else:
                net[strike] = net.get(strike, 0) - prem
        return net

    def compute_gex(
        self, processed_data: Dict[str, Any]
    ) -> Dict[float, float]:
        """
        Compute Gamma Exposure (GEX) per strike across all expirations.

        GEX = OI * Gamma * Spot^2 * 0.01 * Contract_Multiplier
        - Calls contribute positive GEX (dealers are long gamma)
        - Puts contribute negative GEX (dealers are short gamma)

        Returns dict mapping strike -> net GEX in dollars.
        """
        spot = processed_data.get("spot_price", 0)
        if spot == 0:
            return {}

        gex: Dict[float, float] = {}
        for c in processed_data.get("contracts", []):
            strike = c["strike"]
            oi = c.get("openInterest", 0)
            gamma = c.get("gamma", 0)
            # GEX per contract = OI * gamma * spot^2 * 0.01 * multiplier(100)
            contract_gex = oi * gamma * spot * spot * 0.01 * 100
            if c["type"] == "CALL":
                gex[strike] = gex.get(strike, 0) + contract_gex
            else:
                # Puts: dealers are typically short gamma on puts
                gex[strike] = gex.get(strike, 0) - contract_gex
        return gex
