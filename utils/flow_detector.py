"""
TheFlowBoard - Flow Detector
Detects large trades, unusual activity, and calculates market sentiment.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import ThresholdConfig


@dataclass
class FlowAlert:
    """Represents a detected flow event."""
    strike: float
    expiration: str
    option_type: str  # CALL or PUT
    volume: int
    open_interest: int
    premium: float
    volume_oi_ratio: float
    flags: List[str] = field(default_factory=list)
    dte: int = 0
    mid_price: float = 0.0

    @property
    def severity(self) -> str:
        """High if multiple flags, medium if one, low otherwise."""
        if len(self.flags) >= 2:
            return "high"
        elif len(self.flags) == 1:
            return "medium"
        return "low"

    @property
    def premium_display(self) -> str:
        if self.premium >= 1_000_000:
            return f"${self.premium / 1_000_000:.1f}M"
        if self.premium >= 1_000:
            return f"${self.premium / 1_000:.0f}K"
        return f"${self.premium:.0f}"


class FlowDetector:
    """Detects notable options flow and computes sentiment."""

    def __init__(self, thresholds: Optional[ThresholdConfig] = None):
        self.thresholds = thresholds or ThresholdConfig()

    def detect_flows(self, contracts: List[Dict[str, Any]]) -> List[FlowAlert]:
        """Scan all contracts and return notable flow alerts."""
        alerts: List[FlowAlert] = []

        for c in contracts:
            flags = self._check_flags(c)
            if flags:
                alert = FlowAlert(
                    strike=c["strike"],
                    expiration=c["expiration"],
                    option_type=c["type"],
                    volume=c["volume"],
                    open_interest=c["openInterest"],
                    premium=c["premium"],
                    volume_oi_ratio=c.get("volumeOIRatio", 0),
                    flags=flags,
                    dte=c.get("dte", 0),
                    mid_price=c.get("mid", 0),
                )
                alerts.append(alert)

        # Sort by premium descending
        alerts.sort(key=lambda a: a.premium, reverse=True)
        return alerts

    def _check_flags(self, contract: Dict[str, Any]) -> List[str]:
        """Check a contract against all thresholds and return list of triggered flags."""
        flags = []
        vol = contract.get("volume", 0)
        oi = contract.get("openInterest", 0)
        premium = contract.get("premium", 0)

        if vol >= self.thresholds.large_volume:
            flags.append("LARGE_VOL")

        if premium >= self.thresholds.large_premium:
            flags.append("LARGE_PREM")

        if oi > 0 and (vol / oi) >= self.thresholds.unusual_volume_ratio:
            flags.append("UNUSUAL_VOL")

        if vol >= self.thresholds.block_trade_min and premium >= 25_000:
            flags.append("BLOCK")

        return flags

    def compute_sentiment(
        self, contracts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute overall call/put sentiment from the flow data.

        Returns:
            - call_volume, put_volume, total_volume
            - call_premium, put_premium, total_premium
            - call_pct, put_pct
            - sentiment: 'Bullish', 'Bearish', or 'Neutral'
            - imbalance: float (-1 to 1, positive = bullish)
        """
        call_vol = 0
        put_vol = 0
        call_prem = 0.0
        put_prem = 0.0

        for c in contracts:
            if c["type"] == "CALL":
                call_vol += c["volume"]
                call_prem += c["premium"]
            else:
                put_vol += c["volume"]
                put_prem += c["premium"]

        total_vol = call_vol + put_vol
        total_prem = call_prem + put_prem

        if total_vol > 0:
            call_pct = call_vol / total_vol
            put_pct = put_vol / total_vol
            imbalance = (call_vol - put_vol) / total_vol
        else:
            call_pct = 0.5
            put_pct = 0.5
            imbalance = 0.0

        if imbalance > 0.1:
            sentiment = "Bullish"
        elif imbalance < -0.1:
            sentiment = "Bearish"
        else:
            sentiment = "Neutral"

        return {
            "call_volume": call_vol,
            "put_volume": put_vol,
            "total_volume": total_vol,
            "call_premium": call_prem,
            "put_premium": put_prem,
            "total_premium": total_prem,
            "call_pct": round(call_pct * 100, 1),
            "put_pct": round(put_pct * 100, 1),
            "sentiment": sentiment,
            "imbalance": round(imbalance, 3),
        }

    def top_flow_strikes(
        self, contracts: List[Dict[str, Any]], top_n: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Aggregate flow by strike and return the top N strikes by total premium.

        Each entry has: strike, call_vol, put_vol, call_prem, put_prem,
                        net_prem, dominant_type, total_premium
        """
        strike_agg: Dict[float, Dict[str, Any]] = {}

        for c in contracts:
            s = c["strike"]
            if s not in strike_agg:
                strike_agg[s] = {
                    "strike": s,
                    "call_vol": 0,
                    "put_vol": 0,
                    "call_prem": 0.0,
                    "put_prem": 0.0,
                }
            entry = strike_agg[s]
            if c["type"] == "CALL":
                entry["call_vol"] += c["volume"]
                entry["call_prem"] += c["premium"]
            else:
                entry["put_vol"] += c["volume"]
                entry["put_prem"] += c["premium"]

        results = []
        for entry in strike_agg.values():
            entry["total_premium"] = entry["call_prem"] + entry["put_prem"]
            entry["net_prem"] = entry["call_prem"] - entry["put_prem"]
            entry["total_vol"] = entry["call_vol"] + entry["put_vol"]
            tv = entry["total_vol"]
            entry["dominant_type"] = (
                "CALL" if tv > 0 and entry["call_vol"] / tv > 0.55
                else "PUT" if tv > 0 and entry["put_vol"] / tv > 0.55
                else "MIXED"
            )
            if entry["total_vol"] > 0:
                results.append(entry)

        results.sort(key=lambda x: x["total_premium"], reverse=True)
        return results[:top_n]

    def format_premium(self, value: float) -> str:
        """Format premium for display."""
        if abs(value) >= 1_000_000:
            return f"${value / 1_000_000:.1f}M"
        if abs(value) >= 1_000:
            return f"${value / 1_000:.0f}K"
        return f"${value:.0f}"
