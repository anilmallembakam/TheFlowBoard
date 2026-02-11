# Architecture

## System Design

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser (Streamlit)                   │
│  ┌──────────┐  ┌─────────────────────┐  ┌───────────────┐  │
│  │ Sidebar  │  │   Heatmap + Charts  │  │  Flow Panel   │  │
│  │          │  │                     │  │               │  │
│  │ Symbol   │  │  Plotly Heatmap     │  │ Sentiment     │  │
│  │ Price    │  │  (strike x exp)     │  │ Metrics       │  │
│  │ Settings │  │                     │  │ Alerts        │  │
│  │ Thresh.  │  │  Net Contracts Bar  │  │ Premium Bars  │  │
│  │ Refresh  │  │  Snapshot Compare   │  │               │  │
│  └──────────┘  └─────────────────────┘  └───────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Top Flow Strikes Table                  │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                    Streamlit Session State
                              │
              ┌───────────────┼───────────────┐
              │               │               │
      ┌───────▼──────┐ ┌─────▼──────┐ ┌──────▼──────┐
      │  schwab_api  │ │  data_     │ │   flow_     │
      │  .py         │ │  processor │ │   detector  │
      │              │ │  .py       │ │   .py       │
      │ SchwabAPI    │ │            │ │             │
      │ MockSchwabAPI│ │ process()  │ │ detect()    │
      └──────────────┘ │ snapshot() │ │ sentiment() │
                       └────────────┘ │ top_strikes │
                                      └─────────────┘
```

## Data Flow

### 1. Fetch Phase
- `create_api_client()` returns `MockSchwabAPI` (demo) or `SchwabAPIClient` (live)
- `get_quote(symbol)` returns price data
- `get_option_chain(symbol)` returns full chain with calls/puts by expiration

### 2. Process Phase
- `DataProcessor.process_chain()` transforms raw API response:
  - Parses call/put maps into flat contract records
  - Computes DTE, premium, volume/OI ratio per contract
  - Finds ATM strike (closest to spot)
  - Filters strikes to configured range around ATM
  - Builds grid dict: `(strike, exp) -> {call_vol, put_vol, net, call_pct, ...}`

### 3. Detection Phase
- `FlowDetector.detect_flows()` scans contracts for:
  - `LARGE_VOL`: volume >= threshold (default 100)
  - `LARGE_PREM`: premium >= threshold (default $50K)
  - `UNUSUAL_VOL`: vol/OI >= threshold (default 3.0x)
  - `BLOCK`: volume >= 50 and premium >= $25K
- `FlowDetector.compute_sentiment()` calculates call/put imbalance

### 4. Render Phase
- Heatmap: Plotly `go.Heatmap` with custom green/red colorscale, text overlay
- Net contracts: Plotly horizontal bar chart
- Flow panel: HTML-rendered cards with flag chips
- Premium bars: Plotly horizontal bar chart sorted by premium
- Table: Streamlit `st.dataframe` with aggregated strike data

## Session State

```python
st.session_state = {
    "api_client":       MockSchwabAPI or SchwabAPIClient,
    "processor":        DataProcessor,
    "flow_detector":    FlowDetector,
    "last_update":      datetime,
    "processed_data":   dict,      # output of process_chain()
    "quote_data":       dict,      # output of get_quote()
    "auto_refresh":     bool,
    "symbol":           str,
    "display_config":   DisplayConfig,
    "threshold_config": ThresholdConfig,
    "color_config":     ColorConfig,
}
```

## Snapshot System

- Every 30s refresh saves a parquet file: `snapshots/{SYMBOL}_{YYYYMMDD_HHMMSS}.parquet`
- Columns: all contract fields + `snapshot_time` + `spot_price`
- Lookup: `load_snapshot(symbol, minutes_ago=60)` finds closest file within 10min window
- Cleanup: files older than 24 hours are deleted on each refresh cycle

## Mock Data Generation

`MockSchwabAPI` produces realistic data:
- Prices follow simplified Black-Scholes with random jitter
- Volume/OI concentrates around ATM (exponential decay by moneyness)
- 8% chance of volume spike per contract (3-15x multiplier)
- Greeks approximated from moneyness and time to expiration
- Cache refreshes every 25 seconds to simulate market movement
