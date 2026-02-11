# Project Summary

## TheFlowBoard - Options Flow Dashboard

### Overview
A real-time options flow visualization tool that displays multi-expiration heatmaps with flow detection and sentiment analysis. Designed for options traders who want to monitor institutional activity and unusual options flow.

### Core Components

| File | Purpose | Lines |
|------|---------|-------|
| `app.py` | Main Streamlit dashboard with heatmap, flow panel, charts | ~500 |
| `schwab_api.py` | Schwab API client + Mock API for demo mode | ~300 |
| `utils/data_processor.py` | Options chain processing, grid building, snapshots | ~250 |
| `utils/flow_detector.py` | Flow detection, sentiment, top strikes analysis | ~250 |
| `config.py` | Centralized configuration | ~80 |

### Key Features

1. **Multi-Expiration Heatmap**: Plotly-based interactive grid showing net volume across strikes and expirations
2. **Color-Coded Cells**: Green for call-dominant, red for put-dominant, intensity by dominance percentage
3. **ATM Highlight**: Yellow line marking the at-the-money strike
4. **Flow Detection Engine**: Flags large volume (>100), large premium (>$50K), unusual vol/OI (>3x), block trades
5. **Sentiment Indicator**: Bullish/Bearish/Neutral based on call/put volume imbalance
6. **Net Contracts Bar Chart**: Horizontal bars per strike showing directional positioning
7. **Premium Flow Bars**: Top trades sorted by premium size
8. **Top Strikes Table**: Aggregated metrics for most active strikes
9. **Historical Snapshots**: Parquet files saved every 30s for 60m/15m lookback
10. **Auto-Refresh**: Configurable 30s refresh cycle
11. **Demo Mode**: Realistic mock data for immediate use without API credentials

### Data Flow

```
API (Schwab or Mock)
    → Raw Option Chain JSON
    → DataProcessor.process_chain()
        → Strike grid with call/put volumes
        → Contract list with computed fields
    → FlowDetector.detect_flows()
        → Flow alerts with severity flags
    → FlowDetector.compute_sentiment()
        → Bullish/Bearish/Neutral
    → Plotly heatmap + bar charts
    → Streamlit renders to browser
```

### Symbols
SPX (default), SPY, QQQ, IWM. Add more in `config.py` `SYMBOLS` list with corresponding mock prices and strike intervals.
