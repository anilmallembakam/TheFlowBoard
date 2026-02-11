# TheFlowBoard

Real-time options flow heatmap dashboard built with Streamlit. Visualizes multi-expiration options activity with color-coded call/put dominance, flow detection, and sentiment analysis.

## Features

- **Multi-Expiration Heatmap**: Strike ladder vs expiration columns with color-coded net volume
- **Flow Detection**: Automatically flags large volume, large premium, unusual volume/OI, and block trades
- **Sentiment Indicator**: Real-time bullish/bearish/neutral based on call/put imbalance
- **Net Contracts Chart**: Horizontal bar chart showing net positioning per strike
- **Top Flow Strikes Table**: Aggregated metrics for the most active strikes
- **Historical Snapshots**: Automatic 30s snapshots for 60m/15m comparison
- **Demo Mode**: Works immediately with realistic simulated data (no API required)

## Supported Symbols

SPX, SPY, QQQ, IWM (extensible in `config.py`)

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app launches in Demo Mode by default with mock data.

## Schwab API (Live Data)

1. Copy `.env.template` to `.env`
2. Add your Schwab API credentials
3. In `app.py`, change `create_api_client(use_mock=True)` to `create_api_client(use_mock=False)`
4. Complete OAuth2 flow when prompted

### Getting Schwab API Credentials

1. Create a developer account at [developer.schwab.com](https://developer.schwab.com)
2. Register a new application
3. Note your Client ID and Client Secret
4. Set redirect URI to `https://127.0.0.1:8182/callback`

## Project Structure

```
TheFlowBoard/
├── app.py                    # Main Streamlit dashboard
├── schwab_api.py             # API client (real + mock)
├── config.py                 # Configuration & thresholds
├── requirements.txt          # Python dependencies
├── .env.template             # API credentials template
├── install.bat               # Windows installer
├── install.sh                # Linux/Mac installer
├── utils/
│   ├── data_processor.py     # Options chain processing
│   └── flow_detector.py      # Flow detection & sentiment
└── snapshots/                # Auto-saved historical data
```

## Configuration

All thresholds are adjustable via the sidebar:

| Setting | Default | Description |
|---------|---------|-------------|
| Strikes Above/Below ATM | 20 | Number of strikes to show around ATM |
| Large Volume | 100 | Minimum contracts for large volume flag |
| Large Premium | $50K | Minimum premium for large premium flag |
| Unusual Vol/OI Ratio | 3.0x | Volume/OI ratio threshold |

## Color Scheme

- **Green cells**: Call-dominant volume
- **Red cells**: Put-dominant volume
- **Gray cells**: Balanced or no volume
- **Yellow line**: At-the-money (ATM) strike
- Color intensity reflects how strongly one side dominates

## Rate Limits

- Schwab API: 120 requests/minute
- Dashboard uses approximately 4-6 requests/minute at 30s refresh
