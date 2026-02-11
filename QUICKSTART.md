# QuickStart Guide

## 1. Install & Launch

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 2. Dashboard Layout

The dashboard has three main sections:

### Left Sidebar
- **Symbol selector**: Switch between SPX, SPY, QQQ, IWM
- **Price display**: Current price with change
- **Settings**: Adjust strikes above/below ATM
- **Thresholds**: Tune flow detection sensitivity
- **Refresh controls**: Auto-refresh toggle + manual refresh button

### Center: Heatmap
- Rows = strike prices (highest at top)
- Columns = expiration dates
- Cell values = net volume (calls - puts)
- Green = call-dominant, Red = put-dominant
- Yellow horizontal line = ATM strike

### Right: Flow Panel
- Sentiment badge (Bullish/Bearish/Neutral)
- Volume and premium metrics
- Live flow alerts with flags (VOL, PREM, UNUSUAL, BLOCK)
- Top flow by premium bar chart

## 3. Reading the Heatmap

| Cell Color | Meaning |
|-----------|---------|
| Bright green | Strong call volume dominance |
| Light green | Moderate call lean |
| Gray | Balanced or no activity |
| Light red | Moderate put lean |
| Bright red | Strong put volume dominance |

Values show net contracts: positive = more calls, negative = more puts.

## 4. Flow Detection Flags

| Flag | Trigger |
|------|---------|
| VOL | Volume >= 100 contracts |
| PREM | Premium >= $50K |
| UNUSUAL | Volume/OI ratio >= 3.0x |
| BLOCK | Volume >= 50 contracts + Premium >= $25K |

Adjust these thresholds in the sidebar to tune sensitivity.

## 5. Switching to Live Data

1. Copy `.env.template` to `.env`
2. Fill in your Schwab API credentials
3. Change `use_mock=True` to `use_mock=False` in `app.py` line where `create_api_client` is called
4. Restart the app
