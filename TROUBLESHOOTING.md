# Troubleshooting

## Common Issues

### "ModuleNotFoundError: No module named 'streamlit'"
```bash
pip install -r requirements.txt
```
Make sure you're in the correct virtual environment if using one.

### "No data available" warning on startup
- Click **Manual Refresh Now** in the sidebar
- If using live API, verify your credentials in `.env`
- Check that your symbol is valid

### Heatmap is blank or shows only dashes
- Ensure strikes above/below ATM sliders are set (default: 20)
- Try increasing the range or switching symbols
- In demo mode, data regenerates every ~25 seconds

### Auto-refresh not working
- Verify the "Auto-refresh every 30s" checkbox is checked in the sidebar
- Streamlit reruns the entire script on refresh; this is expected behavior

### Schwab API authentication errors
- Double-check `SCHWAB_CLIENT_ID` and `SCHWAB_CLIENT_SECRET` in `.env`
- Ensure redirect URI matches: `https://127.0.0.1:8182/callback`
- OAuth tokens expire after 30 minutes; re-authenticate if needed

### "Rate limit exceeded" error
- Schwab allows 120 requests/minute
- The dashboard uses ~4-6 per minute at 30s refresh
- If you hit limits, increase the refresh interval in the sidebar

### Snapshot files accumulating
- Snapshots older than 24 hours are auto-deleted on each refresh
- Manually delete the `snapshots/` directory to clear all history

### Port 8501 already in use
```bash
streamlit run app.py --server.port 8502
```

### Performance: dashboard feels slow
- Reduce strikes above/below ATM (try 10 instead of 20)
- Reduce max expirations in `config.py` (`max_expirations`)
- Disable auto-refresh when not actively monitoring

## Getting Help

- Check the Streamlit docs: https://docs.streamlit.io
- Schwab API docs: https://developer.schwab.com
