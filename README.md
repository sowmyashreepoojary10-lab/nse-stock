# NSE Stock Analyzer (Streamlit)

Free NSE stock dashboard: candlestick charts, 20+ indicators, a rule-based signal score, alerts,
fundamentals, backtesting, stock comparison and a screener. Data comes from Yahoo Finance (free, ~15 min delayed).

> Screening aid only — not investment advice.

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy free on Streamlit Community Cloud
1. Create a **public** GitHub repo and push these files to its root (`streamlit_app.py`, `nse_core.py`,
   `indicators_extra.py`, `risk_metrics.py`, `signal_engine.py`, `backtest.py`, `requirements.txt`).
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**.
3. Pick the repo, branch `main`, main file `streamlit_app.py`, then **Deploy**.
4. Every `git push` redeploys automatically.

Free apps sleep after inactivity; the first visit afterwards takes ~30 seconds to wake.

## Files
| File | Purpose |
|---|---|
| `streamlit_app.py` | the UI (tabs: Overview, Alerts, Fundamentals, Backtest, Compare, Screener) |
| `nse_core.py` | your original indicator + fundamentals code, plus batched data download |
| `indicators_extra.py` | Stochastic, ATR, OBV, VWAP, Supertrend, pivots, candle patterns, event detection |
| `risk_metrics.py` | volatility, drawdown, Sharpe/Sortino, VaR, beta/alpha vs NIFTY 50 |
| `signal_engine.py` | composite score; tune `WEIGHTS` at the top |
| `backtest.py` | MA-cross, MACD and RSI backtests with costs and next-day execution |
