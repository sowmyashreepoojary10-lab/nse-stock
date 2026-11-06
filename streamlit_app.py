"""
NSE Stock Analyzer — Streamlit app (free to host on Streamlit Community Cloud)
Run locally:  streamlit run streamlit_app.py
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

import nse_core as nc
from backtest import run_backtest
from indicators_extra import detect_events

st.set_page_config(page_title="NSE Stock Analyzer", page_icon="📈", layout="wide")

# ── cached data access (15 min) so reruns/clicks don't re-hit Yahoo ────────

@st.cache_data(ttl=900, show_spinner=False)
def get_hist(ticker: str, period: str = "2y"):
    return nc.fetch_history(ticker, period)


@st.cache_data(ttl=900, show_spinner=False)
def get_many(tickers: tuple, period: str = "2y"):
    return nc.fetch_many(list(tickers), period)


@st.cache_data(ttl=900, show_spinner=False)
def get_bench():
    try:
        return nc.fetch_history(nc.BENCHMARK, "2y")
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_fund(ticker: str):
    return nc.get_fundamentals(yf.Ticker(ticker))


def fmt(v, suffix="", dash="–"):
    return dash if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:,}{suffix}"


# ── sidebar ────────────────────────────────────────────────────────────────

st.sidebar.title("📈 NSE Analyzer")
symbol = st.sidebar.text_input("NSE symbol", value="RELIANCE", help="e.g. TCS, INFY, HDFCBANK (no .NS needed)")
rf = st.sidebar.slider("Risk-free rate (%)", 4.0, 9.0, 7.0, 0.25) / 100
st.sidebar.caption("Data: Yahoo Finance (free, ~15 min delayed). Screening aid only — not investment advice.")

ticker = nc.to_ticker(symbol) if symbol.strip() else None
st.title("NSE Stock Analyzer")

tab_over, tab_alerts, tab_fund, tab_bt, tab_cmp, tab_scan = st.tabs(
    ["Overview", "Alerts", "Fundamentals", "Backtest", "Compare", "Screener"])

# ── Overview ───────────────────────────────────────────────────────────────
with tab_over:
    if not ticker:
        st.info("Enter a symbol in the sidebar.")
    else:
        try:
            hist = get_hist(ticker)
            a = nc.analyze(hist, get_bench(), rf)
            t, e, r, s = a["technical"], a["extra"], a["risk"], a["signal"]

            if a["corporate_action_flag"]:
                st.warning("A >15% single-day move was found in the last 2 years (possible split/demerger). "
                           "Moving averages and Bollinger figures may be distorted.")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(ticker.replace(".NS", ""), fmt(t["current_price"], dash="–"))
            c2.metric("Signal score", f'{s["score"]:+.0f}', s["label"], delta_color="off")
            c3.metric("RSI (14)", fmt(t["rsi_14"]))
            c4.metric("Trend (50/200 DMA)", (t["ma_trend"] or "–").title())

            months = st.select_slider("Chart range (months)", options=[3, 6, 12, 18, 24], value=6)
            df = nc.chart_frame(hist, months)
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.58, 0.21, 0.21],
                                vertical_spacing=0.03, subplot_titles=("Price", "RSI", "MACD"))
            fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"],
                                         close=df["Close"], name="Price"), row=1, col=1)
            for col, name in (("MA50", "50 DMA"), ("MA200", "200 DMA")):
                fig.add_trace(go.Scatter(x=df.index, y=df[col], name=name, line=dict(width=1.5)), row=1, col=1)
            for col, name in (("BB_upper", "BB upper"), ("BB_lower", "BB lower")):
                fig.add_trace(go.Scatter(x=df.index, y=df[col], name=name,
                                         line=dict(width=1, dash="dot", color="gray")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", row=2, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Histogram"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal"), row=3, col=1)
            fig.update_layout(height=720, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=40, b=10),
                              legend=dict(orientation="h", y=1.06))
            st.plotly_chart(fig)

            st.subheader("Why this score?")
            comp = pd.DataFrame(s["components"]).rename(columns={"factor": "Factor", "points": "Points", "note": "Reading"})
            st.dataframe(comp, hide_index=True)

            left, right = st.columns(2)
            with left:
                st.subheader("Risk vs NIFTY 50")
                st.dataframe(pd.DataFrame({
                    "Metric": ["1M return %", "3M return %", "6M return %", "1Y return %", "Volatility (1Y) %",
                               "Max drawdown (1Y) %", "Current drawdown %", "Sharpe", "Sortino", "1-day VaR 95% %",
                               "Beta", "Alpha %", "Rel. strength 6M (pts)"],
                    "Value": [r["return_1m_pct"], r["return_3m_pct"], r["return_6m_pct"], r["return_1y_pct"],
                              r["volatility_1y_pct"], r["max_drawdown_1y_pct"], r["current_drawdown_pct"],
                              r["sharpe_1y"], r["sortino_1y"], r["var_95_1day_pct"], r["beta_vs_nifty"],
                              r["alpha_vs_nifty_pct"], r["relative_strength_6m_pct"]],
                }).astype({"Value": "object"}), hide_index=True)
            with right:
                st.subheader("Other indicators")
                pp = e["pivot_points"]
                st.dataframe(pd.DataFrame({
                    "Indicator": ["Stochastic %K / %D", "ATR(14) / % of price", "2×ATR stop", "Supertrend",
                                  "OBV", "Price vs 20D VWAP %", "Volume vs 20D avg", "ADX(14)", "Candle",
                                  "Pivot / S1 / R1", "52W low – high"],
                    "Value": [f'{fmt(e["stochastic_k"])} / {fmt(e["stochastic_d"])}',
                              f'{fmt(e["atr_14"])} / {fmt(e["atr_pct_of_price"], "%")}', fmt(e["atr_stop_2x"]),
                              f'{e["supertrend_direction"] or "–"} ({fmt(e["supertrend"])})',
                              e["obv_trend"] or "–", fmt(e["price_vs_vwap_pct"], "%"),
                              fmt(e["volume_ratio_vs_20d"], "×"), fmt(t["adx_14"]), e["candle_pattern"] or "–",
                              f'{fmt(pp["pivot"])} / {fmt(pp["s1"])} / {fmt(pp["r1"])}',
                              f'{fmt(e["low_52w"])} – {fmt(e["high_52w"])}'],
                }), hide_index=True)
        except Exception as ex:
            st.error(f"Could not analyze {ticker}: {ex}")

# ── Alerts ─────────────────────────────────────────────────────────────────
with tab_alerts:
    lookback = st.slider("Look back (trading days)", 1, 30, 5)
    if ticker:
        try:
            events = detect_events(get_hist(ticker), lookback)
            if not events:
                st.success("No notable events in this window.")
            icons = {"bullish": "🟢", "bearish": "🔴", "overbought": "🟠", "oversold": "🟠"}
            for ev in events:
                st.write(f'{icons.get(ev["direction"], "⚪")} **{ev["event"].replace("_", " ").title()}** — {ev["detail"]}')
        except Exception as ex:
            st.error(f"Could not compute alerts: {ex}")

# ── Fundamentals ───────────────────────────────────────────────────────────
with tab_fund:
    if ticker and st.button("Load fundamentals", help="Fetched on demand: Yahoo's financials endpoint is slower"):
        with st.spinner("Fetching financials…"):
            f = get_fund(ticker)
        m = st.columns(4)
        m[0].metric("Market cap (₹ Cr)", fmt(f["market_cap_cr"]))
        m[1].metric("P/E (TTM)", fmt(f["pe_ttm"]))
        m[2].metric("Price / Sales", fmt(f["price_to_sales_ttm"]))
        m[3].metric("Dividend yield", fmt(f["dividend_yield_pct"], "%"))
        m = st.columns(4)
        m[0].metric("YoY revenue growth", fmt(f["yoy_revenue_growth_pct"], "%"))
        m[1].metric("YoY earnings growth", fmt(f["yoy_earnings_growth_pct"], "%"))
        m[2].metric("Net margin", fmt(f["net_margin_pct"], "%"))
        m[3].metric("Operating margin", fmt(f["operating_margin_pct"], "%"))
        m = st.columns(3)
        m[0].metric("ROE (approx.)", fmt(f["roe_pct"], "%"))
        m[1].metric("Debt / Equity", fmt(f["debt_to_equity"]))
        m[2].metric("TTM revenue (₹ Cr)", fmt(f["ttm_revenue_cr"]))
        if f["quarters"]:
            st.subheader("Last 4 quarters (₹ Cr)")
            st.dataframe(pd.DataFrame(f["quarters"]), hide_index=True)
        if all(f[k] is None for k in ("market_cap_cr", "ttm_revenue_cr", "yoy_revenue_growth_pct")):
            st.warning("Yahoo returned no financials (common for banks/NBFCs or when rate-limited). Try again later.")

# ── Backtest ───────────────────────────────────────────────────────────────
with tab_bt:
    c1, c2, c3 = st.columns(3)
    strat = c1.selectbox("Strategy", ["ma_cross", "macd", "rsi"],
                         format_func=lambda x: {"ma_cross": "Moving-average crossover", "macd": "MACD crossover",
                                                "rsi": "RSI mean-reversion"}[x])
    years = c2.slider("Years of history", 1, 10, 5)
    cost = c3.number_input("Cost per side (%)", 0.0, 2.0, 0.1, 0.05)
    params = {}
    if strat == "ma_cross":
        p1, p2 = st.columns(2)
        params = {"fast": p1.number_input("Fast MA", 5, 100, 50), "slow": p2.number_input("Slow MA", 20, 300, 200)}
    elif strat == "rsi":
        p1, p2 = st.columns(2)
        params = {"buy_below": p1.number_input("Buy when RSI below", 5, 50, 30),
                  "exit_above": p2.number_input("Exit when RSI above", 40, 90, 55)}
    if ticker and st.button("Run backtest"):
        try:
            res = run_backtest(get_hist(ticker, f"{years}y")["Close"], strat, params, cost, include_curve=True)
            sm, bm = res["strategy_metrics"], res["buy_and_hold_metrics"]
            st.dataframe(pd.DataFrame({
                "": ["Total return %", "CAGR %", "Max drawdown %", "Sharpe"],
                "Strategy": [sm["total_return_pct"], sm["cagr_pct"], sm["max_drawdown_pct"], sm["sharpe"]],
                "Buy & hold": [bm["total_return_pct"], bm["cagr_pct"], bm["max_drawdown_pct"], bm["sharpe"]],
            }), hide_index=True)
            k = st.columns(4)
            k[0].metric("Trades", res["num_trades"])
            k[1].metric("Win rate", fmt(res["win_rate_pct"], "%"))
            k[2].metric("Time in market", fmt(res["time_in_market_pct"], "%"))
            k[3].metric("Latest signal", res["latest_signal"].upper())
            cur = pd.DataFrame(res["equity_curve"]).set_index("date")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cur.index, y=cur["strategy"], name="Strategy"))
            fig.add_trace(go.Scatter(x=cur.index, y=cur["buy_and_hold"], name="Buy & hold"))
            fig.update_layout(title="Growth of 100", height=380, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig)
            if res["recent_trades"]:
                st.caption("Most recent trades")
                st.dataframe(pd.DataFrame(res["recent_trades"]), hide_index=True)
            st.caption(res["note"])
        except Exception as ex:
            st.error(f"Backtest failed: {ex}")

# ── Compare ────────────────────────────────────────────────────────────────
with tab_cmp:
    raw = st.text_input("Symbols (2–5, comma-separated)", "TCS, INFY, HCLTECH")
    months = st.slider("Months", 1, 24, 12, key="cmp_months")
    if st.button("Compare"):
        syms = [nc.to_ticker(x) for x in raw.split(",") if x.strip()][:5]
        try:
            data = get_many(tuple(syms))
            rows, fig = [], go.Figure()
            for t_, h in data.items():
                close = h["Close"].tail(months * 21 + 1)
                fig.add_trace(go.Scatter(x=close.index, y=close / close.iloc[0] * 100, name=t_.replace(".NS", "")))
                from risk_metrics import daily_returns, annualized_volatility, max_drawdown, sharpe_ratio
                ret = daily_returns(close)
                sh = sharpe_ratio(ret, rf)
                rows.append({"Symbol": t_.replace(".NS", ""), "Return %": round(close.iloc[-1] / close.iloc[0] * 100 - 100, 2),
                             "Volatility %": round(float(annualized_volatility(ret)), 2),
                             "Max drawdown %": round(float(max_drawdown(close)[0]), 2),
                             "Sharpe": round(sh, 2) if sh is not None else None})
            fig.update_layout(title="Rebased to 100", height=420, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig)
            st.dataframe(pd.DataFrame(rows).sort_values("Return %", ascending=False), hide_index=True)
            missing = [x.replace(".NS", "") for x in syms if x not in data]
            if missing:
                st.warning(f"No data for: {', '.join(missing)}")
        except Exception as ex:
            st.error(f"Compare failed: {ex}")

# ── Screener ───────────────────────────────────────────────────────────────
with tab_scan:
    preset = st.selectbox("Universe", list(nc.PRESETS) + ["Custom list"])
    custom = ""
    if preset == "Custom list":
        custom = st.text_area("Symbols, comma-separated (max 50)", "RELIANCE, TCS, INFY, SBIN")
    f1, f2, f3, f4 = st.columns(4)
    min_score = f1.number_input("Min score", -100, 100, 20)
    trend_f = f2.selectbox("Trend", ["any", "bullish", "bearish"])
    rsi_rng = f3.slider("RSI range", 0, 100, (0, 100))
    min_vol = f4.number_input("Min volume ratio", 0.0, 10.0, 0.0, 0.1)
    if st.button("Run screener"):
        syms = ([x.strip() for x in custom.split(",") if x.strip()] if preset == "Custom list" else nc.PRESETS[preset])[:50]
        tickers = tuple(nc.to_ticker(x) for x in syms)
        try:
            with st.spinner(f"Scanning {len(tickers)} stocks…"):
                data = get_many(tickers)
                bench = get_bench()
                rows, errs = [], []
                for tk in tickers:
                    if tk not in data:
                        errs.append(tk.replace(".NS", ""))
                        continue
                    try:
                        rows.append(nc.screen_row(tk, data[tk], bench))
                    except Exception:
                        errs.append(tk.replace(".NS", ""))
            df = pd.DataFrame(rows)
            if df.empty:
                st.warning("No data came back (Yahoo may be rate-limiting). Wait a minute and retry.")
            else:
                m = (df["Score"] >= min_score) & df["RSI"].between(*rsi_rng) & (df["Vol ratio"].fillna(0) >= min_vol)
                if trend_f != "any":
                    m &= df["Trend"] == trend_f
                out = df[m].sort_values("Score", ascending=False)
                st.success(f"{len(out)} of {len(df)} scanned stocks match")
                st.dataframe(out, hide_index=True)
                if errs:
                    st.caption(f"No data for: {', '.join(errs)}")
        except Exception as ex:
            st.error(f"Screener failed: {ex}")
