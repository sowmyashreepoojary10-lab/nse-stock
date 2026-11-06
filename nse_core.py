"""
nse_core.py — all the analysis logic, with NO web-framework dependency.
(Your original FastAPI indicator + fundamentals code, merged with the new modules.)
Used by streamlit_app.py; also importable from a notebook.
"""
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from indicators_extra import compute_extra_snapshot, detect_events
from risk_metrics import risk_snapshot
from signal_engine import score_signals

BENCHMARK = "^NSEI"  # NIFTY 50

PRESETS = {
    "Nifty large caps": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC", "SBIN",
                         "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "SUNPHARMA",
                         "TITAN", "ULTRACEMCO", "NESTLEIND", "BAJFINANCE", "WIPRO", "HCLTECH", "NTPC",
                         "POWERGRID", "ONGC", "COALINDIA", "TECHM", "JSWSTEEL", "TATASTEEL", "HINDALCO", "CIPLA"],
    "Banks": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK", "BANKBARODA", "PNB"],
    "IT": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "PERSISTENT", "COFORGE"],
}


# ── helpers ───────────────────────────────────────────────────────────────

def to_ticker(symbol):
    s = symbol.upper().strip()
    if s.startswith("^") or s.endswith(".NS") or s.endswith(".BO"):
        return s
    return f"{s}.NS"


def safe_round(val, decimals=2):
    if val is None or pd.isna(val):
        return None
    return round(float(val), decimals)


def fetch_history(ticker, period="2y", retries=3):
    """Download price history with a small retry/backoff (Yahoo rate-limits shared cloud IPs)."""
    last_err = None
    for attempt in range(retries):
        try:
            hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if hist is not None and not hist.empty:
                return hist
            last_err = ValueError(f"No data found for {ticker}. Check the symbol.")
        except Exception as e:  # network / rate limit
            last_err = e
        time.sleep(1.5 * (attempt + 1))
    raise ValueError(str(last_err))


def fetch_many(tickers, period="2y", retries=3):
    """One batched Yahoo call for many tickers (far fewer requests -> far fewer rate-limit errors).
    Returns {ticker: DataFrame}; tickers with no data are simply missing."""
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return {}
    raw, last_err = None, None
    for attempt in range(retries):
        try:
            raw = yf.download(tickers, period=period, group_by="ticker", auto_adjust=True,
                              threads=True, progress=False)
            if raw is not None and not raw.empty:
                break
        except Exception as e:
            last_err = e
        time.sleep(1.5 * (attempt + 1))
    if raw is None or raw.empty:
        raise ValueError(str(last_err or "No data returned from Yahoo Finance."))
    return split_download(raw, tickers)


def split_download(raw, tickers):
    out = {}
    cols = ["Open", "High", "Low", "Close", "Volume"]
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if t in raw.columns.get_level_values(0):
                df = raw[t].dropna(how="all")
                if not df.empty and all(c in df.columns for c in cols):
                    out[t] = df[cols]
    else:  # single ticker comes back flat
        df = raw.dropna(how="all")
        if not df.empty:
            out[tickers[0]] = df[cols]
    return out


# ── original indicators (unchanged logic) ─────────────────────────────────

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_roc(prices, period=12):
    return prices.pct_change(period) * 100


def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def calculate_moving_averages(prices, short_window=50, long_window=200):
    return prices.rolling(short_window).mean(), prices.rolling(long_window).mean()


def calculate_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm - minus_dm) < 0] = 0
    minus_dm[(minus_dm - plus_dm) < 0] = 0
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def calculate_52w_high_proximity(prices, window=252):
    return (prices / prices.rolling(window).max()) * 100


def calculate_bollinger_bands(prices, window=20, num_std=2):
    middle = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return (prices - lower) / (upper - lower), (upper - lower) / middle * 100


def get_technical_snapshot(hist):
    close, high, low = hist["Close"], hist["High"], hist["Low"]
    corporate_action_flag = bool((close.pct_change().abs() > 0.15).any())

    rsi = calculate_rsi(close)
    roc = calculate_roc(close)
    macd_line, signal_line, macd_hist = calculate_macd(close)
    ma50, ma200 = calculate_moving_averages(close)
    adx = calculate_adx(high, low, close)
    high52 = calculate_52w_high_proximity(close)
    percent_b, bandwidth = calculate_bollinger_bands(close)

    ma_trend = None
    if not pd.isna(ma50.iloc[-1]) and not pd.isna(ma200.iloc[-1]):
        ma_trend = "bullish" if ma50.iloc[-1] > ma200.iloc[-1] else "bearish"

    technical = {
        "current_price": safe_round(close.iloc[-1]),
        "rsi_14": safe_round(rsi.iloc[-1]),
        "roc_12": safe_round(roc.iloc[-1]),
        "macd_line": safe_round(macd_line.iloc[-1]),
        "macd_signal": safe_round(signal_line.iloc[-1]),
        "macd_histogram": safe_round(macd_hist.iloc[-1]),
        "ma_50": safe_round(ma50.iloc[-1]),
        "ma_200": safe_round(ma200.iloc[-1]),
        "ma_trend": ma_trend,
        "adx_14": safe_round(adx.iloc[-1]),
        "pct_of_52w_high": safe_round(high52.iloc[-1]),
        "bollinger_percent_b": safe_round(percent_b.iloc[-1], 3),
        "bollinger_bandwidth_pct": safe_round(bandwidth.iloc[-1]),
    }
    return technical, corporate_action_flag


def chart_frame(hist, months=6):
    """OHLCV + indicator columns for plotting (computed on full history, then sliced)."""
    close = hist["Close"]
    macd_line, signal_line, macd_hist = calculate_macd(close)
    ma50, ma200 = calculate_moving_averages(close)
    upper_mid = close.rolling(20).mean()
    sd = close.rolling(20).std()
    df = hist.copy()
    df["RSI"] = calculate_rsi(close)
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd_line, signal_line, macd_hist
    df["MA50"], df["MA200"] = ma50, ma200
    df["BB_upper"], df["BB_lower"] = upper_mid + 2 * sd, upper_mid - 2 * sd
    return df.tail(max(1, int(months * 21)))


# ── fundamentals (original logic) ─────────────────────────────────────────

def get_yoy_growth(series, tolerance_days=45):
    """Date-matched YoY growth (series sorted newest first)."""
    if len(series) < 2:
        return None, None
    latest_date, latest_val = series.index[0], series.iloc[0]
    target = latest_date - pd.DateOffset(months=12)
    diffs = (series.index - target).days.to_series(index=series.index).abs()
    idx = diffs.idxmin()
    if diffs[idx] > tolerance_days:
        return None, None
    base = series.loc[idx]
    if base == 0 or pd.isna(base):
        return None, None
    return round(((latest_val / base) - 1) * 100, 2), str(idx.date())


def get_fundamentals(stock):
    f = {"yoy_earnings_growth_pct": None, "yoy_revenue_growth_pct": None,
         "operating_margin_pct": None, "net_margin_pct": None, "roe_pct": None,
         "debt_to_equity": None, "market_cap_cr": None,
         "ttm_revenue_cr": None, "ttm_net_income_cr": None, "pe_ttm": None,
         "price_to_sales_ttm": None, "dividend_yield_pct": None, "quarters": []}
    latest_rev = latest_ni = latest_op = None
    try:
        inc = stock.quarterly_income_stmt
        if inc is not None and not inc.empty:
            def row(name):
                return inc.loc[name].dropna().sort_index(ascending=False) if name in inc.index else None
            ni, rev, op = row("Net Income"), row("Total Revenue"), row("Operating Income")
            if ni is not None and len(ni):
                f["yoy_earnings_growth_pct"], _ = get_yoy_growth(ni)
                latest_ni = ni.iloc[0]
                if len(ni) >= 4:
                    f["ttm_net_income_cr"] = safe_round(ni.iloc[:4].sum() / 1e7)
            if rev is not None and len(rev):
                f["yoy_revenue_growth_pct"], _ = get_yoy_growth(rev)
                latest_rev = rev.iloc[0]
                if len(rev) >= 4:
                    f["ttm_revenue_cr"] = safe_round(rev.iloc[:4].sum() / 1e7)
                for d in rev.index[:4]:
                    q = {"quarter_end": str(d.date()), "revenue_cr": safe_round(rev[d] / 1e7)}
                    if ni is not None and d in ni.index:
                        q["net_income_cr"] = safe_round(ni[d] / 1e7)
                    f["quarters"].append(q)
            if op is not None and len(op):
                latest_op = op.iloc[0]
            if latest_rev:
                if latest_ni is not None:
                    f["net_margin_pct"] = safe_round(latest_ni / latest_rev * 100)
                if latest_op is not None:
                    f["operating_margin_pct"] = safe_round(latest_op / latest_rev * 100)
    except Exception:
        pass
    try:
        bs = stock.quarterly_balance_sheet
        if bs is not None and not bs.empty:
            equity = debt = None
            for lab in ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]:
                if lab in bs.index and len(bs.loc[lab].dropna()):
                    equity = bs.loc[lab].dropna().iloc[0]
                    break
            for lab in ["Total Debt", "Net Debt"]:
                if lab in bs.index and len(bs.loc[lab].dropna()):
                    debt = bs.loc[lab].dropna().iloc[0]
                    break
            if equity:
                if debt is not None:
                    f["debt_to_equity"] = safe_round(debt / equity)
                if latest_ni is not None:
                    f["roe_pct"] = safe_round(latest_ni * 4 / equity * 100)
    except Exception:
        pass
    try:
        mcap = stock.fast_info.get("marketCap")
        if mcap:
            f["market_cap_cr"] = safe_round(mcap / 1e7)
            if f["ttm_net_income_cr"] and f["ttm_net_income_cr"] > 0:
                f["pe_ttm"] = safe_round(f["market_cap_cr"] / f["ttm_net_income_cr"])
            if f["ttm_revenue_cr"]:
                f["price_to_sales_ttm"] = safe_round(f["market_cap_cr"] / f["ttm_revenue_cr"])
    except Exception:
        pass
    try:
        divs = stock.dividends
        price = stock.fast_info.get("lastPrice")
        if divs is not None and len(divs) and price:
            cutoff = divs.index.max() - pd.DateOffset(months=12)
            f["dividend_yield_pct"] = safe_round(float(divs[divs.index > cutoff].sum()) / price * 100)
    except Exception:
        pass
    return f


# ── one-stop analysis ─────────────────────────────────────────────────────

def analyze(hist, bench_hist=None, rf=0.07):
    """Everything computed from price history (no extra network calls)."""
    technical, corp_flag = get_technical_snapshot(hist)
    extra = compute_extra_snapshot(hist)
    risk = risk_snapshot(hist, bench_hist, rf)
    signal = score_signals(technical, extra, risk)
    return {"technical": technical, "extra": extra, "risk": risk, "signal": signal,
            "corporate_action_flag": corp_flag}


def screen_row(ticker, hist, bench_hist):
    a = analyze(hist, bench_hist)
    t, e, r, s = a["technical"], a["extra"], a["risk"], a["signal"]
    return {
        "Symbol": ticker.replace(".NS", ""), "Price": t["current_price"], "Score": s["score"], "Signal": s["label"],
        "RSI": t["rsi_14"], "Trend": t["ma_trend"], "ADX": t["adx_14"], "% of 52W high": t["pct_of_52w_high"],
        "Vol ratio": e["volume_ratio_vs_20d"], "3M ret %": r["return_3m_pct"], "RS 6M (pts)": r["relative_strength_6m_pct"],
        "Split/demerger?": a["corporate_action_flag"],
    }
