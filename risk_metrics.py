"""
risk_metrics.py — volatility, drawdown, Sharpe/Sortino, beta/alpha vs NIFTY 50,
VaR and relative strength. Pure pandas/numpy.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _r(val, d=2):
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return round(float(val), d)


def daily_returns(close):
    return close.pct_change().dropna()


def annualized_volatility(returns):
    return returns.std() * np.sqrt(TRADING_DAYS) * 100


def max_drawdown(close):
    """Returns (max drawdown %, trough date as str). Value is negative."""
    dd = close / close.cummax() - 1
    trough = dd.idxmin()
    return dd.min() * 100, str(trough.date()) if hasattr(trough, "date") else str(trough)


def sharpe_ratio(returns, rf_annual=0.07):
    excess = returns - rf_annual / TRADING_DAYS
    sd = excess.std()
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS)) if sd and sd > 0 else None


def sortino_ratio(returns, rf_annual=0.07):
    excess = returns - rf_annual / TRADING_DAYS
    downside = excess[excess < 0].std()
    return float(excess.mean() / downside * np.sqrt(TRADING_DAYS)) if downside and downside > 0 else None


def historical_var(returns, confidence=0.95):
    """One-day historical VaR as a positive % loss."""
    return -np.percentile(returns, (1 - confidence) * 100) * 100


def period_return(close, days):
    if len(close) <= days:
        return None
    return (close.iloc[-1] / close.iloc[-1 - days] - 1) * 100


def beta_alpha(stock_ret, bench_ret, rf_annual=0.07):
    df = pd.concat([stock_ret, bench_ret], axis=1, join="inner").dropna()
    if len(df) < 60:
        return None, None, None
    s, b = df.iloc[:, 0], df.iloc[:, 1]
    var_b = b.var()
    if var_b == 0:
        return None, None, None
    beta = s.cov(b) / var_b
    rf_d = rf_annual / TRADING_DAYS
    alpha_daily = (s.mean() - rf_d) - beta * (b.mean() - rf_d)
    corr = s.corr(b)
    return float(beta), float(alpha_daily * TRADING_DAYS * 100), float(corr)


def risk_snapshot(hist, bench_hist=None, rf_annual=0.07):
    close = hist["Close"]
    ret = daily_returns(close)
    last_year = ret.tail(TRADING_DAYS)
    dd, dd_date = max_drawdown(close.tail(TRADING_DAYS))
    cur_dd = (close.iloc[-1] / close.tail(TRADING_DAYS).max() - 1) * 100

    out = {
        "return_1m_pct": _r(period_return(close, 21)),
        "return_3m_pct": _r(period_return(close, 63)),
        "return_6m_pct": _r(period_return(close, 126)),
        "return_1y_pct": _r(period_return(close, 252)),
        "volatility_1y_pct": _r(annualized_volatility(last_year)),
        "max_drawdown_1y_pct": _r(dd),
        "max_drawdown_1y_trough": dd_date,
        "current_drawdown_pct": _r(cur_dd),
        "sharpe_1y": _r(sharpe_ratio(last_year, rf_annual)),
        "sortino_1y": _r(sortino_ratio(last_year, rf_annual)),
        "var_95_1day_pct": _r(historical_var(last_year)),
        "beta_vs_nifty": None,
        "alpha_vs_nifty_pct": None,
        "correlation_vs_nifty": None,
        "relative_strength_3m_pct": None,
        "relative_strength_6m_pct": None,
        "risk_free_rate_assumed": rf_annual,
    }

    if bench_hist is not None and not bench_hist.empty:
        bclose = bench_hist["Close"]
        beta, alpha, corr = beta_alpha(last_year, daily_returns(bclose).tail(TRADING_DAYS), rf_annual)
        out["beta_vs_nifty"] = _r(beta)
        out["alpha_vs_nifty_pct"] = _r(alpha)
        out["correlation_vs_nifty"] = _r(corr)
        for label, days in (("3m", 63), ("6m", 126)):
            sr, br = period_return(close, days), period_return(bclose, days)
            if sr is not None and br is not None:
                out[f"relative_strength_{label}_pct"] = _r(sr - br)  # outperformance in % points
    return out
