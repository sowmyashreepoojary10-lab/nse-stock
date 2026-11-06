"""
backtest.py — simple long-only, vectorised backtester for three classic strategies.

Rules that keep it honest:
  * Signals are computed on day t's close and traded on day t+1 (no look-ahead).
  * A transaction cost (default 0.1% per side) is charged on every position change.
  * Compared against buy-and-hold over the same window.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _r(v, d=2):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return round(float(v), d)


def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def build_position(close, strategy, params):
    """Returns a 0/1 Series: desired position at end of each day."""
    if strategy == "ma_cross":
        fast, slow = int(params.get("fast", 50)), int(params.get("slow", 200))
        if fast >= slow:
            raise ValueError("fast must be smaller than slow")
        pos = (close.rolling(fast).mean() > close.rolling(slow).mean()).astype(float)
        pos[close.rolling(slow).mean().isna()] = 0.0
        return pos

    if strategy == "macd":
        macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        sig = macd.ewm(span=9, adjust=False).mean()
        pos = (macd > sig).astype(float)
        pos.iloc[:35] = 0.0
        return pos

    if strategy == "rsi":
        buy_below = float(params.get("buy_below", 30))
        exit_above = float(params.get("exit_above", 55))
        rsi = _rsi(close).values
        pos = np.zeros(len(close))
        holding = 0.0
        for i in range(len(close)):
            if np.isnan(rsi[i]):
                pos[i] = 0.0
                continue
            if holding == 0.0 and rsi[i] < buy_below:
                holding = 1.0
            elif holding == 1.0 and rsi[i] > exit_above:
                holding = 0.0
            pos[i] = holding
        return pd.Series(pos, index=close.index)

    raise ValueError(f"Unknown strategy '{strategy}'. Use ma_cross, macd or rsi.")


def _metrics(equity, daily_ret):
    n = len(equity)
    years = max(n / TRADING_DAYS, 1e-9)
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    dd = (equity / equity.cummax() - 1).min()
    sd = daily_ret.std()
    sharpe = daily_ret.mean() / sd * np.sqrt(TRADING_DAYS) if sd and sd > 0 else None
    return {
        "total_return_pct": _r(total * 100),
        "cagr_pct": _r(cagr * 100),
        "max_drawdown_pct": _r(dd * 100),
        "sharpe": _r(sharpe),
    }


def _trades(close, position):
    """List of completed + open trades from the lagged position series."""
    trades = []
    entry_price, entry_date = None, None
    prev = 0.0
    for date, p in position.items():
        price = float(close.loc[date])
        if p == 1.0 and prev == 0.0:
            entry_price, entry_date = price, date
        elif p == 0.0 and prev == 1.0 and entry_price:
            trades.append({"entry": str(entry_date.date()), "exit": str(date.date()),
                           "return_pct": _r((price / entry_price - 1) * 100), "open": False})
            entry_price = None
        prev = p
    if entry_price:
        last = float(close.iloc[-1])
        trades.append({"entry": str(entry_date.date()), "exit": None,
                       "return_pct": _r((last / entry_price - 1) * 100), "open": True})
    return trades


def run_backtest(close, strategy="ma_cross", params=None, cost_pct=0.1, include_curve=False):
    params = params or {}
    close = close.dropna()
    if len(close) < 60:
        raise ValueError("Not enough price history to backtest.")

    desired = build_position(close, strategy, params)
    position = desired.shift(1).fillna(0.0)          # trade next day -> no look-ahead
    ret = close.pct_change().fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    strat_ret = position * ret - turnover * (cost_pct / 100)

    equity = (1 + strat_ret).cumprod()
    bh_equity = (1 + ret).cumprod()

    trades = _trades(close, position)
    closed = [t for t in trades if not t["open"]]
    wins = [t for t in closed if t["return_pct"] is not None and t["return_pct"] > 0]

    result = {
        "strategy": strategy,
        "params": params,
        "cost_pct_per_side": cost_pct,
        "period": {"start": str(close.index[0].date()), "end": str(close.index[-1].date()),
                   "trading_days": int(len(close))},
        "strategy_metrics": _metrics(equity, strat_ret),
        "buy_and_hold_metrics": _metrics(bh_equity, ret),
        "time_in_market_pct": _r(position.mean() * 100),
        "num_trades": len(trades),
        "win_rate_pct": _r(len(wins) / len(closed) * 100) if closed else None,
        "avg_trade_return_pct": _r(np.mean([t["return_pct"] for t in closed])) if closed else None,
        "currently_in_position": bool(position.iloc[-1] == 1.0),
        "latest_signal": "long" if desired.iloc[-1] == 1.0 else "flat",
        "recent_trades": trades[-10:],
        "note": "Past performance does not predict future results. Costs modelled; slippage and taxes are not.",
    }
    if include_curve:
        weekly = pd.DataFrame({"strategy": equity, "buy_and_hold": bh_equity}).iloc[::5]
        result["equity_curve"] = [
            {"date": str(d.date()), "strategy": _r(r["strategy"] * 100, 2), "buy_and_hold": _r(r["buy_and_hold"] * 100, 2)}
            for d, r in weekly.iterrows()
        ]
    return result
