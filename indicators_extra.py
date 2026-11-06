"""
indicators_extra.py — additional technical indicators + event detection
Pure pandas/numpy. No network calls, no dependency on the main API file.
"""
import numpy as np
import pandas as pd


def _r(val, d=2):
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return round(float(val), d)


# ── Indicators ────────────────────────────────────────────────────────────

def calculate_stochastic(high, low, close, k_period=14, d_period=3, smooth=3):
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    raw_k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    k = raw_k.rolling(smooth).mean()
    d = k.rolling(d_period).mean()
    return k, d


def calculate_atr(high, low, close, period=14):
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def calculate_obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def calculate_rolling_vwap(high, low, close, volume, window=20):
    tp = (high + low + close) / 3
    return (tp * volume).rolling(window).sum() / volume.rolling(window).sum().replace(0, np.nan)


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Returns (supertrend_line, direction) where direction is +1 (uptrend) / -1 (downtrend)."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    ub = (hl2 + multiplier * atr).values
    lb = (hl2 - multiplier * atr).values
    c = close.values
    n = len(c)
    fu, fl = ub.copy(), lb.copy()
    direction = np.ones(n)
    st = np.full(n, np.nan)
    for i in range(1, n):
        if np.isnan(ub[i]) or np.isnan(ub[i - 1]):
            direction[i] = direction[i - 1]
            continue
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if direction[i - 1] == 1 and c[i] < fl[i]:
            direction[i] = -1
        elif direction[i - 1] == -1 and c[i] > fu[i]:
            direction[i] = 1
        else:
            direction[i] = direction[i - 1]
        st[i] = fl[i] if direction[i] == 1 else fu[i]
    return pd.Series(st, index=close.index), pd.Series(direction, index=close.index)


def calculate_pivot_points(high, low, close):
    """Classic floor-trader pivots from the LAST completed session (levels for the next session)."""
    h, l, c = float(high.iloc[-1]), float(low.iloc[-1]), float(close.iloc[-1])
    p = (h + l + c) / 3
    return {
        "pivot": _r(p),
        "r1": _r(2 * p - l), "r2": _r(p + (h - l)), "r3": _r(h + 2 * (p - l)),
        "s1": _r(2 * p - h), "s2": _r(p - (h - l)), "s3": _r(l - 2 * (h - p)),
    }


def detect_candle_pattern(hist):
    """Very basic single/two-candle pattern on the latest bar."""
    if len(hist) < 2:
        return None
    o, h, l, c = (float(hist[k].iloc[-1]) for k in ("Open", "High", "Low", "Close"))
    po, pc = float(hist["Open"].iloc[-2]), float(hist["Close"].iloc[-2])
    rng = h - l
    if rng <= 0:
        return None
    body = abs(c - o)
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    if body / rng < 0.1:
        return "doji"
    if lower_wick > 2 * body and upper_wick < body:
        return "hammer"
    if upper_wick > 2 * body and lower_wick < body:
        return "shooting_star"
    if c > o and pc < po and c >= po and o <= pc:
        return "bullish_engulfing"
    if c < o and pc > po and o >= pc and c <= po:
        return "bearish_engulfing"
    return None


# ── Snapshot ──────────────────────────────────────────────────────────────

def compute_extra_snapshot(hist):
    close, high, low, vol = hist["Close"], hist["High"], hist["Low"], hist["Volume"]

    k, d = calculate_stochastic(high, low, close)
    atr = calculate_atr(high, low, close)
    obv = calculate_obv(close, vol)
    obv_sma = obv.rolling(20).mean()
    vwap = calculate_rolling_vwap(high, low, close, vol)
    st_line, st_dir = calculate_supertrend(high, low, close)
    vol_ratio = vol / vol.rolling(20).mean().replace(0, np.nan)

    price = float(close.iloc[-1])
    obv_trend = None
    if not pd.isna(obv_sma.iloc[-1]):
        obv_trend = "accumulation" if obv.iloc[-1] > obv_sma.iloc[-1] else "distribution"

    return {
        "stochastic_k": _r(k.iloc[-1]),
        "stochastic_d": _r(d.iloc[-1]),
        "atr_14": _r(atr.iloc[-1]),
        "atr_pct_of_price": _r(atr.iloc[-1] / price * 100),
        "obv_trend": obv_trend,
        "vwap_20": _r(vwap.iloc[-1]),
        "price_vs_vwap_pct": _r((price / vwap.iloc[-1] - 1) * 100) if not pd.isna(vwap.iloc[-1]) else None,
        "supertrend": _r(st_line.iloc[-1]),
        "supertrend_direction": ("up" if st_dir.iloc[-1] == 1 else "down") if not pd.isna(st_line.iloc[-1]) else None,
        "volume_ratio_vs_20d": _r(vol_ratio.iloc[-1]),
        "high_20d": _r(high.tail(20).max()),
        "low_20d": _r(low.tail(20).min()),
        "high_52w": _r(high.tail(252).max()),
        "low_52w": _r(low.tail(252).min()),
        "pivot_points": calculate_pivot_points(high, low, close),
        "candle_pattern": detect_candle_pattern(hist),
        # ATR-based stop idea: 2x ATR below price (informational, not advice)
        "atr_stop_2x": _r(price - 2 * atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None,
    }


# ── Event / alert detection ───────────────────────────────────────────────

def _has_sign_change(diff, direction):
    prev = diff.shift(1)
    if direction == "up":
        return bool(((prev <= 0) & (diff > 0)).any())
    return bool(((prev >= 0) & (diff < 0)).any())


def detect_events(hist, lookback=5):
    """Returns a list of dicts: {event, direction, detail}. Empty list = nothing fired."""
    close, high, low, vol = hist["Close"], hist["High"], hist["Low"], hist["Volume"]
    events = []

    def add(event, direction, detail):
        events.append({"event": event, "direction": direction, "detail": detail})

    ma50, ma200 = close.rolling(50).mean(), close.rolling(200).mean()
    d = (ma50 - ma200).dropna().tail(lookback + 1)
    if _has_sign_change(d, "up"):
        add("golden_cross", "bullish", f"50-DMA crossed above 200-DMA within last {lookback} sessions")
    if _has_sign_change(d, "down"):
        add("death_cross", "bearish", f"50-DMA crossed below 200-DMA within last {lookback} sessions")

    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rsi = (100 - 100 / (1 + gain / loss)).dropna()
    if len(rsi) > lookback:
        r = rsi.tail(lookback + 1)
        if ((r.shift(1) < 30) & (r >= 30)).any():
            add("rsi_exit_oversold", "bullish", "RSI crossed back above 30")
        if ((r.shift(1) > 70) & (r <= 70)).any():
            add("rsi_exit_overbought", "bearish", "RSI crossed back below 70")
        if r.iloc[-1] < 30:
            add("rsi_oversold", "bullish", f"RSI at {r.iloc[-1]:.1f}")
        if r.iloc[-1] > 70:
            add("rsi_overbought", "bearish", f"RSI at {r.iloc[-1]:.1f}")

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    md = (macd - sig).tail(lookback + 1)
    if _has_sign_change(md, "up"):
        add("macd_bullish_crossover", "bullish", "MACD crossed above signal line")
    if _has_sign_change(md, "down"):
        add("macd_bearish_crossover", "bearish", "MACD crossed below signal line")

    if len(close) > 21:
        prior_high = high.iloc[-21:-1].max()
        prior_low = low.iloc[-21:-1].min()
        if close.iloc[-1] > prior_high:
            add("breakout_20d_high", "bullish", f"Close above prior 20-day high {prior_high:.2f}")
        if close.iloc[-1] < prior_low:
            add("breakdown_20d_low", "bearish", f"Close below prior 20-day low {prior_low:.2f}")

        avg_vol = vol.iloc[-21:-1].mean()
        if avg_vol > 0 and vol.iloc[-1] > 2 * avg_vol:
            day_chg = (close.iloc[-1] / close.iloc[-2] - 1) * 100
            add("volume_spike", "bullish" if day_chg > 0 else "bearish",
                f"Volume {vol.iloc[-1] / avg_vol:.1f}x the 20-day average on a {day_chg:+.1f}% day")

    mid = close.rolling(20).mean()
    sd = close.rolling(20).std()
    if not pd.isna(sd.iloc[-1]):
        if close.iloc[-1] > mid.iloc[-1] + 2 * sd.iloc[-1]:
            add("bollinger_upper_break", "overbought", "Close above upper Bollinger band")
        if close.iloc[-1] < mid.iloc[-1] - 2 * sd.iloc[-1]:
            add("bollinger_lower_break", "oversold", "Close below lower Bollinger band")

    hi52 = high.tail(252).max()
    if close.iloc[-1] >= 0.98 * hi52:
        add("near_52w_high", "bullish", f"Within 2% of 52-week high ({hi52:.2f})")
    lo52 = low.tail(252).min()
    if close.iloc[-1] <= 1.05 * lo52:
        add("near_52w_low", "bearish", f"Within 5% of 52-week low ({lo52:.2f})")

    _, st_dir = calculate_supertrend(high, low, close)
    if len(st_dir) > lookback + 1:
        recent = st_dir.tail(lookback + 1)
        if recent.iloc[-1] == 1 and (recent == -1).any():
            add("supertrend_flip_up", "bullish", "Supertrend flipped to uptrend")
        if recent.iloc[-1] == -1 and (recent == 1).any():
            add("supertrend_flip_down", "bearish", "Supertrend flipped to downtrend")

    return events
