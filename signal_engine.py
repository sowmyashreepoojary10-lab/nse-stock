"""
signal_engine.py — turns the indicator dicts into one composite score (-100..+100)
with a plain-English breakdown. Rule-based and transparent: every point added or
removed is listed in `components`, so you can see exactly why a stock scored as it did.

This is a screening aid, NOT investment advice.
"""

WEIGHTS = {  # tweak these to change how much each factor matters
    "trend": 25, "momentum": 20, "macd": 15, "supertrend": 10,
    "volume_flow": 10, "bollinger": 5, "stochastic": 5, "relative_strength": 10,
}


def _add(components, factor, points, note):
    components.append({"factor": factor, "points": round(points, 1), "note": note})
    return points


def label_for(score):
    if score >= 50:
        return "Strong Buy"
    if score >= 20:
        return "Buy"
    if score > -20:
        return "Neutral"
    if score > -50:
        return "Sell"
    return "Strong Sell"


def score_signals(technical, extra, risk=None):
    c = []
    total = 0.0
    w = WEIGHTS
    adx = technical.get("adx_14")
    strong_trend = adx is not None and adx >= 25
    price = technical.get("current_price")

    # 1) Trend: 50/200 DMA + price vs 50 DMA; amplified when ADX says the trend is real
    t = 0.0
    if technical.get("ma_trend") == "bullish":
        t += 0.6
    elif technical.get("ma_trend") == "bearish":
        t -= 0.6
    if price and technical.get("ma_50"):
        t += 0.4 if price > technical["ma_50"] else -0.4
    t *= w["trend"] * (1.0 if strong_trend else 0.7)
    total += _add(c, "trend", t, f"MA trend {technical.get('ma_trend')}, ADX {adx} "
                                 f"({'strong' if strong_trend else 'weak'} trend)")

    # 2) Momentum: RSI (mean-reversion at extremes, reward healthy 50-70 zone)
    rsi = technical.get("rsi_14")
    m = 0.0
    if rsi is not None:
        if rsi < 30:
            m = 0.6
        elif rsi < 45:
            m = -0.3
        elif rsi < 50:
            m = 0.0
        elif rsi <= 70:
            m = 0.7
        elif rsi <= 80:
            m = -0.3
        else:
            m = -0.7
    total += _add(c, "momentum", m * w["momentum"], f"RSI {rsi}")

    # 3) MACD histogram
    h = technical.get("macd_histogram")
    mc = 0.0
    if h is not None:
        mc = 1.0 if h > 0 else -1.0
    total += _add(c, "macd", mc * w["macd"], f"MACD histogram {h}")

    # 4) Supertrend
    sd = extra.get("supertrend_direction")
    total += _add(c, "supertrend", (w["supertrend"] if sd == "up" else -w["supertrend"] if sd == "down" else 0),
                  f"Supertrend {sd}")

    # 5) Volume flow (OBV) — accumulation vs distribution
    obv = extra.get("obv_trend")
    total += _add(c, "volume_flow", (w["volume_flow"] if obv == "accumulation"
                                     else -w["volume_flow"] if obv == "distribution" else 0),
                  f"OBV shows {obv}")

    # 6) Bollinger %B — outside the bands is stretched
    pb = technical.get("bollinger_percent_b")
    b = 0.0
    if pb is not None:
        b = 1.0 if pb < 0 else -1.0 if pb > 1 else 0.0
    total += _add(c, "bollinger", b * w["bollinger"], f"%B {pb}")

    # 7) Stochastic
    k = extra.get("stochastic_k")
    s = 0.0
    if k is not None:
        s = 1.0 if k < 20 else -1.0 if k > 80 else 0.0
    total += _add(c, "stochastic", s * w["stochastic"], f"Stochastic %K {k}")

    # 8) Relative strength vs NIFTY (6m outperformance in % points)
    rs = (risk or {}).get("relative_strength_6m_pct")
    r = 0.0
    if rs is not None:
        r = max(-1.0, min(1.0, rs / 20.0))  # +/-20 pts of outperformance = full score
    total += _add(c, "relative_strength", r * w["relative_strength"],
                  f"6m vs NIFTY: {rs} pts" if rs is not None else "benchmark unavailable")

    total = max(-100.0, min(100.0, total))
    bullish = [x["factor"] for x in c if x["points"] > 0]
    bearish = [x["factor"] for x in c if x["points"] < 0]
    return {
        "score": round(total, 1),
        "label": label_for(total),
        "bullish_factors": bullish,
        "bearish_factors": bearish,
        "components": c,
        "disclaimer": "Rule-based screening output, not investment advice.",
    }
