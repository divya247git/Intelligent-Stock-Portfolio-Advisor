"""
technical_indicators.py
========================
Computes standard technical indicators and derives a bounded technical
"score" in [-1, 1] used by the recommendation engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds common indicators as new columns to an OHLCV dataframe."""
    out = df.copy()
    close, high, low, volume = out["close"], out["high"], out["low"], out["volume"]

    out["sma_20"] = ta.trend.sma_indicator(close, window=20)
    out["sma_50"] = ta.trend.sma_indicator(close, window=50)
    out["sma_200"] = ta.trend.sma_indicator(close, window=200)
    out["ema_12"] = ta.trend.ema_indicator(close, window=12)
    out["ema_26"] = ta.trend.ema_indicator(close, window=26)

    macd = ta.trend.MACD(close)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"] = macd.macd_diff()

    out["rsi_14"] = ta.momentum.rsi(close, window=14)

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    out["bb_high"] = bb.bollinger_hband()
    out["bb_low"] = bb.bollinger_lband()
    out["bb_pct"] = bb.bollinger_pband()

    out["atr_14"] = ta.volatility.average_true_range(high, low, close, window=14)
    out["obv"] = ta.volume.on_balance_volume(close, volume)
    out["adx_14"] = ta.trend.adx(high, low, close, window=14)
    out["stoch_k"] = ta.momentum.stoch(high, low, close, window=14)

    out["daily_return"] = close.pct_change()
    out["volatility_20d"] = out["daily_return"].rolling(20).std() * np.sqrt(252)

    return out


def technical_score(df: pd.DataFrame) -> float:
    """
    Combines several indicators into one score in [-1, 1].
    +1 = strongly bullish setup, -1 = strongly bearish setup.
    Uses only the most recent complete row.
    """
    df_ind = add_indicators(df)
    clean = df_ind.dropna()
    if clean.empty:
        return 0.0

    row = clean.iloc[-1]
    signals = []

    # Trend: price vs moving averages
    signals.append(1.0 if row["close"] > row["sma_50"] else -1.0)
    signals.append(1.0 if row["sma_50"] > row["sma_200"] else -1.0)  # golden/death cross regime

    # Momentum: MACD histogram sign
    signals.append(np.clip(row["macd_hist"] / (abs(row["macd"]) + 1e-6), -1, 1))

    # RSI: overbought/oversold mapped to a fade or continuation signal
    rsi = row["rsi_14"]
    if rsi > 70:
        signals.append(-0.6)     # overbought caution
    elif rsi < 30:
        signals.append(0.6)      # oversold, potential bounce
    else:
        signals.append((rsi - 50) / 50)  # mild trend agreement

    # Bollinger %B: near upper band = extended, near lower = discounted
    bb_pct = row["bb_pct"]
    signals.append(np.clip(1 - 2 * bb_pct, -1, 1) * 0.5)

    # ADX: trend strength multiplier (weak trend -> shrink conviction)
    adx = row["adx_14"]
    trend_strength = np.clip(adx / 50, 0, 1)

    base_score = float(np.mean(signals))
    return float(np.clip(base_score * (0.5 + 0.5 * trend_strength), -1, 1))
