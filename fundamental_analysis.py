"""
fundamental_analysis.py
=========================
Turns a FundamentalSnapshot into a bounded score in [-1, 1] by comparing
each metric to reasonable heuristic thresholds. Swap thresholds for
sector-relative z-scores if you have peer-group data.
"""
from __future__ import annotations

import numpy as np

from .data_fetcher import FundamentalSnapshot


def _score_lower_is_better(value, good, bad):
    """Linearly maps value in [good, bad] to [1, -1]; clips outside range."""
    if value is None or np.isnan(value):
        return 0.0
    if value <= good:
        return 1.0
    if value >= bad:
        return -1.0
    return 1.0 - 2 * (value - good) / (bad - good)


def _score_higher_is_better(value, bad, good):
    if value is None or np.isnan(value):
        return 0.0
    if value >= good:
        return 1.0
    if value <= bad:
        return -1.0
    return -1.0 + 2 * (value - bad) / (good - bad)


def fundamental_score(f: FundamentalSnapshot) -> float:
    scores = []

    # Valuation — lower PE/PB is "better" within a sane band
    scores.append(_score_lower_is_better(f.pe_ratio, good=12, bad=40))
    scores.append(_score_lower_is_better(f.price_to_book, good=1.5, bad=8))
    scores.append(_score_lower_is_better(f.peg_ratio, good=1.0, bad=3.0))

    # Leverage — lower debt/equity is better
    scores.append(_score_lower_is_better(f.debt_to_equity, good=50, bad=200))

    # Profitability — higher is better
    scores.append(_score_higher_is_better(f.roe, bad=0.0, good=0.25))
    scores.append(_score_higher_is_better(f.roa, bad=0.0, good=0.12))
    scores.append(_score_higher_is_better(f.profit_margin, bad=0.0, good=0.20))

    # Growth — higher is better
    scores.append(_score_higher_is_better(f.revenue_growth, bad=-0.05, good=0.20))
    scores.append(_score_higher_is_better(f.earnings_growth, bad=-0.10, good=0.25))

    scores = [s for s in scores if s is not None]
    return float(np.clip(np.mean(scores), -1, 1)) if scores else 0.0


def fundamental_summary(f: FundamentalSnapshot) -> str:
    parts = []
    if f.pe_ratio is not None:
        parts.append(f"P/E {f.pe_ratio:.1f}")
    if f.roe is not None:
        parts.append(f"ROE {f.roe*100:.1f}%")
    if f.revenue_growth is not None:
        parts.append(f"Rev growth {f.revenue_growth*100:.1f}%")
    if f.debt_to_equity is not None:
        parts.append(f"D/E {f.debt_to_equity:.0f}")
    return ", ".join(parts) if parts else "Insufficient fundamental data"
