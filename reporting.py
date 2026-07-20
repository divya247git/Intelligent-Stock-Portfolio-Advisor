"""
reporting.py
==============
Monthly performance report generation with Brinson-style attribution:
  - Allocation effect: did overweighting a winning holding help?
  - Selection effect: did the specific pick beat its own benchmark?
  - Trading effect: how much P&L came from realized trades (timing) vs
    simply holding the position for the month?

This is a simplified single-portfolio attribution vs. an equal-weight
benchmark of the same universe — swap in a real benchmark (S&P 500 sector
weights, etc.) for production use.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .paper_trading import PaperTradingAccount, Trade


@dataclass
class HoldingAttribution:
    ticker: str
    avg_weight: float
    holding_return_pct: float
    benchmark_weight: float
    benchmark_return_pct: float
    allocation_effect: float
    selection_effect: float
    total_effect: float


@dataclass
class MonthlyReport:
    period_start: str
    period_end: str
    starting_value: float
    ending_value: float
    total_return_pct: float
    realized_pnl: float
    unrealized_pnl: float
    trades_this_period: int
    win_rate_pct: float
    best_holding: str
    worst_holding: str
    holdings_attribution: List[HoldingAttribution]
    commentary: str


def _period_trades(trades: List[Trade], start: str, end: str) -> List[Trade]:
    return [t for t in trades if start <= t.timestamp[:10] <= end or start <= t.timestamp <= end]


def _win_rate(trades: List[Trade]) -> float:
    sells = [t for t in trades if t.side == "SELL"]
    if not sells:
        return 0.0
    return 100.0 * len(sells) / max(len(trades), 1)


def _get_price_at_date(df: pd.DataFrame, target_date: str) -> Optional[float]:
    if df is None or df.empty or "close" not in df.columns:
        return None
    try:
        sub = df.loc[:target_date]
        if not sub.empty:
            return float(sub["close"].iloc[-1])
    except Exception:
        pass
    return float(df["close"].iloc[-1])


def build_monthly_report(
    account: PaperTradingAccount,
    price_history: Dict[str, pd.DataFrame],
    period_start: str,
    period_end: str,
) -> MonthlyReport:
    trades = _period_trades(account.trade_log, period_start, period_end)

    start_prices = {}
    end_prices = {}
    for t, df in price_history.items():
        p_start = _get_price_at_date(df, period_start)
        p_end = _get_price_at_date(df, period_end)
        if p_start is not None:
            start_prices[t] = p_start
        if p_end is not None:
            end_prices[t] = p_end

    starting_value = account.portfolio_value(start_prices)
    ending_value = account.portfolio_value(end_prices)
    total_return = (ending_value / starting_value - 1) * 100 if starting_value else 0.0

    # --- per-holding weights & returns over the period ---
    tickers = [t for t, p in account.positions.items() if p.quantity > 0]
    holding_returns, weights = {}, {}
    for t in tickers:
        if t in start_prices and t in end_prices and start_prices[t] > 0:
            holding_returns[t] = (end_prices[t] / start_prices[t] - 1) * 100
        else:
            holding_returns[t] = 0.0
        weights[t] = account.positions[t].market_value(end_prices.get(t, 0)) / ending_value if ending_value else 0

    # Equal-weight benchmark across the same universe
    n = len(tickers) or 1
    bench_weight = 1 / n
    bench_return = float(np.mean(list(holding_returns.values()))) if holding_returns else 0.0

    attributions = []
    for t in tickers:
        w, r = weights[t], holding_returns[t]
        allocation_effect = (w - bench_weight) * (bench_return)
        selection_effect = bench_weight * (r - bench_return)
        attributions.append(HoldingAttribution(
            ticker=t, avg_weight=w, holding_return_pct=r,
            benchmark_weight=bench_weight, benchmark_return_pct=bench_return,
            allocation_effect=allocation_effect, selection_effect=selection_effect,
            total_effect=allocation_effect + selection_effect,
        ))
    attributions.sort(key=lambda a: a.total_effect, reverse=True)

    realized = sum(t.price * t.quantity * (1 if t.side == "SELL" else -1) for t in trades) if trades else 0.0
    unrealized = sum(p.unrealized_pnl(end_prices.get(t, p.avg_cost)) for t, p in account.positions.items())

    best = attributions[0].ticker if attributions else "N/A"
    worst = attributions[-1].ticker if attributions else "N/A"

    commentary = _generate_commentary(total_return, attributions, trades)

    return MonthlyReport(
        period_start=period_start, period_end=period_end,
        starting_value=starting_value, ending_value=ending_value,
        total_return_pct=total_return, realized_pnl=account.realized_pnl,
        unrealized_pnl=unrealized, trades_this_period=len(trades),
        win_rate_pct=_win_rate(trades), best_holding=best, worst_holding=worst,
        holdings_attribution=attributions, commentary=commentary,
    )


def _generate_commentary(total_return: float, attributions: List[HoldingAttribution],
                          trades: List[Trade]) -> str:
    lines = [f"Portfolio returned {total_return:+.2f}% this period."]
    if attributions:
        top = attributions[0]
        bottom = attributions[-1]
        lines.append(
            f"{top.ticker} was the top contributor ({top.total_effect:+.2f} pts attribution: "
            f"{top.allocation_effect:+.2f} allocation, {top.selection_effect:+.2f} selection)."
        )
        if bottom.ticker != top.ticker:
            lines.append(
                f"{bottom.ticker} was the largest drag ({bottom.total_effect:+.2f} pts attribution)."
            )
    if trades:
        lines.append(f"{len(trades)} trades executed this period.")
    else:
        lines.append("No trades executed this period — return is purely from holding existing positions.")
    return " ".join(lines)


def report_to_markdown(report: MonthlyReport) -> str:
    lines = [
        f"# Portfolio Performance Report",
        f"**Period:** {report.period_start} to {report.period_end}\n",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Starting Value | ${report.starting_value:,.2f} |",
        f"| Ending Value | ${report.ending_value:,.2f} |",
        f"| Total Return | {report.total_return_pct:+.2f}% |",
        f"| Realized P&L | ${report.realized_pnl:,.2f} |",
        f"| Unrealized P&L | ${report.unrealized_pnl:,.2f} |",
        f"| Trades This Period | {report.trades_this_period} |",
        f"| Best Holding | {report.best_holding} |",
        f"| Worst Holding | {report.worst_holding} |",
        "",
        "## Attribution by Holding",
        "| Ticker | Weight | Return | Allocation Effect | Selection Effect | Total Effect |",
        "|---|---|---|---|---|---|",
    ]
    for a in report.holdings_attribution:
        lines.append(
            f"| {a.ticker} | {a.avg_weight*100:.1f}% | {a.holding_return_pct:+.2f}% | "
            f"{a.allocation_effect:+.2f} | {a.selection_effect:+.2f} | {a.total_effect:+.2f} |"
        )
    lines += ["", "## Commentary", report.commentary]
    return "\n".join(lines)
