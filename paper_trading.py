"""
paper_trading.py
==================
A simple, deterministic paper-trading engine. Users submit orders against
historical or live price data; the engine tracks cash, positions, realized
and unrealized P&L, and a full trade blotter for later attribution analysis.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Literal, Optional

from .config import CONFIG

Side = Literal["BUY", "SELL"]


@dataclass
class Trade:
    timestamp: str
    ticker: str
    side: Side
    quantity: float
    price: float
    commission: float
    reason: str = ""


@dataclass
class Position:
    ticker: str
    quantity: float = 0.0
    avg_cost: float = 0.0

    def market_value(self, current_price: float) -> float:
        return self.quantity * current_price

    def unrealized_pnl(self, current_price: float) -> float:
        return (current_price - self.avg_cost) * self.quantity


class PaperTradingAccount:
    def __init__(self, starting_cash: float = None, config=CONFIG, account_name: str = "default"):
        self.config = config
        self.account_name = account_name
        self.cash: float = starting_cash if starting_cash is not None else config.starting_cash
        self.starting_cash = self.cash
        self.positions: Dict[str, Position] = {}
        self.trade_log: List[Trade] = []
        self.realized_pnl: float = 0.0

    # ------------------------------------------------------------------ #
    def _apply_slippage(self, price: float, side: Side) -> float:
        bps = self.config.slippage_bps / 10_000
        return price * (1 + bps) if side == "BUY" else price * (1 - bps)

    def place_order(self, ticker: str, side: Side, quantity: float,
                     price: float, timestamp: Optional[str] = None,
                     reason: str = "") -> Trade:
        if quantity <= 0:
            raise ValueError("Quantity must be positive")

        exec_price = self._apply_slippage(price, side)
        commission = self.config.commission_per_trade
        cost = exec_price * quantity + commission

        pos = self.positions.setdefault(ticker, Position(ticker=ticker))

        if side == "BUY":
            if cost > self.cash:
                raise ValueError(f"Insufficient cash: need {cost:.2f}, have {self.cash:.2f}")
            new_qty = pos.quantity + quantity
            pos.avg_cost = (pos.avg_cost * pos.quantity + exec_price * quantity) / new_qty
            pos.quantity = new_qty
            self.cash -= cost
        else:  # SELL
            if quantity > pos.quantity:
                raise ValueError(f"Cannot sell {quantity}, only holding {pos.quantity} of {ticker}")
            realized = (exec_price - pos.avg_cost) * quantity - commission
            self.realized_pnl += realized
            pos.quantity -= quantity
            self.cash += exec_price * quantity - commission
            if pos.quantity == 0:
                pos.avg_cost = 0.0

        trade = Trade(
            timestamp=timestamp or datetime.now().isoformat(),
            ticker=ticker, side=side, quantity=quantity,
            price=exec_price, commission=commission, reason=reason,
        )
        self.trade_log.append(trade)
        return trade

    # ------------------------------------------------------------------ #
    def portfolio_value(self, current_prices: Dict[str, float]) -> float:
        holdings = sum(
            pos.market_value(current_prices.get(t, pos.avg_cost))
            for t, pos in self.positions.items() if pos.quantity > 0
        )
        return self.cash + holdings

    def total_return_pct(self, current_prices: Dict[str, float]) -> float:
        return (self.portfolio_value(current_prices) / self.starting_cash - 1) * 100

    def snapshot(self, current_prices: Dict[str, float]) -> dict:
        return {
            "account": self.account_name,
            "cash": round(self.cash, 2),
            "positions": {
                t: {
                    "quantity": p.quantity,
                    "avg_cost": round(p.avg_cost, 2),
                    "market_value": round(p.market_value(current_prices.get(t, p.avg_cost)), 2),
                    "unrealized_pnl": round(p.unrealized_pnl(current_prices.get(t, p.avg_cost)), 2),
                }
                for t, p in self.positions.items() if p.quantity > 0
            },
            "realized_pnl": round(self.realized_pnl, 2),
            "portfolio_value": round(self.portfolio_value(current_prices), 2),
            "total_return_pct": round(self.total_return_pct(current_prices), 2),
        }

    def export_trade_log(self) -> str:
        return json.dumps([asdict(t) for t in self.trade_log], indent=2)


class StrategyBacktester:
    """
    Replays a recommendation-driven strategy over historical price data,
    one bar at a time, using a PaperTradingAccount. Strategy callback
    receives (date, ticker, price_row_history) and returns an action
    'BUY' / 'SELL' / 'HOLD' plus a target position size in dollars.
    """

    def __init__(self, account: PaperTradingAccount):
        self.account = account
        self.equity_curve: List[dict] = []

    def run(self, price_df, ticker: str, strategy_fn, target_notional_frac: float = 0.2):
        for i in range(60, len(price_df)):  # warm-up window for indicator lookback
            window = price_df.iloc[: i + 1]
            date = window.index[-1]
            price = float(window["close"].iloc[-1])

            action = strategy_fn(window)
            current_qty = self.account.positions.get(ticker, Position(ticker)).quantity

            if action == "BUY" and current_qty == 0:
                notional = self.account.cash * target_notional_frac
                qty = notional / price
                if qty > 0:
                    self.account.place_order(ticker, "BUY", qty, price, str(date), reason="strategy signal")
            elif action == "SELL" and current_qty > 0:
                self.account.place_order(ticker, "SELL", current_qty, price, str(date), reason="strategy signal")

            self.equity_curve.append({
                "date": str(date),
                "portfolio_value": self.account.portfolio_value({ticker: price}),
            })
        return self.equity_curve
