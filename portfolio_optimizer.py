"""
portfolio_optimizer.py
========================
Modern Portfolio Theory (Markowitz) allocation:
  - Efficient frontier via Monte Carlo simulation + SLSQP refinement
  - Max Sharpe ratio and Min volatility portfolios
  - Risk-appetite presets (conservative / balanced / aggressive) that pick
    a target point along the frontier
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import CONFIG

RiskAppetite = Literal["conservative", "balanced", "aggressive"]


@dataclass
class PortfolioResult:
    weights: Dict[str, float]
    expected_annual_return: float
    annual_volatility: float
    sharpe_ratio: float


def compute_returns(price_data: Union[Dict[str, pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
    """Aligns close prices across tickers and returns daily pct-change returns."""
    if isinstance(price_data, pd.DataFrame):
        return price_data.pct_change().dropna()
    closes = pd.DataFrame({t: df["close"] for t, df in price_data.items() if not df.empty and "close" in df.columns})
    closes = closes.dropna(how="any")
    return closes.pct_change().dropna()


def _portfolio_perf(weights, mean_returns, cov_matrix, risk_free_rate):
    ret = float(np.sum(mean_returns * weights) * 252)
    vol = float(np.sqrt(weights.T @ (cov_matrix * 252) @ weights))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def _negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    return -_portfolio_perf(weights, mean_returns, cov_matrix, risk_free_rate)[2]


def _volatility(weights, mean_returns, cov_matrix, risk_free_rate):
    return _portfolio_perf(weights, mean_returns, cov_matrix, risk_free_rate)[1]


class PortfolioOptimizer:
    def __init__(self, config=CONFIG):
        self.config = config

    def _bounds_constraints(self, n_assets: int, max_weight: float):
        bounds = tuple((0.0, max_weight) for _ in range(n_assets))
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        return bounds, constraints

    def max_sharpe_portfolio(self, returns: pd.DataFrame, max_weight: float = 0.4) -> PortfolioResult:
        tickers = list(returns.columns)
        n = len(tickers)
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        bounds, constraints = self._bounds_constraints(n, max_weight)
        init = np.repeat(1 / n, n)

        result = minimize(
            _negative_sharpe, init,
            args=(mean_returns.values, cov_matrix.values, self.config.risk_free_rate),
            method="SLSQP", bounds=bounds, constraints=constraints,
        )
        return self._to_result(result.x, tickers, mean_returns, cov_matrix)

    def min_volatility_portfolio(self, returns: pd.DataFrame, max_weight: float = 0.4) -> PortfolioResult:
        tickers = list(returns.columns)
        n = len(tickers)
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        bounds, constraints = self._bounds_constraints(n, max_weight)
        init = np.repeat(1 / n, n)

        result = minimize(
            _volatility, init,
            args=(mean_returns.values, cov_matrix.values, self.config.risk_free_rate),
            method="SLSQP", bounds=bounds, constraints=constraints,
        )
        return self._to_result(result.x, tickers, mean_returns, cov_matrix)

    def target_volatility_portfolio(self, returns: pd.DataFrame, target_vol: float,
                                     max_weight: float = 0.4) -> PortfolioResult:
        """Maximizes return subject to annual volatility <= target_vol."""
        tickers = list(returns.columns)
        n = len(tickers)
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        bounds, base_constraints = self._bounds_constraints(n, max_weight)

        constraints = base_constraints + [{
            "type": "ineq",
            "fun": lambda w: target_vol - _volatility(w, mean_returns.values, cov_matrix.values, 0),
        }]

        def neg_return(w):
            return -np.sum(mean_returns.values * w) * 252

        init = np.repeat(1 / n, n)
        result = minimize(neg_return, init, method="SLSQP", bounds=bounds, constraints=constraints)
        return self._to_result(result.x, tickers, mean_returns, cov_matrix)

    def efficient_frontier(self, returns: pd.DataFrame, n_points: int = 40,
                            max_weight: float = 0.4) -> List[PortfolioResult]:
        min_vol = self.min_volatility_portfolio(returns, max_weight)
        max_sharpe = self.max_sharpe_portfolio(returns, max_weight)
        lo, hi = min_vol.annual_volatility, max(max_sharpe.annual_volatility * 1.5, min_vol.annual_volatility * 1.1)
        frontier = []
        for target in np.linspace(lo, hi, n_points):
            try:
                frontier.append(self.target_volatility_portfolio(returns, target, max_weight))
            except Exception:
                continue
        return frontier

    def recommend_by_risk_appetite(self, returns: pd.DataFrame, appetite: RiskAppetite,
                                    max_weight: float = 0.4) -> PortfolioResult:
        min_vol = self.min_volatility_portfolio(returns, max_weight)
        max_sharpe = self.max_sharpe_portfolio(returns, max_weight)

        if appetite == "conservative":
            return min_vol
        if appetite == "aggressive":
            # push past max-Sharpe toward the higher-return, higher-vol end
            target = max_sharpe.annual_volatility * 1.35
            return self.target_volatility_portfolio(returns, target, max_weight)
        # balanced -> the max Sharpe (best risk-adjusted) portfolio
        return max_sharpe

    def _to_result(self, weights, tickers, mean_returns, cov_matrix) -> PortfolioResult:
        weights = np.clip(weights, 0, None)
        weights = weights / weights.sum()
        ret, vol, sharpe = _portfolio_perf(weights, mean_returns.values, cov_matrix.values,
                                            self.config.risk_free_rate)
        weight_map = {t: float(w) for t, w in zip(tickers, weights) if w > 0.005}
        return PortfolioResult(weights=weight_map, expected_annual_return=ret,
                                annual_volatility=vol, sharpe_ratio=sharpe)
