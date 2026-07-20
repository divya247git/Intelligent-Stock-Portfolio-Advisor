"""
recommendation_engine.py
==========================
Combines fundamental, technical, news-sentiment, and social-sentiment
scores (each in [-1, 1]) into a single weighted recommendation, plus
an optional LSTM price-direction tilt shown separately (kept out of the
core weighted score since it's a single noisy model, not an ensemble).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import CONFIG
from .data_fetcher import DataFetcher
from .technical_indicators import add_indicators, technical_score
from .fundamental_analysis import fundamental_score, fundamental_summary
from .sentiment_analysis import get_default_analyzer
from .lstm_predictor import LSTMPredictor, LSTMForecastResult


@dataclass
class Recommendation:
    ticker: str
    action: str                    # "BUY", "HOLD", "SELL"
    composite_score: float         # [-1, 1]
    fundamental_score: float
    technical_score: float
    news_sentiment_score: float
    social_sentiment_score: float
    news_sample_size: int
    social_sample_size: int
    fundamental_notes: str
    lstm_forecast: Optional[LSTMForecastResult]
    rationale: str


def _action_from_score(score: float) -> str:
    if score >= 0.20:
        return "BUY"
    if score <= -0.20:
        return "SELL"
    return "HOLD"


def _build_rationale(f, t, n, s, action: str) -> str:
    drivers = []
    for name, val in [("fundamentals", f), ("technicals", t),
                       ("news sentiment", n), ("social sentiment", s)]:
        if abs(val) >= 0.15:
            direction = "supportive" if val > 0 else "unfavorable"
            drivers.append(f"{name} ({direction}, {val:+.2f})")
    driver_text = "; ".join(drivers) if drivers else "no single factor stood out strongly"
    return f"{action} driven by: {driver_text}."


class RecommendationEngine:
    def __init__(self, config=CONFIG, fetcher: DataFetcher = None):
        self.config = config
        self.fetcher = fetcher or DataFetcher(config)
        self.sentiment = get_default_analyzer()
        self.lstm = LSTMPredictor(config)

    def analyze(self, ticker: str, include_lstm: bool = True,
                include_social: bool = True) -> Recommendation:
        cfg = self.config

        price_df = self.fetcher.get_price_history(ticker)
        price_df = add_indicators(price_df)
        tech_score = technical_score(price_df)

        fundamentals = self.fetcher.get_fundamentals(ticker)
        fund_score = fundamental_score(fundamentals)
        fund_notes = fundamental_summary(fundamentals)

        news = self.fetcher.get_news(ticker)
        news_texts = [f"{n.title}. {n.summary}" for n in news]
        news_score, news_n = self.sentiment.aggregate_score(news_texts)

        social_score, social_n = 0.0, 0
        if include_social:
            social_texts = self.fetcher.get_social_mentions(ticker)
            social_score, social_n = self.sentiment.aggregate_score(social_texts)

        composite = (
            cfg.weight_fundamental * fund_score
            + cfg.weight_technical * tech_score
            + cfg.weight_news_sentiment * news_score
            + cfg.weight_social_sentiment * social_score
        )
        action = _action_from_score(composite)
        rationale = _build_rationale(fund_score, tech_score, news_score, social_score, action)

        lstm_result = None
        if include_lstm:
            try:
                lstm_result = self.lstm.train_and_forecast(price_df, ticker)
                if lstm_result.predicted_return_pct > 1.5 and action == "HOLD":
                    rationale += f" LSTM model also projects a {lstm_result.predicted_return_pct:+.1f}% next-session move (informational only, not weighted in the score)."
            except Exception as e:
                rationale += f" (LSTM forecast unavailable: {e})"

        return Recommendation(
            ticker=ticker,
            action=action,
            composite_score=composite,
            fundamental_score=fund_score,
            technical_score=tech_score,
            news_sentiment_score=news_score,
            social_sentiment_score=social_score,
            news_sample_size=news_n,
            social_sample_size=social_n,
            fundamental_notes=fund_notes,
            lstm_forecast=lstm_result,
            rationale=rationale,
        )
