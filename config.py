"""
Central configuration for the Intelligent Stock Portfolio Advisor.
Keep API keys and tunables here (or override via environment variables).
"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # --- Data sources ---
    price_lookback_days: int = 730          # 2 years of daily bars for indicators/LSTM
    news_lookback_days: int = 14

    # --- Reddit / social sentiment (optional; leave blank to disable) ---
    reddit_client_id: str = os.environ.get("REDDIT_CLIENT_ID", "")
    reddit_client_secret: str = os.environ.get("REDDIT_CLIENT_SECRET", "")
    reddit_user_agent: str = "stock-advisor/1.0"
    social_subreddits: List[str] = field(
        default_factory=lambda: ["stocks", "investing", "wallstreetbets"]
    )

    # --- News API (optional; leave blank to fall back to yfinance news) ---
    news_api_key: str = os.environ.get("NEWS_API_KEY", "")

    # --- Sentiment model ---
    sentiment_model_name: str = "ProsusAI/finbert"

    # --- LSTM ---
    lstm_sequence_length: int = 60
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_epochs: int = 30
    lstm_batch_size: int = 32
    lstm_learning_rate: float = 1e-3
    lstm_train_test_split: float = 0.85

    # --- Portfolio optimization ---
    risk_free_rate: float = 0.045           # annualized, ~T-bill proxy
    mc_simulations: int = 20000             # Monte Carlo portfolios for the efficient frontier

    # --- Paper trading ---
    starting_cash: float = 100_000.0
    commission_per_trade: float = 0.0
    slippage_bps: float = 5.0               # 5 basis points

    # --- Recommendation engine weights (must sum to 1.0) ---
    weight_fundamental: float = 0.30
    weight_technical: float = 0.30
    weight_news_sentiment: float = 0.25
    weight_social_sentiment: float = 0.15


CONFIG = Config()
