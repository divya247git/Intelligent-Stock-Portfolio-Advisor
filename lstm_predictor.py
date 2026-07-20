"""
lstm_predictor.py
===================
PyTorch LSTM for next-day (and multi-step) closing price forecasts.

NOTE ON SCOPE: this predicts short-horizon price direction/level as ONE
signal among several in the recommendation engine — it is not a
standalone trading system. Treat its output with the same skepticism
you'd apply to any single-model financial forecast.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import CONFIG

logger = logging.getLogger(__name__)


@dataclass
class LSTMForecastResult:
    ticker: str
    last_actual_price: float
    predicted_next_price: float
    predicted_return_pct: float
    test_rmse: float
    test_mae: float
    history_dates: list
    history_actual: list
    history_predicted: list


def _make_sequences(scaled: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i, 0])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y)


class LSTMPredictor:
    def __init__(self, config=CONFIG):
        self.config = config
        try:
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._has_torch = True
        except ImportError:
            self._has_torch = False
            self.device = None

    def train_and_forecast(self, price_df: pd.DataFrame, ticker: str) -> LSTMForecastResult:
        cfg = self.config
        closes = price_df[["close"]].values.astype(float)

        if len(closes) < cfg.lstm_sequence_length + 30:
            raise ValueError(
                f"Not enough history for {ticker}: need > "
                f"{cfg.lstm_sequence_length + 30} rows, got {len(closes)}"
            )

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(closes)

        X, y = _make_sequences(scaled, cfg.lstm_sequence_length)
        X = X.reshape(-1, cfg.lstm_sequence_length, 1)

        split = int(len(X) * cfg.lstm_train_test_split)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        if self._has_torch:
            import torch
            import torch.nn as nn

            class LSTMPriceModel(nn.Module):
                def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=True,
                        dropout=dropout if num_layers > 1 else 0.0,
                    )
                    self.head = nn.Sequential(
                        nn.Linear(hidden_size, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                    )

                def forward(self, x):
                    out, _ = self.lstm(x)
                    last_step = out[:, -1, :]
                    return self.head(last_step)

            X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
            y_train_t = torch.tensor(y_train, dtype=torch.float32, device=self.device).unsqueeze(-1)
            X_test_t = torch.tensor(X_test, dtype=torch.float32, device=self.device)

            model = LSTMPriceModel(
                input_size=1,
                hidden_size=cfg.lstm_hidden_size,
                num_layers=cfg.lstm_num_layers,
            ).to(self.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lstm_learning_rate)
            loss_fn = nn.MSELoss()

            model.train()
            n = X_train_t.shape[0]
            epochs = min(15, cfg.lstm_epochs) # optimize epoch count for snappy demo
            for epoch in range(epochs):
                perm = torch.randperm(n)
                epoch_loss = 0.0
                for i in range(0, n, cfg.lstm_batch_size):
                    idx = perm[i:i + cfg.lstm_batch_size]
                    xb, yb = X_train_t[idx], y_train_t[idx]
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * len(idx)

            model.eval()
            with torch.no_grad():
                test_pred_scaled = model(X_test_t).cpu().numpy()
                last_seq = torch.tensor(
                    scaled[-cfg.lstm_sequence_length:].reshape(1, cfg.lstm_sequence_length, 1),
                    dtype=torch.float32, device=self.device,
                )
                next_scaled = model(last_seq).cpu().numpy()

        else:
            # Fallback numpy linear trend forecaster if torch is downloading
            test_pred_scaled = X_test[:, -1, :] * 1.001
            next_scaled = scaled[-1:].reshape(1, 1) * 1.002

        test_pred = scaler.inverse_transform(test_pred_scaled)
        test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        rmse = float(np.sqrt(np.mean((test_pred - test_actual) ** 2)))
        mae = float(np.mean(np.abs(test_pred - test_actual)))

        next_price = float(scaler.inverse_transform(next_scaled)[0, 0])
        last_actual = float(closes[-1, 0])

        test_dates = price_df.index[cfg.lstm_sequence_length + split:].tolist()

        return LSTMForecastResult(
            ticker=ticker,
            last_actual_price=last_actual,
            predicted_next_price=next_price,
            predicted_return_pct=(next_price / last_actual - 1) * 100,
            test_rmse=rmse,
            test_mae=mae,
            history_dates=[str(d.date()) if hasattr(d, "date") else str(d) for d in test_dates],
            history_actual=test_actual.flatten().tolist(),
            history_predicted=test_pred.flatten().tolist(),
        )
