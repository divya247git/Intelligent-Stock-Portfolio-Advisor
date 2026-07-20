"""
sentiment_analysis.py
=======================
NLP sentiment scoring for news headlines and social posts using a
finance-tuned transformer (FinBERT by default). Falls back to a simple
lexicon scorer if transformers/torch aren't available, so the rest of
the platform still runs.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Tuple

import numpy as np

from .config import CONFIG

logger = logging.getLogger(__name__)

_POS_WORDS = {"beat", "beats", "surge", "soar", "record", "growth", "upgrade",
              "outperform", "strong", "profit", "bullish", "rally", "gain"}
_NEG_WORDS = {"miss", "misses", "plunge", "crash", "downgrade", "underperform",
              "weak", "loss", "bearish", "selloff", "lawsuit", "fraud", "layoffs"}


def _lexicon_fallback(text: str) -> float:
    words = set(text.lower().split())
    pos = len(words & _POS_WORDS)
    neg = len(words & _NEG_WORDS)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


class SentimentAnalyzer:
    """
    Wraps a HuggingFace sentiment pipeline (FinBERT). Lazily loaded so the
    module can be imported without pulling in torch until actually needed.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or CONFIG.sentiment_model_name
        self._pipeline = None
        self._load_failed = False

    def _get_pipeline(self):
        if self._pipeline is not None or self._load_failed:
            return self._pipeline
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                truncation=True,
                max_length=256,
            )
        except Exception as e:
            logger.warning("Could not load transformer model (%s). "
                            "Falling back to lexicon sentiment. Error: %s",
                            self.model_name, e)
            self._load_failed = True
            self._pipeline = None
        return self._pipeline

    def score_texts(self, texts: List[str]) -> List[float]:
        """Returns a list of scores in [-1, 1], one per input text."""
        if not texts:
            return []

        pipe = self._get_pipeline()
        if pipe is None:
            return [_lexicon_fallback(t) for t in texts]

        results = pipe(texts, batch_size=16)
        scores = []
        for r in results:
            label = r["label"].lower()
            conf = r["score"]
            if "pos" in label:
                scores.append(conf)
            elif "neg" in label:
                scores.append(-conf)
            else:
                scores.append(0.0)
        return scores

    def aggregate_score(self, texts: List[str]) -> Tuple[float, int]:
        """Mean sentiment score and the number of texts it was computed from."""
        scores = self.score_texts(texts)
        if not scores:
            return 0.0, 0
        return float(np.mean(scores)), len(scores)


@lru_cache(maxsize=1)
def get_default_analyzer() -> SentimentAnalyzer:
    """Shared singleton so the model is only loaded once per process."""
    return SentimentAnalyzer()
