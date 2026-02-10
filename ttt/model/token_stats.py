"""Token-frequency based phase-1 language model.

This is intentionally simple and fast for orchestration validation. It provides
real token-dependent losses and updates without requiring deep learning
frameworks during the early reproduction phase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TokenStatsState:
    token_counts: dict[int, float]
    total_count: float

    def to_jsonable(self) -> dict:
        return {
            "token_counts": {str(k): v for k, v in self.token_counts.items()},
            "total_count": self.total_count,
        }

    @staticmethod
    def from_jsonable(payload: dict) -> "TokenStatsState":
        raw = payload.get("token_counts", {})
        return TokenStatsState(
            token_counts={int(k): float(v) for k, v in raw.items()},
            total_count=float(payload.get("total_count", 0.0)),
        )


class TokenStatsModel:
    def __init__(self, vocab_size: int, smoothing_alpha: float = 1.0):
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if smoothing_alpha <= 0:
            raise ValueError(
                f"smoothing_alpha must be positive, got {smoothing_alpha}"
            )

        self.vocab_size = vocab_size
        self.alpha = smoothing_alpha

    def fresh_state(self) -> TokenStatsState:
        return TokenStatsState(token_counts={}, total_count=0.0)

    def _prob(self, state: TokenStatsState, token: int) -> float:
        token_count = state.token_counts.get(token, 0.0)
        numerator = token_count + self.alpha
        denominator = state.total_count + self.alpha * self.vocab_size
        return numerator / denominator

    def nll(self, state: TokenStatsState, tokens: list[int]) -> float:
        if not tokens:
            return 0.0

        total = 0.0
        for t in tokens:
            p = self._prob(state, t)
            total += -math.log(max(p, 1e-12))

        return total / len(tokens)

    def update(self, state: TokenStatsState, tokens: list[int], weight: float = 1.0) -> None:
        if weight <= 0:
            return

        for t in tokens:
            state.token_counts[t] = state.token_counts.get(t, 0.0) + weight
            state.total_count += weight

    def clone_state(self, state: TokenStatsState) -> TokenStatsState:
        return TokenStatsState(token_counts=dict(state.token_counts), total_count=state.total_count)
