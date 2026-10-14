from __future__ import annotations

import math
import random
import unittest

from ttt.model.token_stats import TokenStatsModel


class TokenStatsModelTest(unittest.TestCase):
    def test_counted_updates_match_per_token_reference(self) -> None:
        rng = random.Random(0)
        vocab = 50
        model = TokenStatsModel(vocab_size=vocab)
        state = model.fresh_state()
        ref_counts: dict[int, float] = {}
        ref_total = 0.0
        for weight in (1.0, 0.3, 2.5):
            tokens = [rng.randrange(vocab) for _ in range(400)]

            # Per-position reference: add-one smoothed unigram NLL.
            ref_nll = 0.0
            for t in tokens:
                p = (ref_counts.get(t, 0.0) + 1.0) / (ref_total + vocab)
                ref_nll += -math.log(p)
            ref_nll /= len(tokens)
            self.assertAlmostEqual(model.nll(state, tokens), ref_nll, places=9)

            model.update(state, tokens, weight=weight)
            for t in tokens:
                ref_counts[t] = ref_counts.get(t, 0.0) + weight
                ref_total += weight
            self.assertAlmostEqual(state.total_count, ref_total, places=9)
            for t, count in ref_counts.items():
                self.assertAlmostEqual(state.token_counts[t], count, places=9)

    def test_empty_and_non_positive_weight_are_no_ops(self) -> None:
        model = TokenStatsModel(vocab_size=10)
        state = model.fresh_state()
        self.assertEqual(model.nll(state, []), 0.0)
        model.update(state, [1, 2, 3], weight=0.0)
        self.assertEqual(state.token_counts, {})
        self.assertEqual(state.total_count, 0.0)


if __name__ == "__main__":
    unittest.main()
