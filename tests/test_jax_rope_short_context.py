from __future__ import annotations

import importlib.util
import unittest


HAS_JAX = importlib.util.find_spec("jax") is not None


@unittest.skipUnless(HAS_JAX, "jax not installed")
class ShortContextRopeTest(unittest.TestCase):
    """SWA rotates keys at positions up to window + mini_batch.

    The RoPE table used to hold only 2 * seq_len rows, and jnp.take fills
    out-of-range rows with NaN, so SWA / TTT-E2E models scored at a context
    shorter than about half their window returned NaN loss (for the 125M
    configs: any context below 4.6K tokens, e.g. zero-shot evals at 1K).
    """

    def _cfg(self, *, seq_len: int, window: int, mini_batch: int, prime: bool):
        from ttt.config import Config

        cfg = Config()
        cfg.model.vocab_size = 64
        cfg.model.hidden_size = 32
        cfg.model.intermediate_size = 64
        cfg.model.num_hidden_layers = 2
        cfg.model.num_attention_heads = 4
        cfg.model.tie_word_embeddings = True
        cfg.model.seq_modeling_block = "SWA"
        cfg.model.seq_len = seq_len
        cfg.model.sliding_window_size = window
        cfg.model.mini_batch_size = mini_batch
        cfg.model.prime = prime
        cfg.model.suffix_len = 1 if prime else 0
        cfg.training.spec_inner = ["language_model.**.suffix_blocks.feed_forward_prime.**"]
        cfg.training.train_mode = cfg.training.TrainMode.meta if prime else cfg.training.TrainMode.pretrain
        return cfg

    def _loss(self, cfg) -> float:
        import equinox as eqx
        import jax
        import jax.numpy as jnp

        from ttt.jax_runtime.model.data import Batch
        from ttt.jax_runtime.model.transformer import MetaModel

        model, state = eqx.nn.make_with_state(MetaModel)(cfg, key=jax.random.PRNGKey(0))
        state = state.set(model.step_index, jnp.array(0, dtype=jnp.int32))
        n = int(cfg.model.seq_len)
        tokens = jnp.arange(n + 1, dtype=jnp.int32) % int(cfg.model.vocab_size)
        seq = Batch(
            input_ids=tokens[:-1],
            target_tokens=tokens[1:],
            loss_masks=jnp.ones((n,), dtype=jnp.float32),
        )
        mesh = jax.make_mesh((1, 1), ("data", "state"))
        with mesh:
            return float(eqx.filter_jit(lambda m, s, st: m.loss_for_sequence(s, st)[0])(model, seq, state))

    def test_swa_context_shorter_than_half_the_window_is_finite(self) -> None:
        import math

        # seq_len 16, window 32, mini-batch 16: keys reach position 48 > 2 * 16.
        loss = self._loss(self._cfg(seq_len=16, window=32, mini_batch=16, prime=False))
        self.assertTrue(math.isfinite(loss), loss)

    def test_ttt_e2e_context_shorter_than_half_the_window_is_finite(self) -> None:
        import math

        loss = self._loss(self._cfg(seq_len=16, window=32, mini_batch=16, prime=True))
        self.assertTrue(math.isfinite(loss), loss)

    def test_larger_table_keeps_existing_rows(self) -> None:
        import numpy as np

        from ttt.jax_runtime.model.attention import precompute_freqs_cis

        small = np.asarray(precompute_freqs_cis(8, 32, theta=5e5))
        large = np.asarray(precompute_freqs_cis(8, 96, theta=5e5))
        self.assertTrue(np.array_equal(small, large[:32]))


if __name__ == "__main__":
    unittest.main()
