from __future__ import annotations

import importlib.util
import unittest
from unittest import mock


HAS_JAX = importlib.util.find_spec("jax") is not None


@unittest.skipUnless(HAS_JAX, "jax not installed")
class JaxAttentionParityTest(unittest.TestCase):
    def _make_cfg(self, *, block: str = "self_attention"):
        from ttt.config import ModelConfig

        return ModelConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            seq_modeling_block=block,
            mini_batch_size=4,
            sliding_window_size=8,
            seq_len=16,
            vocab_size=32,
            force_flash=True,
        )

    def _make_seq(self, seq_len: int):
        import jax.numpy as jnp

        from ttt.jax_runtime.model.data import Batch

        return Batch(
            input_ids=jnp.zeros((seq_len,), dtype=jnp.int32),
            target_tokens=jnp.zeros((seq_len,), dtype=jnp.int32),
            loss_masks=jnp.ones((seq_len,), dtype=jnp.float32),
            attention_mask=None,
            position_ids=None,
        )

    def test_full_attention_call_signature_matches_reference(self) -> None:
        import equinox as eqx
        import jax.numpy as jnp
        import jax.random as jrandom

        from ttt.jax_runtime.model.attention import Attention

        cfg = self._make_cfg(block="self_attention")
        model, state = eqx.nn.make_with_state(Attention)(cfg, key=jrandom.PRNGKey(0))
        hidden_states = jnp.ones((4, cfg.hidden_size), dtype=jnp.float32)
        seq = self._make_seq(4)

        recorded: dict[str, object] = {}

        def fake_attention(query, key, value, **kwargs):
            recorded["kwargs"] = kwargs
            return query

        with (
            mock.patch("jax.default_backend", return_value="gpu"),
            mock.patch("jax.lax.with_sharding_constraint", side_effect=lambda x, *_args, **_kwargs: x),
            mock.patch("jax.nn.dot_product_attention", side_effect=fake_attention),
        ):
            _out, _state = model(hidden_states, seq, state, is_prefix=False)

        self.assertEqual(
            recorded["kwargs"],
            {
                "is_causal": True,
                "implementation": "cudnn",
            },
        )

    def test_swa_full_call_signature_matches_reference(self) -> None:
        import equinox as eqx
        import jax.numpy as jnp
        import jax.random as jrandom

        from ttt.jax_runtime.model.attention import SWAFull

        cfg = self._make_cfg(block="SWA")
        model, state = eqx.nn.make_with_state(SWAFull)(cfg, key=jrandom.PRNGKey(0))
        hidden_states = jnp.ones((4, cfg.hidden_size), dtype=jnp.float32)
        seq = self._make_seq(4)

        recorded: dict[str, object] = {}

        def fake_attention(query, key, value, **kwargs):
            recorded["kwargs"] = kwargs
            return query

        with (
            mock.patch("jax.default_backend", return_value="gpu"),
            mock.patch("jax.lax.with_sharding_constraint", side_effect=lambda x, *_args, **_kwargs: x),
            mock.patch("jax.nn.dot_product_attention", side_effect=fake_attention),
        ):
            _out, _state = model(hidden_states, seq, state, is_prefix=False)

        self.assertEqual(
            recorded["kwargs"],
            {
                "local_window_size": (cfg.sliding_window_size - 1, 0),
                "is_causal": True,
                "implementation": "cudnn",
            },
        )


if __name__ == "__main__":
    unittest.main()
