from __future__ import annotations

import importlib.util
import unittest


HAS_JAX = importlib.util.find_spec("jax") is not None


@unittest.skipUnless(HAS_JAX, "jax not installed")
class JaxStatefulParityTest(unittest.TestCase):
    def test_block_collection_split_state(self) -> None:
        import jax.numpy as jnp

        from ttt.jax_runtime.model.transformer import BlockCollectionSplit

        state = {
            "a": jnp.arange(12, dtype=jnp.int32).reshape(6, 2),
            "b": jnp.arange(18, dtype=jnp.int32).reshape(6, 3),
        }

        prefix, suffix = BlockCollectionSplit.split_state(state, 2)
        self.assertEqual(prefix["a"].shape, (4, 2))
        self.assertEqual(prefix["b"].shape, (4, 3))
        self.assertEqual(suffix["a"].shape, (2, 2))
        self.assertEqual(suffix["b"].shape, (2, 3))

        prefix_only, suffix_none = BlockCollectionSplit.split_state(state, 0)
        self.assertEqual(prefix_only["a"].shape, (6, 2))
        self.assertIsNone(suffix_none)

    def test_swa_updates_chunk_and_cache_state(self) -> None:
        import equinox as eqx
        import jax.numpy as jnp
        import jax.random as jrandom

        from ttt.config import ModelConfig
        from ttt.jax_runtime.model.attention import SWA
        from ttt.jax_runtime.model.data import Batch

        cfg = ModelConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            seq_modeling_block="SWA",
            mini_batch_size=4,
            sliding_window_size=8,
            seq_len=16,
            vocab_size=32,
            force_flash=False,
        )
        swa, state = eqx.nn.make_with_state(SWA)(cfg, key=jrandom.PRNGKey(0))
        hidden_states = jnp.ones((cfg.mini_batch_size, cfg.hidden_size), dtype=jnp.float32)
        seq = Batch(
            input_ids=jnp.zeros((cfg.mini_batch_size,), dtype=jnp.int32),
            target_tokens=jnp.zeros((cfg.mini_batch_size,), dtype=jnp.int32),
            loss_masks=jnp.ones((cfg.mini_batch_size,), dtype=jnp.float32),
            attention_mask=None,
            position_ids=None,
        )

        initial_chunk = int(state.get(swa.chunk_index))
        initial_k, initial_v = state.get(swa.kv_cache_index)
        _out, new_state = swa(hidden_states, seq, state, is_prefix=False)
        next_chunk = int(new_state.get(swa.chunk_index))
        next_k, next_v = new_state.get(swa.kv_cache_index)

        self.assertEqual(initial_chunk, 0)
        self.assertEqual(next_chunk, 1)
        self.assertEqual(initial_k.shape, next_k.shape)
        self.assertEqual(initial_v.shape, next_v.shape)
        self.assertFalse(bool(jnp.allclose(initial_k, next_k) and jnp.allclose(initial_v, next_v)))


if __name__ == "__main__":
    unittest.main()
