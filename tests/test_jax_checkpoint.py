from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


HAS_JAX = importlib.util.find_spec("jax") is not None


@unittest.skipUnless(HAS_JAX, "jax not installed")
class JaxCheckpointTest(unittest.TestCase):
    def test_save_and_load_latest(self) -> None:
        import jax.numpy as jnp

        from ttt.jax_runtime.checkpoint import JaxCheckpointer

        with tempfile.TemporaryDirectory() as td:
            ckpt = JaxCheckpointer(Path(td))
            params = {"w": jnp.ones((2, 3), dtype=jnp.float32)}
            opt_state = {"m": jnp.zeros((2, 3), dtype=jnp.float32)}

            sidecar = ckpt.save(
                step=3,
                params=params,
                opt_state=opt_state,
                metrics={"loss": 1.23},
                metadata={"mode": "test"},
            )
            self.assertTrue(sidecar.exists())

            restored = ckpt.load(step=None)
            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertEqual(restored.step, 3)
            self.assertIn("metrics", restored.payload)
            self.assertIn("metadata", restored.payload)


if __name__ == "__main__":
    unittest.main()
