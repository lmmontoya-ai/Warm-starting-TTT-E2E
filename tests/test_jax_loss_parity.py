from __future__ import annotations

import importlib.util
import unittest


HAS_JAX = importlib.util.find_spec("jax") is not None


@unittest.skipUnless(HAS_JAX, "jax not installed")
class JaxLossParityTest(unittest.TestCase):
    def test_cross_entropy_matches_reference_contract(self) -> None:
        import jax.numpy as jnp

        from ttt.jax_runtime.model.loss import cross_entropy_loss_and_accuracy as local_ce
        from ttte2e_reference.e2e.ttt.model.loss import cross_entropy_loss_and_accuracy as ref_ce

        logits = jnp.asarray(
            [
                [[2.0, 0.5, -1.0], [0.1, 1.1, -0.7], [-0.2, 0.3, 0.8]],
                [[-0.1, 1.3, 0.4], [1.5, -0.3, 0.1], [0.7, 0.6, -0.9]],
            ],
            dtype=jnp.float32,
        )
        tokens = jnp.asarray([[0, 1, 2], [1, 0, 2]], dtype=jnp.int32)
        valid = jnp.asarray([[1, 1, 1], [1, 1, 0]], dtype=jnp.float32)

        local_loss, local_pure_ce = local_ce(logits, tokens, valid)
        ref_loss, ref_pure_ce = ref_ce(logits, tokens, valid)

        self.assertAlmostEqual(float(local_loss), float(ref_loss), places=6)
        self.assertAlmostEqual(float(local_pure_ce), float(ref_pure_ce), places=6)


if __name__ == "__main__":
    unittest.main()
