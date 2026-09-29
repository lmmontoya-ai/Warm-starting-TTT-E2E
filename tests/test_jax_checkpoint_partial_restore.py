from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HAS_JAX = importlib.util.find_spec("jax") is not None
HAS_ORBAX = importlib.util.find_spec("orbax.checkpoint") is not None


@unittest.skipUnless(HAS_JAX and HAS_ORBAX, "jax/orbax not installed")
class JaxCheckpointPartialRestoreTest(unittest.TestCase):
    def test_params_restore_exact_shapes_prefers_checkpoint_values(self) -> None:
        import jax.numpy as jnp

        from ttt.config import TrainingConfig
        from ttt.jax_runtime.checkpoint import OrbaxCheckpointer

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ckpt"
            ck = OrbaxCheckpointer(root)
            ck.save(
                step=3,
                model_weights={
                    "match": jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
                },
                opt_state=None,
                metrics={"loss": 1.0},
                metadata={},
            )
            restored = ck.load(
                step=3,
                targets={
                    "model_weights": {
                        "match": jnp.zeros((2, 3), dtype=jnp.float32),
                    }
                },
                restore=TrainingConfig.LoadPart.params,
            )

            self.assertEqual(restored.step, 3)
            self.assertTrue((restored.model_weights["match"] == jnp.arange(6, dtype=jnp.float32).reshape(2, 3)).all())
            self.assertIsNotNone(restored.payload)
            skipped = restored.payload.get("skipped_mismatched_params", [])
            self.assertEqual(skipped, [])
            self.assertIsNotNone(restored.report)
            self.assertIn(restored.report.mode, ("exact", "fallback_partial"))
            self.assertEqual(restored.report.fresh_params, 0)
            self.assertEqual(restored.report.restored_params, 6)

    def test_params_restore_skips_shape_mismatches(self) -> None:
        import jax.numpy as jnp

        from ttt.config import TrainingConfig
        from ttt.jax_runtime.checkpoint import OrbaxCheckpointer, unify_dict_with_eqx_module

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ckpt"
            ck = OrbaxCheckpointer(root)
            ck.save(
                step=7,
                model_weights={
                    "match": jnp.full((2, 3), 7.0, dtype=jnp.float32),
                    "mismatch": jnp.full((4,), 9.0, dtype=jnp.float32),
                },
                opt_state=None,
                metrics={"loss": 1.0},
                metadata={},
            )
            restored = ck.load(
                step=7,
                targets={
                    "model_weights": {
                        "match": jnp.zeros((2, 3), dtype=jnp.float32),
                        "mismatch": jnp.zeros((5,), dtype=jnp.float32),
                    }
                },
                restore=TrainingConfig.LoadPart.params,
            )

            self.assertEqual(restored.step, 7)
            # The loader hands back checkpoint-keyed weights; the mismatched
            # tensor keeps its checkpoint shape and is skipped at apply time.
            self.assertTrue((restored.model_weights["match"] == 7.0).all())
            self.assertEqual(restored.model_weights["mismatch"].shape, (4,))
            applied, _, skipped_at_apply = unify_dict_with_eqx_module(
                restored.model_weights,
                {"match": jnp.zeros((2, 3), dtype=jnp.float32), "mismatch": jnp.zeros((5,), dtype=jnp.float32)},
                allow_shape_mismatch=True,
            )
            self.assertTrue((applied["match"] == 7.0).all())
            self.assertTrue((applied["mismatch"] == 0.0).all())
            self.assertEqual(len(skipped_at_apply), 1)
            self.assertIsNotNone(restored.payload)
            skipped = restored.payload.get("skipped_mismatched_params", [])
            self.assertEqual(len(skipped), 1)
            self.assertIn("mismatch", skipped[0])
            self.assertIsNotNone(restored.report)
            self.assertEqual(restored.report.mode, "fallback_partial")
            self.assertEqual(restored.report.restored_params, 6)
            self.assertEqual(restored.report.mismatched_params, 5)
            self.assertEqual(restored.report.missed_params, 0)
            self.assertEqual(restored.report.mismatched_paths, ("['mismatch']",))

    def test_params_restore_falls_back_when_exact_target_mapping_raises(self) -> None:
        import jax.numpy as jnp

        from ttt.config import TrainingConfig
        from ttt.jax_runtime.checkpoint import LATEST_FILENAME, OrbaxCheckpointer

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ckpt"
            root.mkdir(parents=True, exist_ok=True)
            (root / LATEST_FILENAME).write_text(json.dumps({"step": 9}), encoding="utf-8")
            (root / "step_metadata_00000009.json").write_text(json.dumps({"step": 9}), encoding="utf-8")

            ck = OrbaxCheckpointer.__new__(OrbaxCheckpointer)
            ck.checkpoint_dir = root
            ck.manager = mock.Mock()
            ck.manager.latest_step.return_value = 9
            ck.manager.item_metadata.return_value = {
                "model_weights": {
                    "match": jnp.ones((2, 3), dtype=jnp.float32),
                    "mismatch": jnp.ones((4,), dtype=jnp.float32),
                }
            }
            ck.manager.restore.return_value = {
                "model_weights": {
                    "match": jnp.full((2, 3), 5.0, dtype=jnp.float32),
                    "mismatch": jnp.full((4,), 9.0, dtype=jnp.float32),
                }
            }

            with mock.patch(
                "ttt.jax_runtime.checkpoint.fetch_from_eqx_module",
                side_effect=ValueError("shape mismatch"),
            ):
                restored = ck.load(
                    step=9,
                    targets={
                        "model_weights": {
                            "match": jnp.zeros((2, 3), dtype=jnp.float32),
                            "mismatch": jnp.zeros((5,), dtype=jnp.float32),
                        }
                    },
                    restore=TrainingConfig.LoadPart.params,
                )

            self.assertEqual(restored.step, 9)
            self.assertTrue((restored.model_weights["match"] == 5.0).all())
            self.assertEqual(restored.model_weights["mismatch"].shape, (4,))
            self.assertEqual(restored.report.mode, "fallback_partial")
            self.assertEqual(restored.report.restored_params, 6)
            self.assertEqual(ck.manager.restore.call_count, 1)
            self.assertIsNotNone(restored.payload)
            skipped = restored.payload.get("skipped_mismatched_params", [])
            self.assertEqual(len(skipped), 1)
            self.assertIn("mismatch", skipped[0])

    def test_standard_restore_receives_fallback_sharding(self) -> None:
        import jax
        import jax.numpy as jnp

        from ttt.config import TrainingConfig
        from ttt.jax_runtime.checkpoint import LATEST_FILENAME, OrbaxCheckpointer

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ckpt"
            root.mkdir(parents=True, exist_ok=True)
            (root / LATEST_FILENAME).write_text(json.dumps({"step": 9}), encoding="utf-8")
            (root / "step_metadata_00000009.json").write_text(json.dumps({"step": 9}), encoding="utf-8")

            ck = OrbaxCheckpointer.__new__(OrbaxCheckpointer)
            ck.checkpoint_dir = root
            ck.manager = mock.Mock()
            ck.manager.latest_step.return_value = 9
            ck.manager.item_metadata.return_value = {
                "model_weights": {
                    "match": jnp.ones((2, 3), dtype=jnp.float32),
                }
            }
            ck.manager.restore.return_value = {
                "model_weights": {
                    "match": jnp.full((2, 3), 5.0, dtype=jnp.float32),
                }
            }

            restored = ck.load(
                step=9,
                targets={
                    "model_weights": {
                        "match": jnp.zeros((2, 3), dtype=jnp.float32),
                    }
                },
                restore=TrainingConfig.LoadPart.params,
            )

            self.assertEqual(restored.step, 9)
            composite_args = ck.manager.restore.call_args.kwargs["args"]
            standard_args = composite_args.model_weights
            self.assertIsNotNone(standard_args.fallback_sharding)
            self.assertIsInstance(standard_args.fallback_sharding, jax.sharding.SingleDeviceSharding)


@unittest.skipUnless(HAS_JAX and HAS_ORBAX, "jax/orbax not installed")
class MetaModelRestoreValuesTest(unittest.TestCase):
    """Restore into a real MetaModel and compare tensor values, in both restore modes.

    The dict-target tests above could not catch the revision-v2 bug: the
    fallback loader returned a model-shaped tree and the model-side apply then
    matched no keys, so every tensor silently kept its fresh initialization.
    """

    def _cfg(self, *, intermediate_size: int, prime: bool, prime_intermediate_size=None):
        from ttt.config import Config

        cfg = Config()
        cfg.model.vocab_size = 128
        cfg.model.hidden_size = 32
        cfg.model.intermediate_size = intermediate_size
        cfg.model.prime_intermediate_size = prime_intermediate_size
        cfg.model.num_hidden_layers = 4
        cfg.model.num_attention_heads = 4
        cfg.model.tie_word_embeddings = True
        cfg.model.seq_len = 64
        cfg.model.mini_batch_size = 16
        cfg.model.sliding_window_size = 32
        cfg.model.seq_modeling_block = "SWA" if prime else "self_attention"
        cfg.model.prime = prime
        cfg.model.suffix_len = 1 if prime else 0
        cfg.training.spec_inner = ["language_model.**.suffix_blocks.feed_forward_prime.**"]
        cfg.training.train_mode = cfg.training.TrainMode.meta if prime else cfg.training.TrainMode.pretrain
        return cfg

    def _leaves(self, tree):
        import jax
        import numpy as np

        return {jax.tree_util.keystr(p): np.asarray(v) for p, v in jax.tree.flatten_with_path(tree)[0]}

    def _restore(self, parent_cfg, child_cfg):
        import equinox as eqx
        import jax

        from ttt.config import TrainingConfig
        from ttt.jax_runtime.checkpoint import OrbaxCheckpointer, apply_restored_weights, verify_restored_values
        from ttt.jax_runtime.model.transformer import MetaModel

        parent, _ = eqx.nn.make_with_state(MetaModel)(parent_cfg, key=jax.random.PRNGKey(1))
        child, _ = eqx.nn.make_with_state(MetaModel)(child_cfg, key=jax.random.PRNGKey(2))
        with tempfile.TemporaryDirectory() as td:
            ck = OrbaxCheckpointer(Path(td) / "parent")
            ck.save(step=1, model_weights=parent.weights(), opt_state=None, metrics={}, metadata={})
            ck.close()
            loader = OrbaxCheckpointer(Path(td) / "parent", for_saving=False)
            payload = loader.load(step=1, targets={"model_weights": child.weights()}, restore=TrainingConfig.LoadPart.params)
            loader.close()
        restored = apply_restored_weights(child, payload.model_weights)
        check = verify_restored_values(restored.weights(), payload.model_weights)
        return parent, child, restored, payload, check

    def _assert_inherited(self, parent, restored):
        import numpy as np

        parent_leaves = self._leaves(parent.weights())
        restored_leaves = self._leaves(restored.weights())
        inherited = [k for k in restored_leaves if k in parent_leaves and parent_leaves[k].shape == restored_leaves[k].shape]
        self.assertGreater(len(inherited), 0)
        for key in inherited:
            self.assertTrue(np.array_equal(restored_leaves[key], parent_leaves[key]), key)
        # The embedding is the largest inherited tensor; spell it out.
        self.assertTrue([k for k in inherited if "wte" in k], "embedding not inherited")
        return inherited

    def test_exact_mode_width_preserving_conversion(self) -> None:
        parent_cfg = self._cfg(intermediate_size=64, prime=False)
        child_cfg = self._cfg(intermediate_size=64, prime=True, prime_intermediate_size=48)
        parent, _, restored, payload, check = self._restore(parent_cfg, child_cfg)
        self.assertEqual(payload.report.mode, "exact")
        self.assertEqual(payload.report.mismatched_params, 0)
        self.assertGreater(payload.report.missed_params, 0)  # the new prime MLPs
        self.assertEqual(check.equal_tensors, check.checked_tensors)
        self.assertGreater(check.checked_tensors, 0)
        inherited = self._assert_inherited(parent, restored)
        self.assertTrue([k for k in inherited if "feed_forward.w1" in k and "prime" not in k], "FFN not inherited")

    def test_fallback_mode_applies_every_shape_compatible_tensor(self) -> None:
        import numpy as np

        parent_cfg = self._cfg(intermediate_size=64, prime=False)
        child_cfg = self._cfg(intermediate_size=48, prime=True)
        parent, child, restored, payload, check = self._restore(parent_cfg, child_cfg)
        self.assertEqual(payload.report.mode, "fallback_partial")
        self.assertGreater(payload.report.mismatched_params, 0)
        self.assertEqual(check.equal_tensors, check.checked_tensors)
        self.assertGreater(check.checked_tensors, 0)
        self._assert_inherited(parent, restored)
        # Shape-mismatched FFN weights keep the child's fresh values.
        child_leaves = self._leaves(child.weights())
        restored_leaves = self._leaves(restored.weights())
        ffn = [k for k in restored_leaves if "feed_forward.w1" in k and "prime" not in k]
        self.assertTrue(ffn)
        for key in ffn:
            self.assertTrue(np.array_equal(restored_leaves[key], child_leaves[key]), key)



if __name__ == "__main__":
    unittest.main()
