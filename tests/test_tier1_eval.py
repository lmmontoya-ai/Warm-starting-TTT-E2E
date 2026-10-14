from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ttt.jax_runtime.research_eval import (
    make_continuation_score_masks,
    make_padded_batch,
)
from ttt.research.ruler import aggregate_string_match_all, string_match_all
from ttt.research.tier1 import (
    PendingTargetError,
    condition_spec,
    load_tier1_config,
    make_manifest,
    pending_conditions,
    require_targets,
    validate_tier1_config,
    verify_manifest,
)
from ttt.research.tier1_registry import (
    dependency_closure,
    load_tier1_registry,
    registry_plan_payload,
)
from ttt.research.tier1_stats import exact_mcnemar_two_sided, paired_bootstrap_delta
from ttt.research.tier1_zero_shot import render_example


REPO_ROOT = Path(__file__).resolve().parents[1]
HAS_JAX = importlib.util.find_spec("jax") is not None


class Tier1ProtocolTest(unittest.TestCase):
    def test_preregistration_contains_locked_tier1_rules(self) -> None:
        text = (REPO_ROOT / "PREREGISTRATION_TIER12.md").read_text(encoding="utf-8")
        for phrase in (
            "All task attempts and failures are reported",
            "S2_125M_SEED001",
            "S2_MINUS_125M_SEED001",
            "per-layer",
            "final-checkpoint",
            "niah_multikey_1",
            "f4a3fd8e524acd9abd1fea4387e8f179f6d51cf3",
            "38da79d79519ef87aa46ae804f838e1eab7f86d7",
            "three working days",
            "agrees or disagrees descriptively",
            "left-padded",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_registry_golden_plan_is_first_class_and_acyclic(self) -> None:
        stages = load_tier1_registry(REPO_ROOT / "configs/research/tier1_registry.yaml")
        stage_ids = [stage.stage_id for stage in stages]
        self.assertEqual(len(stage_ids), len(set(stage_ids)))
        for required in (
            "T1_PREREG_GATE",
            "T1A_MANIFEST",
            "T1B_MANIFEST",
            "T1C_HARNESS_GATE",
            "T1A_S2_125M",
            "T1B_S2_MINUS_125M",
            "T1C_S3_125M",
            "T1_REPORT_PRIMARY",
        ):
            self.assertIn(required, stage_ids)
        report_plan = dependency_closure(stages, {"T1_REPORT_PRIMARY"})
        self.assertEqual(report_plan[-1].stage_id, "T1_REPORT_PRIMARY")
        self.assertIn("T1C_HARNESS_GATE", [stage.stage_id for stage in report_plan])
        plan_payload = registry_plan_payload(stages)
        s2_restore = next(
            row for row in plan_payload if row["stage_id"] == "T1_RESTORE_S2_125M"
        )
        self.assertTrue(s2_restore["all_model_checkpoints"])

    def test_manifest_hash_is_tamper_evident(self) -> None:
        manifest = make_manifest(
            manifest_type="unit",
            paper_run_id="revision_v2_t1_eval_v1",
            provenance={"pin": "abc"},
            examples=[{"example_id": "x"}],
        )
        self.assertEqual(
            verify_manifest(manifest, expected_type="unit"), manifest["manifest_sha256"]
        )
        manifest["examples"][0]["example_id"] = "changed"
        with self.assertRaisesRegex(ValueError, "Manifest hash mismatch"):
            verify_manifest(manifest, expected_type="unit")

    def test_paired_statistics_are_deterministic(self) -> None:
        left = [1, 1, 0, 1, 0]
        right = [0, 1, 1, 0, 0]
        first = paired_bootstrap_delta(left, right, resamples=1000, seed=7)
        second = paired_bootstrap_delta(left, right, resamples=1000, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean_delta"], 0.2)
        mcnemar = exact_mcnemar_two_sided(left, right)
        self.assertEqual(mcnemar["discordant"], 3.0)

    def test_official_ruler_string_match_semantics(self) -> None:
        self.assertEqual(string_match_all("Alpha only", ["alpha", "beta"]), 0.5)
        self.assertEqual(string_match_all("ALPHA BETA", ["alpha", "beta"]), 1.0)
        self.assertEqual(
            aggregate_string_match_all(
                ["alpha beta", "alpha"], [["alpha", "beta"], ["alpha", "beta"]]
            ),
            75.0,
        )

    def test_zero_shot_renderers_cover_all_preregistered_tasks(self) -> None:
        examples = {
            "hellaswag": {
                "ind": 1,
                "ctx_a": "A person",
                "ctx_b": "starts cooking",
                "endings": ["one", "two", "three", "four"],
                "label": "2",
            },
            "arc_easy": {
                "id": "arc1",
                "question": "Which?",
                "choices": {
                    "text": ["a", "b", "c", "d"],
                    "label": ["A", "B", "C", "D"],
                },
                "answerKey": "C",
            },
            "piqa": {"goal": "Do this", "sol1": "first", "sol2": "second", "label": 1},
            "winogrande": {
                "sentence": "The trophy does not fit because _ is too large.",
                "option1": "the trophy",
                "option2": "the case",
                "answer": "1",
            },
        }
        for task, row in examples.items():
            with self.subTest(task=task):
                rendered = render_example(task, row, row_index=0)
                self.assertTrue(rendered.prompt)
                self.assertGreaterEqual(len(rendered.choices), 2)
                self.assertGreaterEqual(rendered.gold, 0)
                self.assertLess(rendered.gold, len(rendered.choices))

    def test_adaptation_and_continuation_masks_are_separate(self) -> None:
        rows = [[10, 11, 12, 13]]
        batch = make_padded_batch(
            rows,
            choice_starts=None,
            context_length=4,
            pad_token_id=0,
        )
        score_masks = make_continuation_score_masks(
            rows,
            choice_starts=[2],
            context_length=4,
        )
        self.assertEqual(batch.loss_masks.tolist(), [[1, 1, 1, 0]])
        self.assertEqual(score_masks.tolist(), [[0, 1, 1, 0]])


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tier1TargetsTest(unittest.TestCase):
    """Conditions without a revision-v3 target are registered but every runner refuses them."""

    def setUp(self) -> None:
        self.config_path = REPO_ROOT / "configs/research/tier1_eval.yaml"
        self.config = load_tier1_config(self.config_path)

    def test_only_the_fa_seed_has_a_target(self) -> None:
        self.assertEqual(pending_conditions(self.config, ["FA_SEED_125M"]), [])
        spec = condition_spec(self.config, "FA_SEED_125M")
        self.assertEqual(
            (spec.source_paper_run_id, spec.source_stage_id, spec.source_run_id, spec.checkpoint_step),
            ("protocol_r_125m_main_v1", "S0_PRETRAIN_FA_125M", "pretrain-125m-fa", 4799),
        )
        self.assertEqual(
            sorted(pending_conditions(self.config)),
            sorted(set(self.config["conditions"]) - {"FA_SEED_125M"}),
        )

    def test_config_names_no_withdrawn_checkpoint_or_private_repo(self) -> None:
        text = self.config_path.read_text(encoding="utf-8")
        for forbidden in (
            "revision_v2_e1_paired_v1",
            "revision_v2_e3_frontier_v1",
            "protocol_r_760m_author_seed_v1",
            "ext-125m-e2e-32K",
            "repo_id: Luxel",
            "Luxel/",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, text)

    def test_pending_conditions_fail_closed_with_the_reason(self) -> None:
        for condition in ("S2_125M_SEED001", "S2_MINUS_125M_SEED001", "S3_125M", "S2_760M"):
            with self.subTest(condition=condition):
                with self.assertRaisesRegex(PendingTargetError, "pending a dated Tier-1 addendum"):
                    condition_spec(self.config, condition)
        with self.assertRaisesRegex(PendingTargetError, "S3_125M"):
            require_targets(self.config, ["FA_SEED_125M", "S3_125M"])

    def test_pending_entries_may_not_name_a_checkpoint(self) -> None:
        broken = {
            **self.config,
            "conditions": {
                "X": {"scale": "125m", "architecture": "ttt_e2e", "target": "pending", "source_run_id": "old"}
            },
        }
        with self.assertRaisesRegex(ValueError, "must not name a checkpoint"):
            validate_tier1_config(broken)

    def test_registry_runner_refuses_pending_conditions(self) -> None:
        runner = _load_script("tier1_registry_91", "91_run_tier1_registry.py")
        stages = {stage.stage_id: stage for stage in load_tier1_registry(REPO_ROOT / "configs/research/tier1_registry.yaml")}
        with tempfile.TemporaryDirectory() as td:
            args = runner.parse_args(["--restore-source-dir", td])
            fa_restore = runner._command(stages["T1_RESTORE_FA_SEED_125M"], args, self.config)
            self.assertIn("--source-dir", fa_restore)
            self.assertNotIn("--repo-id", fa_restore)
            self.assertTrue(fa_restore[1].endswith("46_restore_stage.py"))
            with self.assertRaises(PendingTargetError):
                runner._command(stages["T1_RESTORE_S2_125M"], args, self.config)
            with self.assertRaisesRegex(ValueError, "--restore-source-dir"):
                runner._command(stages["T1_RESTORE_FA_SEED_125M"], runner.parse_args([]), self.config)

            out = io.StringIO()
            preflight_out = Path(td) / "preflight.json"
            with contextlib.redirect_stdout(out):
                code = runner.main(["--restore-source-dir", td, "--preflight-out", str(preflight_out)])
            self.assertEqual(code, 1)
            self.assertIn("Refusing to run", out.getvalue())
            report = json.loads(preflight_out.read_text(encoding="utf-8"))
            failed = {
                row["stage_id"]
                for row in report["checks"]
                if row["check"] == "condition_target_available" and row["status"] == "FAIL"
            }
            self.assertIn("T1_RESTORE_S2_125M", failed)
            self.assertNotIn("T1_RESTORE_FA_SEED_125M", failed)

    def test_report_builder_refuses_pending_conditions(self) -> None:
        reports = _load_script("tier1_reports_92_targets", "92_make_tier1_reports.py")
        conditions = reports.reported_conditions(
            self.config, include_optional_t1b=False, include_760m_t1c=False
        )
        self.assertIn("S2_125M_SEED001", conditions)
        with self.assertRaises(PendingTargetError):
            require_targets(self.config, conditions)


class Tier1ReportFigureTest(unittest.TestCase):
    def test_t1b_figure_renders_from_synthetic_traces(self) -> None:
        reports = _load_script("tier1_reports_92", "92_make_tier1_reports.py")
        conditions = ["S2_125M_SEED001", "S2_MINUS_125M_SEED001"]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rng = np.random.default_rng(0)
            for offset, condition in enumerate(conditions):
                run_dir = root / "t1b" / condition
                run_dir.mkdir(parents=True)
                (run_dir / "run_manifest.json").write_text(
                    json.dumps({"status": "succeeded", "manifest_sha256": "m", "checkpoint_step": 7}),
                    encoding="utf-8",
                )
                np.savez(
                    run_dir / "fast_weight_trace_step_00000007.npz",
                    example_ids=np.asarray(["a", "b", "c"]),
                    applied_update_l2=rng.random((3, 4, 2)) + offset,
                )
            meta = reports._plot_t1b_fast_weights(root / "figures", root, conditions=conditions)
            self.assertTrue((root / "figures" / "fast_weight_update_magnitude.pdf").is_file())
            self.assertTrue((root / "figures" / "fast_weight_update_magnitude.png").is_file())
            self.assertEqual(meta["S2_125M_SEED001"]["chunks"], 4)
            self.assertGreater(
                meta["S2_MINUS_125M_SEED001"]["mean_applied_update_l2"],
                meta["S2_125M_SEED001"]["mean_applied_update_l2"],
            )


@unittest.skipUnless(HAS_JAX, "jax not installed")
class Tier1JaxPathTest(unittest.TestCase):
    def _meta_fixture(self):
        import equinox as eqx
        import jax.numpy as jnp
        import jax.random as jrandom

        from ttt.config import Config
        from ttt.jax_runtime.model.data import Batch
        from ttt.jax_runtime.model.transformer import MetaModel

        cfg = Config()
        cfg.training.train_mode = cfg.training.TrainMode.meta
        cfg.training.seq_length = 8
        cfg.training.spec_inner = [
            "language_model.**.suffix_blocks.feed_forward_prime.**"
        ]
        cfg.training.inner_remat_freq = 1
        cfg.training.optimizer_inner.lr = 0.1
        cfg.training.optimizer_inner.clip_gradient = 1.0
        cfg.model.vocab_size = 32
        cfg.model.output_size = 32
        cfg.model.hidden_size = 8
        cfg.model.intermediate_size = 16
        cfg.model.num_hidden_layers = 2
        cfg.model.num_attention_heads = 2
        cfg.model.seq_modeling_block = "SWA"
        cfg.model.mini_batch_size = 4
        cfg.model.sliding_window_size = 4
        cfg.model.seq_len = 16
        cfg.model.suffix_len = 1
        cfg.model.prime = True
        cfg.model.force_flash = False
        cfg.model.compute_dtype = "fp32"
        cfg.model.param_dtype = "fp32"
        cfg.model.state_dtype = "fp32"
        model, state = eqx.nn.make_with_state(MetaModel)(cfg, key=jrandom.PRNGKey(0))
        tokens = jnp.asarray([1, 3, 4, 5, 6, 7, 8, 9], dtype=jnp.int32)
        seq = Batch(
            input_ids=tokens,
            target_tokens=jnp.roll(tokens, -1),
            loss_masks=jnp.asarray([1, 1, 1, 1, 1, 1, 1, 0], dtype=jnp.int32),
            attention_mask=None,
            position_ids=jnp.arange(8, dtype=jnp.int32),
        )
        return model, state, seq

    def test_streaming_swa_fragment_matches_native_full_chunk(self) -> None:
        import equinox as eqx
        import jax.numpy as jnp
        import jax.random as jrandom

        from ttt.config import ModelConfig
        from ttt.jax_runtime.meta_inference import _swa_decode_step
        from ttt.jax_runtime.model.attention import SWA
        from ttt.jax_runtime.model.data import Batch
        from ttt.utils.jax_utils import clone_pytree

        cfg = ModelConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            seq_modeling_block="SWA",
            mini_batch_size=4,
            sliding_window_size=4,
            seq_len=16,
            vocab_size=32,
            force_flash=False,
            compute_dtype="fp32",
        )
        swa, state = eqx.nn.make_with_state(SWA)(cfg, key=jrandom.PRNGKey(0))
        first = jrandom.normal(jrandom.PRNGKey(1), (4, 8))
        second = jrandom.normal(jrandom.PRNGKey(2), (4, 8))
        seq = Batch(
            input_ids=jnp.zeros((4,), dtype=jnp.int32),
            target_tokens=jnp.zeros((4,), dtype=jnp.int32),
            loss_masks=jnp.ones((4,), dtype=jnp.int32),
            attention_mask=None,
            position_ids=None,
        )
        _out, state = swa(first, seq, state, is_prefix=False)
        expected, expected_state = swa(
            second, seq, clone_pytree(state), is_prefix=False
        )
        actual, actual_state = _swa_decode_step(swa, second, clone_pytree(state))
        self.assertTrue(bool(jnp.allclose(expected, actual, atol=2e-5, rtol=2e-5)))
        expected_k, expected_v = expected_state.get(swa.kv_cache_index)
        actual_k, actual_v = actual_state.get(swa.kv_cache_index)
        self.assertTrue(bool(jnp.allclose(expected_k, actual_k)))
        self.assertTrue(bool(jnp.allclose(expected_v, actual_v)))

    def test_fast_weight_probe_shapes_and_finiteness(self) -> None:
        import jax
        from jax.sharding import Mesh

        from ttt.jax_runtime.fast_weight_probe import probe_sequence

        model, state, seq = self._meta_fixture()
        mesh = Mesh(
            np.asarray(jax.devices()).reshape(1, -1), ("data_parallel", "state")
        )
        with mesh:
            diagnostics = probe_sequence(model, seq, state)
        self.assertEqual(np.asarray(diagnostics["raw_grad_l2"]).shape, (2, 1))
        self.assertEqual(np.asarray(diagnostics["applied_update_l2"]).shape, (2, 1))
        self.assertTrue(bool(np.all(np.asarray(diagnostics["all_finite"]))))

    def test_meta_generation_freezes_fast_weights_during_decode(self) -> None:
        from ttt.jax_runtime.meta_inference import generate_one

        model, state, seq = self._meta_fixture()
        generated = generate_one(model, seq, state, max_new_tokens=3, eos_token_id=2)
        values = np.asarray(generated)
        self.assertEqual(values.shape, (3,))
        self.assertTrue(bool(np.all((0 <= values) & (values < 32))))


if __name__ == "__main__":
    unittest.main()
