from __future__ import annotations

import math
import unittest
from pathlib import Path

import yaml

from ttt.research.orchestrator import OrchestratorOptions, build_train_command
from ttt.research.preregistration import (
    assert_fresh_optimizer_state,
    assert_matching_extension_seed_policy,
    assert_params_only_restore,
    assert_resume_lineage_direction,
    bridge_effect_from_losses,
    classify_bridge_effect,
    compare_fingerprints,
    compare_resolved_configs,
    exact_mcnemar,
    paired_binary_counts,
    paired_deltas,
    wilson_interval,
    ConfidenceInterval,
)
from ttt.research.registry import load_registry
from ttt.research.types import BudgetSpec


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "configs" / "research" / "warmstart_registry.yaml"


def _load_yaml(rel_path: str) -> dict:
    payload = yaml.safe_load((REPO_ROOT / rel_path).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


class RevisionV2PreregistrationTest(unittest.TestCase):
    def test_preregistration_document_contains_load_bearing_rules(self) -> None:
        text = (REPO_ROOT / "PREREGISTRATION_REVISION_V2.md").read_text(encoding="utf-8")
        required_phrases = [
            "delta_bridge_s = loss(S2_MINUS_s) - loss(S2_s)",
            "Run five paired seeds up front",
            "Bridge helps",
            "Bridge harmful",
            "Bridge negligible",
            "Inconclusive",
            "re-evaluate the canonical comparison checkpoints",
            "Wilson binomial intervals",
            "Exact McNemar test",
            "effective_n < 50",
            "S2_125M vs S2_MINUS_125M",
            "scale = 125M",
            "context_length = 32768",
            "training.model_seed",
            "training.data_seed",
            "training.resume_exp_name=adapt-125m-e2e-8K-from-fa",
            "training.resume_exp_name=pretrain-125m-fa",
            "Golden-Plan Protection",
        ]
        for phrase in required_phrases:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_bridge_effect_uses_paired_deltas_and_preregistered_thresholds(self) -> None:
        deltas = paired_deltas([3.20, 3.30], [3.00, 3.10])
        self.assertAlmostEqual(deltas[0], 0.2)
        self.assertAlmostEqual(deltas[1], 0.2)
        helps = classify_bridge_effect(
            mean_delta=0.14,
            ci95=ConfidenceInterval(low=0.03, high=0.22),
        )
        harmful = classify_bridge_effect(
            mean_delta=-0.14,
            ci95=ConfidenceInterval(low=-0.22, high=-0.03),
        )
        negligible = classify_bridge_effect(
            mean_delta=0.01,
            ci95=ConfidenceInterval(low=-0.04, high=0.06),
        )
        inconclusive = classify_bridge_effect(
            mean_delta=0.12,
            ci95=ConfidenceInterval(low=-0.02, high=0.22),
        )
        self.assertEqual(helps, "helps")
        self.assertEqual(harmful, "harmful")
        self.assertEqual(negligible, "negligible")
        self.assertEqual(inconclusive, "inconclusive")

        result = bridge_effect_from_losses(
            s2_minus_losses=[3.24, 3.25, 3.26, 3.27, 3.28],
            s2_losses=[3.10, 3.11, 3.12, 3.13, 3.14],
            n_resamples=500,
            seed=7,
        )
        self.assertEqual(result.decision, "helps")
        self.assertAlmostEqual(result.mean_delta, 0.14)
        self.assertGreater(result.ci95.low, 0.0)

    def test_wilson_interval_golden_value(self) -> None:
        interval = wilson_interval(successes=5, n=10)
        self.assertTrue(math.isclose(interval.low, 0.23659309051256394, rel_tol=0, abs_tol=1e-12))
        self.assertTrue(math.isclose(interval.high, 0.7634069094874361, rel_tol=0, abs_tol=1e-12))

    def test_exact_mcnemar_golden_value_and_pairing(self) -> None:
        a = {
            "ex1": True,
            "ex2": True,
            "ex3": False,
            "ex4": False,
            "ex5": True,
        }
        b = {
            "ex1": False,
            "ex2": True,
            "ex3": True,
            "ex4": False,
            "ex5": False,
        }
        both_correct, a_only, b_only, both_wrong = paired_binary_counts(a, b)
        self.assertEqual((both_correct, a_only, b_only, both_wrong), (1, 2, 1, 1))

        result = exact_mcnemar(b=8, c=2, alternative="greater")
        self.assertEqual(result.effective_n, 10)
        self.assertTrue(math.isclose(result.p_value, 0.0546875, rel_tol=0, abs_tol=1e-12))

        with self.assertRaisesRegex(ValueError, "Unpaired example ids"):
            paired_binary_counts({"ex1": True}, {"ex2": True})

    def test_existing_direct_e2e_config_is_valid_s2_minus_candidate(self) -> None:
        bridge = _load_yaml("configs/experiment/125m/pretrained/ext-125m-e2e-32K-from-fa-bridge.yaml")
        direct = _load_yaml("configs/experiment/125m/pretrained/ext-125m-e2e-32K-from-fa-direct.yaml")

        comparison = compare_resolved_configs(bridge, direct)
        self.assertTrue(comparison.equal, comparison.differences)
        self.assertEqual(comparison.differences, ())

        assert_fresh_optimizer_state(bridge)
        assert_fresh_optimizer_state(direct)
        assert_params_only_restore(direct)
        self.assertEqual(direct["training"]["resume_exp_name"], "pretrain-125m-fa")
        self.assertEqual(bridge["training"]["resume_exp_name"], "adapt-125m-e2e-8K-from-fa")
        assert_resume_lineage_direction(
            s2_extension_config=bridge,
            s2_minus_config=direct,
            expected_s2_parent="adapt-125m-e2e-8K-from-fa",
            expected_s2_minus_parent="pretrain-125m-fa",
        )

    def test_resume_lineage_direction_fails_if_direct_uses_bridge_parent(self) -> None:
        bridge = _load_yaml("configs/experiment/125m/pretrained/ext-125m-e2e-32K-from-fa-bridge.yaml")
        direct = _load_yaml("configs/experiment/125m/pretrained/ext-125m-e2e-32K-from-fa-direct.yaml")
        direct["training"]["resume_exp_name"] = "adapt-125m-e2e-8K-from-fa"
        with self.assertRaisesRegex(ValueError, "S2-minus extension to resume"):
            assert_resume_lineage_direction(
                s2_extension_config=bridge,
                s2_minus_config=direct,
                expected_s2_parent="adapt-125m-e2e-8K-from-fa",
                expected_s2_minus_parent="pretrain-125m-fa",
            )

    def test_extension_seed_policy_requires_explicit_matched_pair_seeds(self) -> None:
        s2 = {"training": {"model_seed": 3, "data_seed": 3}}
        s2_minus = {"training": {"model_seed": 3, "data_seed": 3}}
        assert_matching_extension_seed_policy(
            s2_extension_config=s2,
            s2_minus_config=s2_minus,
        )

        with self.assertRaisesRegex(ValueError, "Expected explicit S2 extension training.model_seed"):
            assert_matching_extension_seed_policy(
                s2_extension_config={"training": {"data_seed": 3}},
                s2_minus_config=s2_minus,
            )

        with self.assertRaisesRegex(ValueError, "Expected matched extension training.data_seed"):
            assert_matching_extension_seed_policy(
                s2_extension_config=s2,
                s2_minus_config={"training": {"model_seed": 3, "data_seed": 4}},
            )

    def test_config_comparison_fails_on_non_lineage_difference(self) -> None:
        left = {
            "training": {
                "exp_name": "s2",
                "resume_exp_name": "adapt",
                "seq_length": 32768,
                "optimizer_outer": {"lr": 4e-4},
            }
        }
        right = {
            "training": {
                "exp_name": "s2-minus",
                "resume_exp_name": "pretrain",
                "seq_length": 32768,
                "optimizer_outer": {"lr": 5e-4},
            }
        }
        comparison = compare_resolved_configs(left, right)
        self.assertFalse(comparison.equal)
        self.assertEqual(comparison.differences, ("training.optimizer_outer.lr",))

    def test_fingerprint_comparison_enforces_matched_surface(self) -> None:
        base = {
            "dataset": {
                "dataset_id": "books3",
                "split": "train",
                "sha256": "abc",
                "num_tokens": 100,
                "tokenizer_id": "tok",
                "tokenizer_revision": "rev",
            }
        }
        same = {"dataset": dict(base["dataset"])}
        different = {"dataset": {**base["dataset"], "sha256": "def"}}

        self.assertTrue(compare_fingerprints(base, same).equal)
        mismatch = compare_fingerprints(base, different, label_left="S2", label_right="S2_MINUS")
        self.assertFalse(mismatch.equal)
        self.assertEqual(mismatch.differences, ("sha256: S2='abc' S2_MINUS='def'",))

    def test_golden_plan_for_canonical_125m_stages(self) -> None:
        registry = load_registry(REGISTRY_PATH)
        stages = registry.stage_map()
        opts = OrchestratorOptions(
            deploy="interactive",
            runtime_mode="jax_train",
            exp_dir=Path("/tmp/revision-v2/experiments"),
            checkpoint_root=Path("/tmp/revision-v2/checkpoints"),
            profile_root=Path("/tmp/revision-v2/profiles"),
            dclm_root=Path("/tmp/revision-v2/dclm"),
            books_root=Path("/tmp/revision-v2/books"),
            exp_folder="paper",
            wandb_entity="none",
            wandb_project="none",
            wandb_key="none",
            ext_global_batch_size=32,
            seq_length=32768,
            paper_run_id="revision_v2_golden",
        )
        budget = BudgetSpec(budget_id="golden", pretrain_steps=4800, adapt_steps=480, ext_steps=120)

        expected_by_stage = {
            "S1_125M": [
                "uv",
                "run",
                "--exact",
                "train",
                "+deploy=interactive",
                "+experiment=125m/pretrained/ext-125m-swa-32K-from-fa",
                "training.exp_folder=paper",
                "training.exp_dir=/tmp/revision-v2/experiments",
                "training.exp_name=ext-125m-swa-32K-from-fa",
                "training.total_steps=120",
                "training.runtime_mode=jax_train",
                "training.wandb_entity=none",
                "training.wandb_project=none",
                "training.wandb_key=none",
                "deploy_paths.data.dclm_filter_8k=/tmp/revision-v2/dclm",
                "deploy_paths.data.books3=/tmp/revision-v2/books",
                "deploy_paths.checkpoint=/tmp/revision-v2/checkpoints",
                "training.checkpoint_path=/tmp/revision-v2/checkpoints",
                "training.paper_run_id=revision_v2_golden",
                "training.stage_id=S1_125M",
                "training.run_id=ext-125m-swa-32K-from-fa",
                "training.global_batch_size=32",
                "training.seq_length=32768",
                "training.log_wandb=false",
                "training.load_part=params",
                "training.resume_exp_name=pretrain-125m-fa",
            ],
            "S2_125M": [
                "uv",
                "run",
                "--exact",
                "train",
                "+deploy=interactive",
                "+experiment=125m/pretrained/ext-125m-e2e-32K-from-fa-bridge",
                "training.exp_folder=paper",
                "training.exp_dir=/tmp/revision-v2/experiments",
                "training.exp_name=ext-125m-e2e-32K-from-fa-bridge",
                "training.total_steps=120",
                "training.runtime_mode=jax_train",
                "training.wandb_entity=none",
                "training.wandb_project=none",
                "training.wandb_key=none",
                "deploy_paths.data.dclm_filter_8k=/tmp/revision-v2/dclm",
                "deploy_paths.data.books3=/tmp/revision-v2/books",
                "deploy_paths.checkpoint=/tmp/revision-v2/checkpoints",
                "training.checkpoint_path=/tmp/revision-v2/checkpoints",
                "training.paper_run_id=revision_v2_golden",
                "training.stage_id=S2_125M",
                "training.run_id=ext-125m-e2e-32K-from-fa-bridge",
                "training.global_batch_size=32",
                "training.seq_length=32768",
                "training.log_wandb=false",
            ],
            "S3_125M": [
                "uv",
                "run",
                "--exact",
                "train",
                "+deploy=interactive",
                "+experiment=125m/extension/ext-125m-e2e-32K",
                "training.exp_folder=paper",
                "training.exp_dir=/tmp/revision-v2/experiments",
                "training.exp_name=ext-125m-e2e-32K",
                "training.total_steps=120",
                "training.runtime_mode=jax_train",
                "training.wandb_entity=none",
                "training.wandb_project=none",
                "training.wandb_key=none",
                "deploy_paths.data.dclm_filter_8k=/tmp/revision-v2/dclm",
                "deploy_paths.data.books3=/tmp/revision-v2/books",
                "deploy_paths.checkpoint=/tmp/revision-v2/checkpoints",
                "training.checkpoint_path=/tmp/revision-v2/checkpoints",
                "training.paper_run_id=revision_v2_golden",
                "training.stage_id=S3_125M",
                "training.run_id=ext-125m-e2e-32K",
                "training.global_batch_size=32",
                "training.seq_length=32768",
                "training.log_wandb=false",
                "training.load_part=params",
                "training.resume_exp_name=pretrain-125m-e2e",
            ],
        }

        for stage_id, expected in expected_by_stage.items():
            with self.subTest(stage_id=stage_id):
                stage = stages[stage_id]
                command = build_train_command(
                    stage=stage,
                    opts=opts,
                    steps=budget.ext_steps,
                    run_id=stage.exp_name,
                )
                self.assertEqual(command, expected)


if __name__ == "__main__":
    unittest.main()
