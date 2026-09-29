from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from ttt.research.registry import load_registry
from ttt.research.warmstart_validity import (
    check_registry_warmstart_shapes,
    compare_model_sections,
    render_shape_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "configs" / "research" / "warmstart_registry.yaml"
HAS_HYDRA = importlib.util.find_spec("hydra") is not None


class CompareModelSectionsTest(unittest.TestCase):
    def test_ffn_width_change_is_an_error(self) -> None:
        parent = {"intermediate_size": 2048, "hidden_size": 768, "prime": False, "suffix_len": 0, "rope_theta": 5e5}
        child = {"intermediate_size": 1664, "hidden_size": 768, "prime": True, "suffix_len": 3, "rope_theta": 5e5}
        findings = compare_model_sections(stage_id="S2_ADAPT", parent_stage_id="S0_PRETRAIN_FA", parent_model=parent, child_model=child)
        errors = [f for f in findings if f.severity == "error"]
        infos = [f for f in findings if f.severity == "info"]
        self.assertEqual([f.field_name for f in errors], ["model.intermediate_size"])
        self.assertEqual({f.field_name for f in infos}, {"model.prime", "model.suffix_len"})

    def test_rope_theta_change_is_a_warning(self) -> None:
        parent = {"intermediate_size": 2048, "rope_theta": 5e5}
        child = {"intermediate_size": 2048, "rope_theta": 2e6}
        findings = compare_model_sections(stage_id="S0", parent_stage_id="S0_PRETRAIN_FA", parent_model=parent, child_model=child)
        self.assertEqual([(f.severity, f.field_name) for f in findings], [("warning", "model.rope_theta")])

    def test_declared_rope_theta_change_is_info(self) -> None:
        parent = {"intermediate_size": 2048, "rope_theta": 5e5}
        child = {"intermediate_size": 2048, "rope_theta": 2e6}
        findings = compare_model_sections(
            stage_id="S0",
            parent_stage_id="S0_PRETRAIN_FA",
            parent_model=parent,
            child_model=child,
            intended_changes=["model.rope_theta"],
        )
        self.assertEqual([(f.severity, f.field_name) for f in findings], [("info", "model.rope_theta")])

    def test_declaring_a_shape_change_does_not_excuse_it(self) -> None:
        parent = {"intermediate_size": 2048}
        child = {"intermediate_size": 1664}
        findings = compare_model_sections(
            stage_id="S2_ADAPT",
            parent_stage_id="S0_PRETRAIN_FA",
            parent_model=parent,
            child_model=child,
            intended_changes=["model.intermediate_size"],
        )
        self.assertEqual([(f.severity, f.field_name) for f in findings], [("error", "model.intermediate_size")])

    def test_prime_width_change_between_prime_models_is_an_error(self) -> None:
        parent = {"intermediate_size": 2048, "prime": True, "suffix_len": 3, "prime_intermediate_size": 1664}
        child = {"intermediate_size": 2048, "prime": True, "suffix_len": 3, "prime_intermediate_size": 2048}
        findings = compare_model_sections(stage_id="S2", parent_stage_id="S2_ADAPT", parent_model=parent, child_model=child)
        errors = [(f.severity, f.field_name) for f in findings if f.severity == "error"]
        self.assertEqual(errors, [("error", "model.prime_intermediate_size")])

    def test_prime_width_default_matches_explicit_value(self) -> None:
        # None means "same as intermediate_size", so these two prime models agree.
        parent = {"intermediate_size": 1664, "prime": True, "suffix_len": 3, "prime_intermediate_size": None}
        child = {"intermediate_size": 1664, "prime": True, "suffix_len": 3, "prime_intermediate_size": 1664}
        findings = compare_model_sections(stage_id="S3", parent_stage_id="S3_PRETRAIN", parent_model=parent, child_model=child)
        self.assertEqual([f for f in findings if f.severity == "error"], [])


@unittest.skipUnless(HAS_HYDRA, "hydra not installed")
class RegistryWarmStartShapeTest(unittest.TestCase):
    def test_every_params_restoring_stage_keeps_inherited_shapes(self) -> None:
        registry = load_registry(REGISTRY_PATH)
        report = check_registry_warmstart_shapes(registry=registry, repo_root=REPO_ROOT)
        rendered = render_shape_report(report)
        self.assertTrue(report.ok, rendered)
        checked = {pair[0] for pair in report.checked_pairs}
        # The conversions that lost their FFN weights in revision v2 must be covered,
        # and so must the parameter-matched scratch control.
        for stage_id in ("S1_125M", "S2_ADAPT_125M", "S2_MINUS_125M", "S2_125M", "S0_125M", "S3W_125M"):
            self.assertIn(stage_id, checked, rendered)
        # No parent/child pair may change a behaviour field such as rope_theta
        # unless the child experiment declares it.
        warnings = [f for f in report.findings if f.severity == "warning"]
        self.assertEqual(warnings, [], rendered)
        # S0 raises the RoPE base for the 32K extension (the published recipe) and says so.
        s0_rope = [f for f in report.findings if f.stage_id == "S0_125M" and f.field_name == "model.rope_theta"]
        self.assertEqual([(f.severity, f.parent_value, f.child_value) for f in s0_rope], [("info", 500000.0, 2000000.0)], rendered)


if __name__ == "__main__":
    unittest.main()
