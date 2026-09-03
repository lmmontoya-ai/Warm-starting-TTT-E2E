from __future__ import annotations

import unittest

from ttt.jax_runtime.warmstart_guard import (
    WarmStartValidationError,
    build_coverage_report,
    check_initial_loss,
    check_restore_report,
    classify_param_path,
)


LEAVES = [
    (".language_model.model.wte.embedding", 1000),
    (".language_model.model.h.blocks.seq_modeling_block.wq.weight", 400),
    (".language_model.model.h.blocks.feed_forward.w1.weight", 600),
    (".language_model.model.h.blocks.feed_forward.w2.weight", 600),
    (".language_model.model.h.prime_storage.feed_forward_prime.w1.weight", 200),
    (".language_model.model.h.blocks.ffn_norm.weight", 10),
]


class CoverageReportTest(unittest.TestCase):
    def test_classify_param_path(self) -> None:
        self.assertEqual(classify_param_path(LEAVES[0][0]), "embedding")
        self.assertEqual(classify_param_path(LEAVES[1][0]), "attention")
        self.assertEqual(classify_param_path(LEAVES[2][0]), "ffn")
        self.assertEqual(classify_param_path(LEAVES[4][0]), "ffn_prime")
        self.assertEqual(classify_param_path(LEAVES[5][0]), "norm")

    def test_full_restore_report(self) -> None:
        report = build_coverage_report(mode="exact", target_leaves=LEAVES, missed=[], mismatched=[])
        self.assertEqual(report.total_params, 2810)
        self.assertEqual(report.restored_params, 2810)
        self.assertEqual(report.fresh_params, 0)
        self.assertEqual(report.fresh_fraction, 0.0)
        check_restore_report(report, allow_shape_mismatch=False, max_new_param_fraction=0.25)

    def test_shape_mismatch_is_counted_and_rejected(self) -> None:
        mismatched = [
            f"{LEAVES[2][0]}: checkpoint=(12, 768, 2048) target=(12, 768, 1664)",
            f"{LEAVES[3][0]}: checkpoint=(12, 2048, 768) target=(12, 1664, 768)",
        ]
        report = build_coverage_report(
            mode="fallback_partial",
            target_leaves=LEAVES,
            missed=[LEAVES[4][0]],
            mismatched=mismatched,
        )
        self.assertEqual(report.mismatched_params, 1200)
        self.assertEqual(report.missed_params, 200)
        self.assertEqual(report.restored_params, 1410)
        self.assertEqual(report.by_component["ffn"], {"mismatched": 1200})
        self.assertEqual(report.by_component["ffn_prime"], {"missed": 200})
        with self.assertRaises(WarmStartValidationError) as ctx:
            check_restore_report(report, allow_shape_mismatch=False, max_new_param_fraction=0.25)
        self.assertIn("intermediate_size", str(ctx.exception))
        # Explicit opt-in accepts the partial warm start.
        check_restore_report(report, allow_shape_mismatch=True, max_new_param_fraction=0.25)

    def test_key_path_drift_is_rejected(self) -> None:
        report = build_coverage_report(
            mode="fallback_partial",
            target_leaves=LEAVES,
            missed=[path for path, _ in LEAVES],
            mismatched=[],
        )
        self.assertEqual(report.restored_params, 0)
        with self.assertRaises(WarmStartValidationError) as ctx:
            check_restore_report(report, allow_shape_mismatch=False, max_new_param_fraction=0.25)
        self.assertIn("no counterpart", str(ctx.exception))

    def test_to_dict_and_summary(self) -> None:
        report = build_coverage_report(mode="exact", target_leaves=LEAVES, missed=[], mismatched=[])
        payload = report.to_dict()
        self.assertEqual(payload["mode"], "exact")
        self.assertEqual(payload["restored_fraction"], 1.0)
        self.assertTrue(any("restore mode=exact" in line for line in report.summary_lines()))


class InitialLossGateTest(unittest.TestCase):
    def test_random_init_loss_is_rejected(self) -> None:
        with self.assertRaises(WarmStartValidationError):
            check_initial_loss(11.9, max_initial_loss=7.0, step=0)

    def test_restored_loss_passes(self) -> None:
        check_initial_loss(4.2, max_initial_loss=7.0, step=0)

    def test_gate_can_be_disabled(self) -> None:
        check_initial_loss(11.9, max_initial_loss=0.0, step=0)


if __name__ == "__main__":
    unittest.main()
