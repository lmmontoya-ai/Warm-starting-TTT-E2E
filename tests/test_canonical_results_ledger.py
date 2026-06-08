from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER = REPO_ROOT / "CANONICAL_RESULTS.md"


def _read_csv(path: str) -> list[dict[str, str]]:
    with (REPO_ROOT / path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class CanonicalResultsLedgerTest(unittest.TestCase):
    def test_ledger_exists_and_names_required_sources(self) -> None:
        text = LEDGER.read_text(encoding="utf-8")
        required_paths = [
            "reports/paper/protocol_r_125m_main_v1/tables/stage_summary_loss_mean.csv",
            "reports/paper/protocol_r_125m_main_v1/tables/run_inventory.csv",
            "reports/paper/protocol_r_125m_main_v1/tables/warmstart_core_deltas.csv",
            "reports/paper/warmstart_paper_v1/plot_data/figure3_continuation_frontier.csv",
            "reports/paper/protocol_r_760m_author_seed_v1/tables/stage_summary_loss_mean.csv",
            "reports/paper/protocol_r_760m_author_seed_v1/tables/run_inventory.csv",
            "reports/paper/protocol_r_760m_author_seed_v1/tables/s2_s3_warmstart_tax.csv",
            "scripts/75_make_paper_plots.py",
            "paper/plots/plot_manifest.json",
        ]
        for rel_path in required_paths:
            with self.subTest(path=rel_path):
                self.assertIn(rel_path, text)
                self.assertTrue((REPO_ROOT / rel_path).exists())

    def test_ledger_marks_known_stale_surfaces(self) -> None:
        text = LEDGER.read_text(encoding="utf-8")
        stale_paths = [
            "reports/paper/draft_v1.md",
            "reports/paper/protocol_r_125m_ablations_v1/iso_quality_summary.csv",
            "reports/paper/protocol_r_125m_ablations_v1/frontier.csv",
            "reports/paper/warmstart_paper_v1/plot_data/plot_data_manifest.json",
            "reports/paper/protocol_r_760m_author_seed_v1/launch/launcher_summary.json",
            "reports/paper/protocol_r_760m_eta_live_v1/eta_summary_combined.json",
        ]
        self.assertIn("Non-Authoritative Or Stale Surfaces", text)
        for rel_path in stale_paths:
            with self.subTest(path=rel_path):
                self.assertIn(rel_path, text)
                self.assertTrue((REPO_ROOT / rel_path).exists())

    def test_125m_headline_numbers_match_authoritative_tables(self) -> None:
        rows = _read_csv("reports/paper/protocol_r_125m_main_v1/tables/stage_summary_loss_mean.csv")
        losses = {row["stage_id"]: float(row["mean"]) for row in rows}
        self.assertEqual(losses["S0_125M"], 6.583984375)
        self.assertEqual(losses["S1_125M"], 6.5418243408203125)
        self.assertEqual(losses["S2_125M"], 3.917266845703125)
        self.assertEqual(losses["S3_125M"], 3.2729225158691406)
        self.assertAlmostEqual(losses["S2_125M"] - losses["S3_125M"], 0.6443443298339844)

        inventory = {
            row["stage_id"]: row
            for row in _read_csv("reports/paper/protocol_r_125m_main_v1/tables/run_inventory.csv")
        }
        warmstart_marginal = (
            float(inventory["S2_ADAPT_125M"]["gpu_hours"])
            + float(inventory["S2_125M"]["gpu_hours"])
        )
        warmstart_full = float(inventory["S0_PRETRAIN_FA_125M"]["gpu_hours"]) + warmstart_marginal
        scratch = (
            float(inventory["S3_PRETRAIN_E2E_125M"]["gpu_hours"])
            + float(inventory["S3_125M"]["gpu_hours"])
        )
        self.assertAlmostEqual(warmstart_marginal, 3.4656347024311414)
        self.assertAlmostEqual(warmstart_full, 22.608198793583757)
        self.assertAlmostEqual(scratch, 25.70727655243232)

    def test_continuation_uses_normalized_curve_not_stale_summary(self) -> None:
        rows = _read_csv("reports/paper/warmstart_paper_v1/plot_data/figure3_continuation_frontier.csv")
        iso_quality = [
            row
            for row in rows
            if row["mode"] == "iso_quality" and row["status"] == "succeeded"
        ]
        iso_total = [
            row
            for row in rows
            if row["mode"] == "iso_total_tokens" and row["status"] == "succeeded"
        ]
        self.assertEqual(max(int(row["extra_steps"]) for row in iso_quality), 1440)
        self.assertEqual(max(int(row["extra_steps"]) for row in iso_total), 960)
        terminal_s2 = next(row for row in iso_quality if row["extra_steps"] == "1440")
        terminal_s3 = next(row for row in iso_total if row["extra_steps"] == "960")
        self.assertEqual(float(terminal_s2["loss_ce_mean"]), 3.8657073974609375)
        self.assertEqual(float(terminal_s3["loss_ce_mean"]), 3.2623252868652344)

        stale_summary = _read_csv(
            "reports/paper/protocol_r_125m_ablations_v1/iso_quality_summary.csv"
        )[0]
        self.assertEqual(stale_summary["status"], "unknown")
        self.assertEqual(stale_summary["numeric_frontier_points"], "0")

    def test_760m_headline_numbers_match_authoritative_tables(self) -> None:
        rows = _read_csv(
            "reports/paper/protocol_r_760m_author_seed_v1/tables/stage_summary_loss_mean.csv"
        )
        losses = {row["stage_id"]: float(row["mean"]) for row in rows}
        self.assertEqual(losses["S2"], 2.9939842224121094)
        self.assertEqual(losses["S3"], 2.675201416015625)
        self.assertAlmostEqual(losses["S2"] - losses["S3"], 0.3187828063964844)

        manifest = json.loads((REPO_ROOT / "paper/plots/plot_manifest.json").read_text())
        caveats = " ".join(
            item
            for values in manifest["caveats"].values()
            for item in values
        )
        self.assertIn("760M full-branch cost is intentionally omitted", caveats)

    def test_revision_guardrails_are_explicit(self) -> None:
        text = LEDGER.read_text(encoding="utf-8")
        self.assertIn("current", text)
        self.assertIn("checkpoint-based float32", text)
        self.assertIn("matching dataset fingerprints", text)
        self.assertIn("new `paper_run_id`", text)
        self.assertIn("revision-v2 gates must use current-pipeline", text)


if __name__ == "__main__":
    unittest.main()
