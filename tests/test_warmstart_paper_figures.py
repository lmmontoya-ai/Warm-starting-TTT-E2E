from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from ttt.research.paper_figures import (
    _load_training_curve,
    load_figure_set_spec,
    prepare_warmstart_figure_data,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "configs" / "research" / "paper_figure_sets.yaml"
EXP_DIR = REPO_ROOT / "experiments"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class WarmstartPaperFiguresTest(unittest.TestCase):
    def test_load_figure_set_spec_resolves_cross_scale_policy(self) -> None:
        spec = load_figure_set_spec(SPEC_PATH, figure_set_id="warmstart_paper_v1")
        self.assertEqual(spec.output_report_id, "warmstart_paper_v1")
        self.assertEqual(spec.scales["125M"].main_paper_run_id, "protocol_r_125m_main_v1")
        self.assertEqual(spec.scales["125M"].stage_ids["S2"], "S2_125M")
        self.assertIn("figure4", spec.scales["125M"].enabled_figures)
        self.assertEqual(
            spec.scales["760M"].figure1_cost_policy,
            "local_only_executed",
        )
        self.assertIn("warmstart_full", spec.scales["760M"].figure1_omitted_points)
        self.assertEqual(
            spec.scales["760M"].figure1_omitted_points["warmstart_full"],
            "Omitted until measured seed GPU-hours exist for the author-provided 760M warm-start root.",
        )

    def test_prepare_plot_data_builds_expected_125m_rows(self) -> None:
        spec = load_figure_set_spec(SPEC_PATH, figure_set_id="warmstart_paper_v1")
        with tempfile.TemporaryDirectory() as td:
            plot_data_dir = Path(td) / "plot_data"
            manifest = prepare_warmstart_figure_data(
                repo_root=REPO_ROOT,
                exp_dir=EXP_DIR,
                spec=spec,
                plot_data_dir=plot_data_dir,
                strict=False,
                spec_path=SPEC_PATH,
            )

            figure1_rows = _read_csv(plot_data_dir / "figure1_cost_quality_points.csv")
            available_125m = {
                row["point_kind"]: row
                for row in figure1_rows
                if row["scale"] == "125M" and row["is_available"] == "True"
            }
            self.assertAlmostEqual(
                float(available_125m["warmstart_marginal"]["gpu_hours"]),
                3.4656347024311414,
                places=9,
            )
            self.assertAlmostEqual(
                float(available_125m["warmstart_full"]["gpu_hours"]),
                22.608198793583757,
                places=9,
            )
            self.assertAlmostEqual(
                float(available_125m["scratch"]["loss_mean"]),
                3.2729225158691406,
                places=12,
            )

            omitted_760m = {
                row["point_kind"]: row
                for row in figure1_rows
                if row["scale"] == "760M"
            }
            self.assertEqual(omitted_760m["warmstart_full"]["is_available"], "False")
            self.assertIn("measured seed GPU-hours", omitted_760m["warmstart_full"]["omission_reason"])

            figure3_rows = _read_csv(plot_data_dir / "figure3_continuation_frontier.csv")
            modes = {row["mode"] for row in figure3_rows}
            self.assertEqual(modes, {"iso_quality", "iso_total_tokens"})
            self.assertGreaterEqual(len(figure3_rows), 20)
            self.assertEqual(
                max(int(row["extra_steps"]) for row in figure3_rows if row["mode"] == "iso_quality"),
                1440,
            )
            self.assertEqual(
                max(int(row["extra_steps"]) for row in figure3_rows if row["mode"] == "iso_total_tokens"),
                960,
            )
            self.assertEqual(manifest["figures"]["figure3"]["available_scales"], ["125M"])

            figure4_rows = _read_csv(plot_data_dir / "figure4_extension_training_curves.csv")
            conditions = {row["condition"] for row in figure4_rows if row["scale"] == "125M"}
            self.assertEqual(conditions, {"S0", "S1", "S2", "S3"})

    def test_training_curve_parser_ignores_non_train_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            metrics_path = Path(td) / "metrics.jsonl"
            metrics_path.write_text(
                "\n".join(
                    [
                        json.dumps({"step": 0, "loss_ce": 4.0}),
                        json.dumps({"event": "checkpoint_saved", "step": 0, "checkpoint_save_seconds": 1.0}),
                        json.dumps({"step": 1, "loss_ce": 3.5}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            rows = _load_training_curve(metrics_path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["step"], 0.0)
            self.assertEqual(rows[1]["loss_ce_raw"], 3.5)

    def test_strict_prepare_fails_when_missing_scale_is_required(self) -> None:
        raw = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
        raw["figure_sets"]["warmstart_paper_v1"]["scales"]["760M"]["required_in_strict"] = [
            "figure1"
        ]
        with tempfile.TemporaryDirectory() as td:
            spec_path = Path(td) / "paper_figure_sets.yaml"
            spec_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            spec = load_figure_set_spec(spec_path, figure_set_id="warmstart_paper_v1")
            with self.assertRaises(FileNotFoundError):
                prepare_warmstart_figure_data(
                    repo_root=REPO_ROOT,
                    exp_dir=EXP_DIR,
                    spec=spec,
                    plot_data_dir=Path(td) / "plot_data",
                    strict=True,
                    spec_path=spec_path,
                )

    def test_prepare_and_render_scripts_write_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            plot_data_dir = Path(td) / "plot_data"
            figures_dir = Path(td) / "figures"
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "68_prepare_warmstart_figure_data.py"),
                    "--figure-set-id",
                    "warmstart_paper_v1",
                    "--plot-data-dir",
                    str(plot_data_dir),
                ],
                check=True,
                cwd=REPO_ROOT,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "69_make_warmstart_paper_figures.py"),
                    "--figure-set-id",
                    "warmstart_paper_v1",
                    "--plot-data-dir",
                    str(plot_data_dir),
                    "--figures-dir",
                    str(figures_dir),
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            for stem in (
                "figure1_cost_quality",
                "figure2_main_comparison",
                "figure3_continuation",
                "figure4_extension_training",
            ):
                self.assertTrue((figures_dir / f"{stem}.png").exists())
                self.assertTrue((figures_dir / f"{stem}.pdf").exists())
                self.assertTrue((figures_dir / f"{stem}.manifest.json").exists())
            self.assertTrue((figures_dir / "figure_render_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
