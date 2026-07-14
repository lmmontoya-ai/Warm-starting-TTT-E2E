from __future__ import annotations

import math
import subprocess
import tempfile
import unittest
from pathlib import Path

from ttt.research.e2a_proxy import (
    PRIMARY_COMPARISON_ID,
    apply_secondary_holm,
    compare_conditions,
    expansion_verdict,
    generate_example_manifest,
    load_manifest,
    validate_condition_rows,
    write_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class E2aHarnessTest(unittest.TestCase):
    def test_manifest_generation_is_deterministic_and_hash_checked(self) -> None:
        first = generate_example_manifest(
            num_examples=6,
            seed=123,
            context_length=32,
            vocab_size=128,
            candidates=4,
            positions=(0.1, 0.5, 0.9),
            scale="125M",
        )
        second = generate_example_manifest(
            num_examples=6,
            seed=123,
            context_length=32,
            vocab_size=128,
            candidates=4,
            positions=(0.1, 0.5, 0.9),
            scale="125M",
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first["examples"]), 6)
        self.assertEqual(first["examples"][0]["example_id"], "125m_ctx32_seed123_ex000000")
        self.assertIn("manifest_hash", first)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "manifest.json"
            write_manifest(path, first)
            self.assertEqual(load_manifest(path), first)
            tampered = path.read_text(encoding="utf-8").replace('"needle":', '"needle":')
            path.write_text(tampered.replace('"num_examples": 6', '"num_examples": 7'), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Manifest hash mismatch"):
                load_manifest(path)

    def test_condition_schema_and_pairing_refuse_mismatched_manifests(self) -> None:
        manifest_a = generate_example_manifest(
            num_examples=4,
            seed=1,
            context_length=16,
            vocab_size=64,
            candidates=4,
        )
        manifest_b = generate_example_manifest(
            num_examples=4,
            seed=2,
            context_length=16,
            vocab_size=64,
            candidates=4,
        )
        rows_a = [
            _row(example["example_id"], True, "S2_125M", "run-a", manifest_a["manifest_hash"])
            for example in manifest_a["examples"]
        ]
        rows_b = [
            _row(example["example_id"], False, "S3_125M", "run-b", manifest_b["manifest_hash"])
            for example in manifest_b["examples"]
        ]
        out_a = validate_condition_rows(rows_a, expected_manifest_hash=manifest_a["manifest_hash"])
        out_b = validate_condition_rows(rows_b, expected_manifest_hash=manifest_b["manifest_hash"])
        with self.assertRaisesRegex(ValueError, "different example manifests"):
            compare_conditions(
                comparison_id=PRIMARY_COMPARISON_ID,
                model_a=out_a,
                model_b=out_b,
                requested_examples=500,
            )

    def test_primary_comparison_stats_and_expansion_ladder(self) -> None:
        manifest = generate_example_manifest(
            num_examples=5,
            seed=3,
            context_length=32768,
            vocab_size=128,
            candidates=4,
        )
        ids = [example["example_id"] for example in manifest["examples"]]
        rows_s2 = [
            _row(ids[0], True, "S2_125M", "s2", manifest["manifest_hash"]),
            _row(ids[1], True, "S2_125M", "s2", manifest["manifest_hash"]),
            _row(ids[2], False, "S2_125M", "s2", manifest["manifest_hash"]),
            _row(ids[3], False, "S2_125M", "s2", manifest["manifest_hash"]),
            _row(ids[4], True, "S2_125M", "s2", manifest["manifest_hash"]),
        ]
        rows_s3 = [
            _row(ids[0], False, "S3_125M", "s3", manifest["manifest_hash"]),
            _row(ids[1], True, "S3_125M", "s3", manifest["manifest_hash"]),
            _row(ids[2], True, "S3_125M", "s3", manifest["manifest_hash"]),
            _row(ids[3], False, "S3_125M", "s3", manifest["manifest_hash"]),
            _row(ids[4], False, "S3_125M", "s3", manifest["manifest_hash"]),
        ]
        out_s2 = validate_condition_rows(rows_s2, expected_manifest_hash=manifest["manifest_hash"])
        out_s3 = validate_condition_rows(rows_s3, expected_manifest_hash=manifest["manifest_hash"])
        result = compare_conditions(
            comparison_id=PRIMARY_COMPARISON_ID,
            model_a=out_s2,
            model_b=out_s3,
            requested_examples=500,
            bootstrap_resamples=100,
        )
        self.assertEqual(result["hierarchy"], "PRIMARY")
        self.assertEqual(result["correction_family"], "confirmatory_uncorrected")
        self.assertEqual(result["a_only"], 2)
        self.assertEqual(result["b_only"], 1)
        self.assertEqual(result["discordant_pairs"], 3)
        self.assertEqual(result["expansion_verdict"], "expand_1000")
        self.assertTrue(math.isclose(result["mcnemar_p"], 0.5, rel_tol=0, abs_tol=1e-12))

    def test_expansion_ladder_boundaries(self) -> None:
        self.assertEqual(expansion_verdict(effective_n=49, requested_examples=500), "expand_1000")
        self.assertEqual(expansion_verdict(effective_n=50, requested_examples=500), "adequate")
        self.assertEqual(expansion_verdict(effective_n=49, requested_examples=1000), "expand_2000")
        self.assertEqual(expansion_verdict(effective_n=49, requested_examples=2000), "underpowered")

    def test_secondary_holm_and_primary_labeling(self) -> None:
        rows = [
            {"comparison_id": PRIMARY_COMPARISON_ID, "hierarchy": "PRIMARY", "mcnemar_p": 0.04},
            {"comparison_id": "s2_vs_s2minus", "hierarchy": "SECONDARY", "mcnemar_p": 0.02},
            {"comparison_id": "s2minus_vs_s1", "hierarchy": "SECONDARY", "mcnemar_p": 0.03},
        ]
        adjusted = apply_secondary_holm(rows)
        by_id = {row["comparison_id"]: row for row in adjusted}
        self.assertEqual(by_id[PRIMARY_COMPARISON_ID]["p_adjustment"], "none_primary_uncorrected")
        self.assertEqual(by_id[PRIMARY_COMPARISON_ID]["mcnemar_p_adjusted"], 0.04)
        self.assertEqual(by_id["s2_vs_s2minus"]["p_adjustment"], "holm_secondary_family")
        self.assertTrue(math.isclose(by_id["s2_vs_s2minus"]["mcnemar_p_adjusted"], 0.04))
        self.assertTrue(math.isclose(by_id["s2minus_vs_s1"]["mcnemar_p_adjusted"], 0.04))

    def test_cli_end_to_end_mock_harness(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "manifest.json"
            s2 = root / "s2.csv"
            s3 = root / "s3.csv"
            out_json = root / "analysis.json"
            out_csv = root / "analysis.csv"
            out_md = root / "analysis.md"

            self._run_cli(
                "generate-manifest",
                "--out",
                str(manifest),
                "--num-examples",
                "12",
                "--context-length",
                "32768",
                "--vocab-size",
                "256",
                "--candidates",
                "4",
            )
            self._run_cli(
                "mock-score",
                "--manifest",
                str(manifest),
                "--condition-id",
                "S2_125M",
                "--stage-id",
                "S2_125M",
                "--run-id",
                "s2",
                "--checkpoint-id",
                "s2@479",
                "--mode",
                "strong",
                "--out-csv",
                str(s2),
            )
            self._run_cli(
                "mock-score",
                "--manifest",
                str(manifest),
                "--condition-id",
                "S3_125M",
                "--stage-id",
                "S3_125M",
                "--run-id",
                "s3",
                "--checkpoint-id",
                "s3@479",
                "--mode",
                "weak",
                "--out-csv",
                str(s3),
            )
            self._run_cli(
                "aggregate",
                "--manifest",
                str(manifest),
                "--condition-csv",
                str(s2),
                "--condition-csv",
                str(s3),
                "--comparison",
                f"{PRIMARY_COMPARISON_ID}:S2_125M:S3_125M:greater",
                "--bootstrap-resamples",
                "100",
                "--out-json",
                str(out_json),
                "--out-csv",
                str(out_csv),
                "--out-md",
                str(out_md),
            )
            self.assertTrue(out_json.exists())
            self.assertTrue(out_csv.exists())
            self.assertTrue(out_md.exists())
            self.assertIn("PRIMARY", out_md.read_text(encoding="utf-8"))

    def _run_cli(self, *args: str) -> None:
        subprocess.run(
            ["python", "scripts/81_e2a_proxy.py", *args],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )


def _row(example_id: str, correct: bool, stage_id: str, run_id: str, manifest_hash: str) -> dict[str, object]:
    return {
        "example_id": example_id,
        "correct": correct,
        "condition_id": stage_id,
        "stage_id": stage_id,
        "run_id": run_id,
        "checkpoint_id": f"{run_id}@479",
        "context_length": 32768,
        "manifest_hash": manifest_hash,
        "seed": 20260610,
    }


if __name__ == "__main__":
    unittest.main()
