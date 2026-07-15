#!/usr/bin/env python3
"""Build preregistered Tier-1 tables, statistics, and result-language drafts."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ttt.research.tier1 import (
    load_tier1_config,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_payload,
    write_json,
)
from ttt.research.tier1_stats import exact_mcnemar_two_sided, paired_bootstrap_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate all completed Tier-1 evaluations."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/research/tier1_eval.yaml")
    )
    parser.add_argument(
        "--tier1-root",
        type=Path,
        default=Path("reports/revision_v2/tier1/revision_v2_t1_eval_v1"),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=Path("paper/plots/figures"))
    parser.add_argument("--include-optional-t1b", action="store_true")
    parser.add_argument("--include-760m-t1c", action="store_true")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty Markdown table")
    fields = list(rows[0])
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(row[field]) for field in fields) + " |" for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_successful_run(run_dir: Path, experiment: str) -> dict[str, Any]:
    manifest = read_json(run_dir / "run_manifest.json")
    if (
        manifest.get("experiment") != experiment
        or manifest.get("status") != "succeeded"
    ):
        raise RuntimeError(f"Missing successful {experiment} run at {run_dir}")
    claimed = str(manifest.get("run_manifest_sha256", ""))
    actual = sha256_payload(
        {key: value for key, value in manifest.items() if key != "run_manifest_sha256"}
    )
    if claimed != actual:
        raise ValueError(f"Run manifest hash mismatch at {run_dir}")
    return manifest


def _require_identical(
    manifests: list[dict[str, Any]], fields: list[str], label: str
) -> None:
    for field in fields:
        observed = {
            json.dumps(manifest.get(field), sort_keys=True) for manifest in manifests
        }
        if len(observed) != 1:
            raise ValueError(
                f"{label} compared conditions differ on {field}: {observed}"
            )


def _preregistered_relation(statistic: dict[str, float]) -> str:
    if float(statistic["ci95_low"]) > 0.0:
        return "higher"
    if float(statistic["ci95_high"]) < 0.0:
        return "lower"
    return "not resolved on this battery"


def _aggregate_t1a(config: dict[str, Any], root: Path, out_dir: Path) -> dict[str, Any]:
    conditions = [
        "FA_SEED_125M",
        "S2_125M_SEED001",
        "S2_MINUS_125M_SEED001",
        "S3_125M",
        "S2_BRIDGE_40PCT_125M",
    ]
    manifests = [
        _load_successful_run(root / "t1a" / condition, "T1a")
        for condition in conditions
    ]
    _require_identical(
        manifests,
        [
            "manifest_sha256",
            "tier1_config_sha256",
            "scorer_sha256",
            "context_length",
            "harness_version",
        ],
        "T1a",
    )
    rows_by_condition: dict[str, list[dict[str, Any]]] = {
        condition: read_jsonl(root / "t1a" / condition / "example_scores.jsonl")
        for condition in conditions
    }
    expected_ids = [str(row["example_id"]) for row in rows_by_condition[conditions[0]]]
    for condition, rows in rows_by_condition.items():
        if [str(row["example_id"]) for row in rows] != expected_ids:
            raise ValueError(f"T1a example order mismatch for {condition}")
    table: list[dict[str, Any]] = []
    for condition, manifest in zip(conditions, manifests, strict=True):
        table.append(
            {
                "condition": condition,
                **{
                    task: f"{100.0 * float(manifest['metrics'][task]['acc_norm']):.2f}"
                    for task in map(str, config["t1a"]["tasks"])
                },
                "average": f"{100.0 * float(manifest['metrics']['macro']['acc_norm']):.2f}",
            }
        )
    _write_csv(out_dir / "table_t1a_short_context.csv", table)
    _write_markdown_table(out_dir / "table_t1a_short_context.md", table)
    _write_csv(
        out_dir / "table_t1a_sample_counts.csv",
        [
            {
                "condition": condition,
                **{
                    task: int(manifest["metrics"][task]["n"])
                    for task in map(str, config["t1a"]["tasks"])
                },
            }
            for condition, manifest in zip(conditions, manifests, strict=True)
        ],
    )

    paired: dict[str, Any] = {}
    s2_rows = rows_by_condition["S2_125M_SEED001"]
    for comparator in ("FA_SEED_125M", "S3_125M", "S2_MINUS_125M_SEED001"):
        other = rows_by_condition[comparator]
        key = f"S2_125M_SEED001_minus_{comparator}"
        paired[key] = {
            "paired_bootstrap_acc_norm": paired_bootstrap_delta(
                [row["acc_norm"] for row in s2_rows],
                [row["acc_norm"] for row in other],
                resamples=int(config["t1a"]["bootstrap_resamples"]),
                seed=int(config["t1a"]["bootstrap_seed"]),
            ),
            "mcnemar_acc_norm": exact_mcnemar_two_sided(
                [bool(row["acc_norm"]) for row in s2_rows],
                [bool(row["acc_norm"]) for row in other],
            ),
        }
    write_json(out_dir / "t1a_paired_statistics.json", paired)
    s2_fa = paired["S2_125M_SEED001_minus_FA_SEED_125M"]["paired_bootstrap_acc_norm"]
    s2_s3 = paired["S2_125M_SEED001_minus_S3_125M"]["paired_bootstrap_acc_norm"]
    narrative = (
        "T1a reporting-rule draft (descriptive, not an equivalence claim): bridged S2 was "
        f"{_preregistered_relation(s2_fa)} than the FA seed and "
        f"{_preregistered_relation(s2_s3)} than scratch S3 under the preregistered paired-CI rule. "
        "The S2 "
        f"macro normalized-accuracy difference versus the FA seed was {100*s2_fa['mean_delta']:+.2f} "
        f"percentage points (paired bootstrap 95% CI [{100*s2_fa['ci95_low']:+.2f}, "
        f"{100*s2_fa['ci95_high']:+.2f}]); versus scratch S3 it was "
        f"{100*s2_s3['mean_delta']:+.2f} points (95% CI [{100*s2_s3['ci95_low']:+.2f}, "
        f"{100*s2_s3['ci95_high']:+.2f}]). All four preregistered tasks are reported in Table T1a."
    )
    (out_dir / "t1a_result_language.md").write_text(narrative + "\n", encoding="utf-8")
    return {
        "conditions": conditions,
        "manifest_sha256": manifests[0]["manifest_sha256"],
    }


def _aggregate_t1b(
    config: dict[str, Any], root: Path, out_dir: Path, *, include_optional: bool
) -> dict[str, Any]:
    conditions = [str(x) for x in config["t1b"]["conditions"]]
    if include_optional:
        conditions.extend(str(x) for x in config["t1b"].get("optional_conditions", []))
    manifests = [
        _load_successful_run(root / "t1b" / condition, "T1b")
        for condition in conditions
    ]
    _require_identical(
        manifests,
        [
            "manifest_sha256",
            "tier1_config_sha256",
            "scorer_sha256",
            "instrumentation_version",
        ],
        "T1b",
    )
    trajectory_rows: list[dict[str, Any]] = []
    for condition, manifest in zip(conditions, manifests, strict=True):
        for checkpoint in manifest["checkpoint_summaries"]:
            trajectory_rows.append(
                {
                    "condition": condition,
                    "checkpoint_step": int(checkpoint["checkpoint_step"]),
                    "mean_raw_grad_l2": float(checkpoint["mean_raw_grad_l2"]),
                    "mean_applied_update_l2": float(
                        checkpoint["mean_applied_update_l2"]
                    ),
                    "mean_relative_update_l2": float(
                        checkpoint["mean_relative_update_l2"]
                    ),
                    "clip_crossing_rate": float(checkpoint["clip_crossing_rate"]),
                    "trajectory_scope": manifest["checkpoint_trajectory_scope"],
                }
            )
    _write_csv(out_dir / "table_t1b_checkpoint_trajectory.csv", trajectory_rows)
    primary = manifests[:2]
    values: dict[str, float] = {}
    layer_rows: list[dict[str, Any]] = []
    for condition, manifest in zip(conditions[:2], primary, strict=True):
        final_step = int(manifest["checkpoint_step"])
        trace = np.load(
            root / "t1b" / condition / f"fast_weight_trace_step_{final_step:08d}.npz"
        )
        values[condition] = float(
            np.asarray(trace["applied_update_l2"], dtype=np.float64).mean()
        )
        raw_grad = np.asarray(trace["raw_grad_l2"], dtype=np.float64)
        applied = np.asarray(trace["applied_update_l2"], dtype=np.float64)
        relative = np.asarray(trace["relative_update_l2"], dtype=np.float64)
        finite = np.asarray(trace["all_finite"], dtype=np.bool_)
        clip_rate = float(np.asarray(trace["clip_crossed"], dtype=np.float64).mean())
        for layer_index in range(raw_grad.shape[2]):
            layer_rows.append(
                {
                    "condition": condition,
                    "checkpoint_step": final_step,
                    "suffix_layer_index": layer_index,
                    "mean_raw_grad_l2": float(raw_grad[:, :, layer_index].mean()),
                    "mean_applied_update_l2": float(applied[:, :, layer_index].mean()),
                    "mean_relative_update_l2": float(
                        relative[:, :, layer_index].mean()
                    ),
                    "clip_crossing_rate": clip_rate,
                    "non_finite_count": int(
                        np.size(finite[:, :, layer_index])
                        - np.count_nonzero(finite[:, :, layer_index])
                    ),
                }
            )
    _write_csv(out_dir / "table_t1b_final_by_layer.csv", layer_rows)
    delta = values[conditions[0]] - values[conditions[1]]
    scope = ", ".join(
        f"{condition}={manifest['checkpoint_trajectory_scope']}"
        for condition, manifest in zip(conditions[:2], primary, strict=True)
    )
    (out_dir / "t1b_result_language.md").write_text(
        "T1b exploratory reporting-rule draft: mean final-checkpoint prime-MLP applied-update "
        f"norm was {values[conditions[0]]:.6g} for bridged S2 and {values[conditions[1]]:.6g} "
        f"for paired S2-minus (descriptive difference {delta:+.6g}). Checkpoint scope: {scope}. "
        "These measurements annotate the behavioral bridge result; they do not establish a full mechanism.\n",
        encoding="utf-8",
    )
    return {
        "conditions": conditions,
        "manifest_sha256": manifests[0]["manifest_sha256"],
    }


def _aggregate_t1c(
    config: dict[str, Any], root: Path, out_dir: Path, *, include_760m: bool
) -> dict[str, Any]:
    conditions = [str(x) for x in config["t1c"]["conditions_125m"]]
    if include_760m:
        conditions.extend(str(x) for x in config["t1c"]["conditions_760m"])
    manifests = [
        _load_successful_run(root / "t1c" / condition, "T1c")
        for condition in conditions
    ]
    _require_identical(
        manifests,
        [
            "manifest_sha256",
            "validation_report_sha256",
            "tier1_config_sha256",
            "scorer_version",
            "scorer_sha256",
            "context_length",
            "harness_version",
        ],
        "T1c",
    )
    table: list[dict[str, Any]] = []
    for condition, manifest in zip(conditions, manifests, strict=True):
        table.append(
            {
                "condition": condition,
                **{
                    task: f"{float(manifest['metrics'][task]['official_string_match_all_percent']):.2f}"
                    for task in map(str, config["t1c"]["tasks"])
                },
                "average": f"{float(manifest['metrics']['macro']['official_string_match_all_percent']):.2f}",
                "evidence_fence": manifest["condition"].get("evidence_fence", ""),
            }
        )
    _write_csv(out_dir / "table_t1c_external_ruler.csv", table)
    _write_markdown_table(out_dir / "table_t1c_external_ruler.md", table)
    count_rows: list[dict[str, Any]] = []
    rows_by_condition: dict[str, list[dict[str, Any]]] = {}
    for condition in conditions:
        rows = read_jsonl(root / "t1c" / condition / "example_scores.jsonl")
        rows_by_condition[condition] = rows
        for task in map(str, config["t1c"]["tasks"]):
            selected = [row for row in rows if row["task"] == task]
            count_rows.append(
                {
                    "condition": condition,
                    "task": task,
                    "n": len(selected),
                    "sum_fractional_score": float(
                        sum(row["score_fraction"] for row in selected)
                    ),
                    "exact_all_references_correct": int(
                        sum(row["exact_all_references"] for row in selected)
                    ),
                }
            )
    _write_csv(out_dir / "table_t1c_external_ruler_counts.csv", count_rows)
    paired: dict[str, Any] = {}
    pairs = [("S2_125M_SEED001", "S3_125M")]
    if include_760m:
        pairs.append(("S2_760M", "S3_760M"))
    for left, right in pairs:
        left_rows = rows_by_condition[left]
        right_rows = rows_by_condition[right]
        if [row["example_id"] for row in left_rows] != [
            row["example_id"] for row in right_rows
        ]:
            raise ValueError(f"T1c paired example order mismatch for {left}/{right}")
        paired[f"{left}_minus_{right}"] = {
            "paired_bootstrap_fractional_score": paired_bootstrap_delta(
                [row["score_fraction"] for row in left_rows],
                [row["score_fraction"] for row in right_rows],
                resamples=10_000,
                seed=0,
            ),
            "mcnemar_exact_all_references": exact_mcnemar_two_sided(
                [bool(row["exact_all_references"]) for row in left_rows],
                [bool(row["exact_all_references"]) for row in right_rows],
            ),
        }
    write_json(out_dir / "t1c_paired_statistics.json", paired)
    by_condition = {
        condition: float(
            manifest["metrics"]["macro"]["official_string_match_all_percent"]
        )
        for condition, manifest in zip(conditions, manifests, strict=True)
    }
    delta_125 = by_condition["S2_125M_SEED001"] - by_condition["S3_125M"]
    paired_125 = paired["S2_125M_SEED001_minus_S3_125M"][
        "paired_bootstrap_fractional_score"
    ]
    if float(paired_125["ci95_low"]) > 0.0:
        agreement_125 = "disagrees"
    elif float(paired_125["mean_delta"]) <= 0.0:
        agreement_125 = "agrees directionally"
    else:
        agreement_125 = "does not resolve a disagreement"
    sentence = (
        "T1c descriptive reporting-rule draft: on the external RULER subset, S2 minus S3 "
        f"was {delta_125:+.2f} percentage points at 125M; this {agreement_125} "
        "with E2a's closed finding of no warm-start retrieval advantage. This is corroborative, "
        "not a new confirmatory retrieval test."
    )
    if include_760m:
        delta_760 = by_condition["S2_760M"] - by_condition["S3_760M"]
        paired_760 = paired["S2_760M_minus_S3_760M"][
            "paired_bootstrap_fractional_score"
        ]
        if float(paired_760["ci95_low"]) > 0.0:
            agreement_760 = "disagrees"
        elif float(paired_760["mean_delta"]) <= 0.0:
            agreement_760 = "agrees directionally"
        else:
            agreement_760 = "does not resolve a disagreement"
        sentence += (
            f" At 760M the descriptive difference was {delta_760:+.2f} points and {agreement_760} "
            "with the E2a finding; these inherited-model rows remain exploratory quality evidence."
        )
    (out_dir / "t1c_result_language.md").write_text(sentence + "\n", encoding="utf-8")
    return {
        "conditions": conditions,
        "manifest_sha256": manifests[0]["manifest_sha256"],
    }


def main() -> int:
    args = parse_args()
    config = load_tier1_config(args.config)
    root = args.tier1_root.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else root / "aggregate"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {
        "T1a": _aggregate_t1a(config, root, out_dir),
        "T1b": _aggregate_t1b(
            config, root, out_dir, include_optional=bool(args.include_optional_t1b)
        ),
        "T1c": _aggregate_t1c(
            config, root, out_dir, include_760m=bool(args.include_760m_t1c)
        ),
    }
    plot_manifest = out_dir / "tier1_plot_manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("75_make_paper_plots.py")),
            "--tier1-only",
            "--tier1-fast-weight-root",
            str(root),
            "--figures-dir",
            str(args.figures_dir),
            "--manifest-path",
            str(plot_manifest),
        ],
        check=True,
    )
    artifact_files = sorted(path for path in out_dir.iterdir() if path.is_file())
    aggregate = {
        "schema_version": "1.0",
        "paper_run_id": config["paper_run_id"],
        "tier1_config_sha256": config["_config_sha256"],
        "summaries": summaries,
        "artifacts": {path.name: sha256_file(path) for path in artifact_files},
        "figure_manifest": str(plot_manifest),
        "status": "succeeded",
    }
    aggregate["aggregate_manifest_sha256"] = sha256_payload(aggregate)
    write_json(out_dir / "aggregate_manifest.json", aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
