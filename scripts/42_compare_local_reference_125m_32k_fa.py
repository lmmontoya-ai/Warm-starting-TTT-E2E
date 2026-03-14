#!/usr/bin/env python3
from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = REPO_ROOT
REFERENCE_ROOT = REPO_ROOT / "ttte2e_reference" / "e2e"


def _utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@dataclass(frozen=True)
class CompareTarget:
    label: str
    local_rel: str
    reference_rel: str
    purpose: str


TARGETS = [
    CompareTarget(
        label="train_runtime",
        local_rel="ttt/jax_runtime/train.py",
        reference_rel="ttt/train.py",
        purpose="Top-level train loop and batch/sharding orchestration",
    ),
    CompareTarget(
        label="loop_runtime",
        local_rel="ttt/jax_runtime/loop.py",
        reference_rel="ttt/model/loop.py",
        purpose="Per-step reduction semantics and evaluator contract",
    ),
    CompareTarget(
        label="attention_runtime",
        local_rel="ttt/jax_runtime/model/attention.py",
        reference_rel="ttt/model/attention.py",
        purpose="FA/SWA attention call shapes, masks, and sharding constraints",
    ),
    CompareTarget(
        label="jax_utils",
        local_rel="ttt/utils/jax_utils.py",
        reference_rel="ttt/utils/jax_utils.py",
        purpose="Gradient checkpointing, remat, and utility wrappers",
    ),
    CompareTarget(
        label="sharding_runtime",
        local_rel="ttt/jax_runtime/sharding.py",
        reference_rel="ttt/model/sharding.py",
        purpose="Model parameter sharding and mesh layout",
    ),
    CompareTarget(
        label="experiment_ext_125m_32k_fa",
        local_rel="configs/experiment/125m/extension/ext-125m-fa-32K.yaml",
        reference_rel="configs/experiment/125m/extension/ext-125m-fa-32K.yaml",
        purpose="User-facing extension experiment config",
    ),
    CompareTarget(
        label="deploy_interactive",
        local_rel="configs/deploy/interactive.yaml",
        reference_rel="configs/deploy/interactive.yaml",
        purpose="Deployment-level runtime defaults",
    ),
]


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def _metric_summary(lines: list[str]) -> dict[str, Any]:
    return {
        "line_count": len(lines),
        "char_count": sum(len(line) for line in lines),
    }


def _count_changed_lines(diff_lines: list[str]) -> tuple[int, int]:
    added = 0
    removed = 0
    for line in diff_lines:
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed


def _contains(lines: list[str], needle: str) -> bool:
    return any(needle in line for line in lines)


def _findings_for_target(target: CompareTarget, local_lines: list[str], ref_lines: list[str]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if target.label == "train_runtime":
        if _contains(local_lines, "put_replicated(state, replicated_sharding)") and _contains(
            ref_lines, "jax.device_put("
        ):
            findings.append(
                {
                    "severity": "high",
                    "title": "Local train path still replicates model state explicitly",
                    "why_it_matters": (
                        "The local runtime keeps the mutable Equinox state on a replicated sharding path, "
                        "while the reference path relies on the mesh-oriented model/data flow. That can retain "
                        "extra device copies or force less favorable compile decisions in long-context extension runs."
                    ),
                }
            )
        if _contains(local_lines, "to_data_parallel_batch(") and _contains(ref_lines, "jax.make_array_from_process_local_data("):
            findings.append(
                {
                    "severity": "high",
                    "title": "Local batch path still uses reshape-based helper instead of reference batch loading",
                    "why_it_matters": (
                        "The reference train loop builds global arrays from process-local data and only then "
                        "rearranges the batch for data parallelism. The local helper hides that flow behind "
                        "`to_data_parallel_batch`, which is a prime suspect for extra transient materialization."
                    ),
                }
            )
        if _contains(local_lines, "record = {") and _contains(local_lines, "inner_loss_proxy"):
            findings.append(
                {
                    "severity": "medium",
                    "title": "Local train loop still materializes richer per-step metric bookkeeping",
                    "why_it_matters": (
                        "This is unlikely to explain a 103 GB compile allocation by itself, but it does mean "
                        "the local training function still carries more metric logic than the reference smoke path."
                    ),
                }
            )
    elif target.label == "loop_runtime":
        if _contains(local_lines, "@eqx.filter_vmap(in_axes=(None, None, None, 0)") and _contains(
            ref_lines, "@eqx.filter_vmap(axis_name=\"data_parallel\", in_axes=(None, 0, None)"
        ):
            findings.append(
                {
                    "severity": "high",
                    "title": "Local train-step signature still differs from the reference axis contract",
                    "why_it_matters": (
                        "The local step vmaps over `(state, model, opt_state, batch)` in a different shape than "
                        "the reference train step. That difference changes what the compiler sees as batched data "
                        "versus replicated state and is a strong candidate for the extension OOM."
                    ),
                }
            )
        if _contains(local_lines, "MetricType.token_nll_loss") or _contains(local_lines, "token_nll_curve"):
            findings.append(
                {
                    "severity": "medium",
                    "title": "Local loop still emphasizes token-level metric aggregation inside the train/eval helpers",
                    "why_it_matters": (
                        "The reference evaluator collects metrics at loader boundaries. The local path still keeps "
                        "token-NLL handling close to the loop helper, which is another place to check for retained arrays."
                    ),
                }
            )
    elif target.label == "attention_runtime":
        if _contains(local_lines, "implementation=\"cudnn\" if jax.default_backend() == \"gpu\" else None") and _contains(
            ref_lines, "implementation=\"cudnn\""
        ):
            findings.append(
                {
                    "severity": "medium",
                    "title": "Local SWA/FA attention still has backend-conditional flash branches",
                    "why_it_matters": (
                        "The reference extension path assumes GPU execution and applies flash-attention constraints "
                        "more directly. The local backend-conditional branches make the graph slightly less faithful "
                        "and are worth ruling out in the compile-memory comparison."
                    ),
                }
            )
    elif target.label == "jax_utils":
        if _contains(local_lines, "maybe_double_remat") and _contains(ref_lines, "scan_remat_chunk"):
            findings.append(
                {
                    "severity": "high",
                    "title": "Remat helper surface is still smaller than the reference utility stack",
                    "why_it_matters": (
                        "The reference utility layer exposes scan/remat chunking patterns that can materially change "
                        "compile-memory behavior in long-context runs. The local helper parity is improved, but still not "
                        "a one-to-one match at the utility boundary."
                    ),
                }
            )
    return findings


def main() -> int:
    artifact_dir = REPO_ROOT / "artifacts" / "oom_diagnosis" / f"compare_125m_32k_fa_{_utc_slug()}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "artifact_dir": str(artifact_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "targets": [],
        "overall_findings": [],
    }

    overall_findings: list[dict[str, Any]] = []

    for target in TARGETS:
        local_path = LOCAL_ROOT / target.local_rel
        reference_path = REFERENCE_ROOT / target.reference_rel
        local_lines = _read_lines(local_path)
        ref_lines = _read_lines(reference_path)
        diff_lines = list(
            difflib.unified_diff(
                local_lines,
                ref_lines,
                fromfile=str(local_path),
                tofile=str(reference_path),
                n=3,
            )
        )
        added, removed = _count_changed_lines(diff_lines)
        findings = _findings_for_target(target, local_lines, ref_lines)
        for finding in findings:
            overall_findings.append(
                {
                    "target": target.label,
                    **finding,
                }
            )

        diff_path = artifact_dir / "diffs" / f"{target.label}.diff"
        _write_text(diff_path, "".join(diff_lines))

        summary["targets"].append(
            {
                "label": target.label,
                "purpose": target.purpose,
                "local_path": str(local_path),
                "reference_path": str(reference_path),
                "local_metrics": _metric_summary(local_lines),
                "reference_metrics": _metric_summary(ref_lines),
                "diff_path": str(diff_path),
                "diff_stats": {
                    "added_lines": added,
                    "removed_lines": removed,
                },
                "findings": findings,
            }
        )

    summary["overall_findings"] = overall_findings
    _write_json(artifact_dir / "summary.json", summary)

    lines = [
        "# 125M 32K FA Local-vs-Reference Comparison",
        "",
        f"Artifact dir: `{artifact_dir}`",
        "",
        "## Highest-priority findings",
    ]
    if overall_findings:
        for finding in overall_findings:
            lines.append(f"- `{finding['severity']}` `{finding['target']}`: {finding['title']}")
            lines.append(f"  - {finding['why_it_matters']}")
    else:
        lines.append("- No heuristic findings were generated.")
    lines.extend(
        [
            "",
            "## Diff inventory",
        ]
    )
    for item in summary["targets"]:
        lines.append(
            f"- `{item['label']}`: +{item['diff_stats']['added_lines']} / -{item['diff_stats']['removed_lines']} "
            f"-> `{Path(item['diff_path']).name}`"
        )
    _write_text(artifact_dir / "report.md", "\n".join(lines) + "\n")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
