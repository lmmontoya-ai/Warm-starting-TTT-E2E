#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ttt.research.continuation_ablations import (
    ISO_QUALITY_TARGET_LOSS,
    S2_ISOQ_SOURCE,
    canonical_branch_costs,
    canonical_eval_loss,
)
from ttt.research.types import utc_now_iso


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _sorted_frontier(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("frontier_points", [])
    if not isinstance(rows, list):
        return []
    cleaned = [row for row in rows if isinstance(row, dict)]
    return sorted(
        cleaned,
        key=lambda row: (
            int(row.get("extra_steps", 0) or 0),
            int(row.get("checkpoint_step", -1) or -1),
        ),
    )


def _numeric_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in points if row.get("loss_ce_mean") is not None]


def summarize_iso_quality_payload(payload: dict[str, Any]) -> dict[str, Any]:
    points = _sorted_frontier(payload)
    numeric = _numeric_points(points)
    terminal = payload.get("terminal", {})
    if not isinstance(terminal, dict):
        terminal = {}
    status = str(payload.get("status", "unknown"))
    first_match = next((row for row in points if bool(row.get("matched_target"))), None)
    best_point = min(
        numeric,
        key=lambda row: float(row["loss_ce_mean"]),
        default=None,
    )

    summary: dict[str, Any] = {
        "mode": "iso_quality",
        "status": status,
        "source_stage_id": str(payload.get("source", {}).get("stage_id", "")),
        "source_run_id": str(payload.get("source", {}).get("run_id", "")),
        "source_checkpoint_step": int(payload.get("source", {}).get("checkpoint_step", 0) or 0),
        "target_loss_ce_mean": float(payload.get("source", {}).get("target_loss_ce_mean", ISO_QUALITY_TARGET_LOSS) or ISO_QUALITY_TARGET_LOSS),
        "frontier_points": len(points),
        "numeric_frontier_points": len(numeric),
        "terminal_reason": str(terminal.get("reason", "")),
        "did_not_reach_target": first_match is None,
    }
    if first_match is not None:
        summary.update(
            {
                "matched_target": True,
                "first_matching_stage_id": str(first_match.get("stage_id", "")),
                "first_matching_run_id": str(first_match.get("run_id", "")),
                "first_matching_checkpoint_step": int(first_match.get("checkpoint_step", 0) or 0),
                "steps_to_target": int(first_match.get("extra_steps", 0) or 0),
                "tokens_to_target": int(first_match.get("extra_tokens", 0) or 0),
                "gpu_hours_to_target": float(first_match.get("extra_gpu_hours", 0.0) or 0.0),
                "loss_ce_mean_at_target": float(first_match.get("loss_ce_mean", 0.0) or 0.0),
                "loss_delta_vs_target": float(first_match.get("loss_ce_mean", 0.0) or 0.0) - float(summary["target_loss_ce_mean"]),
                "total_branch_tokens_at_target": float(first_match.get("total_branch_tokens", 0.0) or 0.0),
                "total_branch_gpu_hours_at_target": float(first_match.get("total_branch_gpu_hours", 0.0) or 0.0),
            }
        )
    else:
        summary["matched_target"] = False

    if best_point is not None:
        summary.update(
            {
                "best_loss_ce_mean": float(best_point.get("loss_ce_mean", 0.0) or 0.0),
                "best_stage_id": str(best_point.get("stage_id", "")),
                "best_run_id": str(best_point.get("run_id", "")),
                "best_checkpoint_step": int(best_point.get("checkpoint_step", 0) or 0),
                "best_extra_steps": int(best_point.get("extra_steps", 0) or 0),
                "best_extra_tokens": int(best_point.get("extra_tokens", 0) or 0),
                "best_extra_gpu_hours": float(best_point.get("extra_gpu_hours", 0.0) or 0.0),
            }
        )
    return summary


def summarize_iso_total_tokens_payload(
    payload: dict[str, Any],
    *,
    exp_dir: Path,
) -> dict[str, Any]:
    points = _sorted_frontier(payload)
    numeric = _numeric_points(points)
    terminal = payload.get("terminal", {})
    if not isinstance(terminal, dict):
        terminal = {}
    status = str(payload.get("status", "unknown"))
    expected_extra_steps = int(payload.get("source", {}).get("extra_steps_budget", 0) or 0)
    final_point = next(
        (
            row
            for row in reversed(numeric)
            if int(row.get("extra_steps", 0) or 0) == expected_extra_steps
        ),
        None,
    )
    branch_costs = canonical_branch_costs(exp_dir)
    canonical_s2_loss = canonical_eval_loss(
        exp_dir,
        stage_id=S2_ISOQ_SOURCE[0],
        run_id=S2_ISOQ_SOURCE[1],
    )
    summary: dict[str, Any] = {
        "mode": "iso_total_tokens",
        "status": status,
        "source_stage_id": str(payload.get("source", {}).get("stage_id", "")),
        "source_run_id": str(payload.get("source", {}).get("run_id", "")),
        "source_checkpoint_step": int(payload.get("source", {}).get("checkpoint_step", 0) or 0),
        "extra_steps_budget": int(payload.get("source", {}).get("extra_steps_budget", 0) or 0),
        "extra_tokens_budget": int(payload.get("source", {}).get("extra_tokens_budget", 0) or 0),
        "frontier_points": len(points),
        "numeric_frontier_points": len(numeric),
        "terminal_reason": str(terminal.get("reason", "")),
        "canonical_s2_loss_ce_mean": canonical_s2_loss,
        "canonical_s2_branch_total_tokens": float(branch_costs["s2_branch"]["base_total_tokens"]),
        "canonical_s2_branch_total_gpu_hours": float(branch_costs["s2_branch"]["base_total_gpu_hours"]),
        "canonical_s2_branch_marginal_gpu_hours": float(branch_costs["s2_branch"]["base_marginal_gpu_hours"]),
    }
    if status == "dry_run" or final_point is None:
        summary["has_final_endpoint"] = False
        return summary

    quality_delta = float(final_point.get("loss_ce_mean", 0.0) or 0.0) - canonical_s2_loss
    marginal_delta = float(final_point.get("total_branch_marginal_gpu_hours", 0.0) or 0.0) - float(branch_costs["s2_branch"]["base_marginal_gpu_hours"])
    total_delta = float(final_point.get("total_branch_gpu_hours", 0.0) or 0.0) - float(branch_costs["s2_branch"]["base_total_gpu_hours"])
    token_delta = float(final_point.get("total_branch_tokens", 0.0) or 0.0) - float(branch_costs["s2_branch"]["base_total_tokens"])
    if quality_delta < 0.0:
        interpretation = "Scratch TTT-E2E still wins at equal total tokens."
    elif quality_delta > 0.0:
        interpretation = "Warm-start still wins at equal total tokens."
    else:
        interpretation = "Scratch TTT-E2E and warm-start are tied at equal total tokens."

    summary.update(
        {
            "has_final_endpoint": True,
            "final_stage_id": str(final_point.get("stage_id", "")),
            "final_run_id": str(final_point.get("run_id", "")),
            "final_checkpoint_step": int(final_point.get("checkpoint_step", 0) or 0),
            "final_loss_ce_mean": float(final_point.get("loss_ce_mean", 0.0) or 0.0),
            "quality_delta_vs_canonical_s2": quality_delta,
            "marginal_gpu_hours_delta_vs_canonical_s2_path": marginal_delta,
            "total_gpu_hours_delta_vs_canonical_s2_branch": total_delta,
            "total_tokens_delta_vs_canonical_s2_branch": token_delta,
            "final_total_branch_tokens": float(final_point.get("total_branch_tokens", 0.0) or 0.0),
            "final_total_branch_gpu_hours": float(final_point.get("total_branch_gpu_hours", 0.0) or 0.0),
            "final_total_branch_marginal_gpu_hours": float(final_point.get("total_branch_marginal_gpu_hours", 0.0) or 0.0),
            "interpretation": interpretation,
        }
    )
    return summary


def _md_number(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _write_iso_quality_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(summary.get("status", "")) == "dry_run":
        body = "# Iso-Quality Summary\n\nThis file was generated from a dry-run ledger. No continuation quality result is recorded yet.\n"
    elif bool(summary.get("matched_target")):
        body = (
            f"# Iso-Quality Summary\n\n"
            f"`S2_125M` continuation reached the canonical `S3_125M` Books32K target loss "
            f"({summary['target_loss_ce_mean']:.12f}) at checkpoint step "
            f"`{summary['first_matching_checkpoint_step']}` after `+{summary['steps_to_target']}` steps, "
            f"`{summary['tokens_to_target']}` extra tokens, and `{summary['gpu_hours_to_target']:.6f}` extra GPU-hours. "
            f"The first matching point achieved `loss_ce_mean = {summary['loss_ce_mean_at_target']:.12f}`.\n"
        )
    else:
        best_loss = summary.get("best_loss_ce_mean")
        body = (
            f"# Iso-Quality Summary\n\n"
            f"`S2_125M` continuation did not reach the canonical `S3_125M` Books32K target loss "
            f"({summary['target_loss_ce_mean']:.12f}). The best observed checkpoint was "
            f"`{summary.get('best_checkpoint_step', 'n/a')}` with `loss_ce_mean = {_md_number(best_loss, 12)}` "
            f"after `+{summary.get('best_extra_steps', 'n/a')}` steps. "
            f"The run stopped because `{summary.get('terminal_reason', 'unknown')}`.\n"
        )
    path.write_text(body, encoding="utf-8")


def _write_iso_total_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(summary.get("status", "")) == "dry_run":
        body = "# Iso-Total-Tokens Summary\n\nThis file was generated from a dry-run ledger. No equal-total-tokens continuation result is recorded yet.\n"
    elif not bool(summary.get("has_final_endpoint")):
        body = "# Iso-Total-Tokens Summary\n\nNo final continuation endpoint was recorded.\n"
    else:
        body = (
            f"# Iso-Total-Tokens Summary\n\n"
            f"The equal-total-tokens `S3_125M` continuation finished at checkpoint step "
            f"`{summary['final_checkpoint_step']}` with `loss_ce_mean = {summary['final_loss_ce_mean']:.12f}`. "
            f"Relative to canonical `S2_125M`, the quality delta is "
            f"`{summary['quality_delta_vs_canonical_s2']:.12f}` and the marginal GPU-hour delta versus "
            f"`S2_ADAPT_125M + S2_125M` is `{summary['marginal_gpu_hours_delta_vs_canonical_s2_path']:.6f}`. "
            f"{summary['interpretation']}\n"
        )
    path.write_text(body, encoding="utf-8")


def _write_combined_markdown(
    path: Path,
    *,
    paper_run_id: str,
    iso_quality: dict[str, Any],
    iso_total: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 125M Continuation Ablations",
        "",
        f"- paper_run_id: `{paper_run_id}`",
        f"- generated_at_utc: `{utc_now_iso()}`",
        "",
        "## Iso-Quality",
    ]
    if str(iso_quality.get("status", "")) == "dry_run":
        lines.append("Dry-run only. No iso-quality continuation result is recorded yet.")
    elif bool(iso_quality.get("matched_target")):
        lines.append(
            f"`S2_125M` matched the canonical `S3_125M` Books32K loss target after "
            f"`+{iso_quality['steps_to_target']}` steps and `{iso_quality['gpu_hours_to_target']:.6f}` GPU-hours."
        )
    else:
        lines.append(
            f"`S2_125M` did not match the canonical `S3_125M` Books32K target. "
            f"Best loss was `{_md_number(iso_quality.get('best_loss_ce_mean'), 12)}` and stop reason was "
            f"`{iso_quality.get('terminal_reason', 'unknown')}`."
        )
    lines.extend(["", "## Iso-Total-Tokens"])
    if str(iso_total.get("status", "")) == "dry_run":
        lines.append("Dry-run only. No iso-total-tokens continuation result is recorded yet.")
    elif bool(iso_total.get("has_final_endpoint")):
        lines.append(
            f"At equal total tokens, the continued `S3_125M` endpoint reached "
            f"`loss_ce_mean = {iso_total['final_loss_ce_mean']:.12f}` with quality delta "
            f"`{iso_total['quality_delta_vs_canonical_s2']:.12f}` versus canonical `S2_125M`. "
            f"{iso_total.get('interpretation', '')}"
        )
    else:
        lines.append("No final iso-total-tokens endpoint was recorded.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize 125M continuation ablation outputs.")
    parser.add_argument("--paper-run-id", required=True)
    parser.add_argument("--canonical-paper-run-id", required=True)
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--reports-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exp_dir = args.exp_dir.expanduser().resolve()
    reports_root = args.reports_root.expanduser().resolve()
    ledger_path = reports_root / "runner" / "ablation_ledger.json"
    ledger = _load_json(ledger_path)

    modes = ledger.get("modes", {})
    if not isinstance(modes, dict):
        raise ValueError(f"Invalid modes payload in {ledger_path}")
    iso_quality_payload = modes.get("iso_quality", {})
    iso_total_payload = modes.get("iso_total_tokens", {})
    if not isinstance(iso_quality_payload, dict):
        iso_quality_payload = {}
    if not isinstance(iso_total_payload, dict):
        iso_total_payload = {}

    frontier_rows: list[dict[str, Any]] = []
    for mode_name in ("iso_quality", "iso_total_tokens"):
        payload = modes.get(mode_name, {})
        if not isinstance(payload, dict):
            continue
        frontier_rows.extend(_sorted_frontier(payload))
    frontier_rows = sorted(
        frontier_rows,
        key=lambda row: (str(row.get("mode", "")), int(row.get("extra_steps", 0) or 0), int(row.get("checkpoint_step", -1) or -1)),
    )

    iso_quality_summary = summarize_iso_quality_payload(iso_quality_payload)
    iso_total_summary = summarize_iso_total_tokens_payload(iso_total_payload, exp_dir=exp_dir)

    _write_csv(reports_root / "frontier.csv", frontier_rows)
    _write_json(reports_root / "iso_quality_summary.json", iso_quality_summary)
    _write_csv(reports_root / "iso_quality_summary.csv", [iso_quality_summary])
    _write_iso_quality_markdown(reports_root / "iso_quality_summary.md", iso_quality_summary)
    _write_json(reports_root / "iso_total_tokens_summary.json", iso_total_summary)
    _write_csv(reports_root / "iso_total_tokens_summary.csv", [iso_total_summary])
    _write_iso_total_markdown(reports_root / "iso_total_tokens_summary.md", iso_total_summary)
    _write_combined_markdown(
        reports_root / "125m_continuation_ablations.md",
        paper_run_id=args.paper_run_id,
        iso_quality=iso_quality_summary,
        iso_total=iso_total_summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
