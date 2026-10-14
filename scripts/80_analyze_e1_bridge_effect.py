#!/usr/bin/env python3
"""Compute the preregistered E1 bridge effect from paired S2 / S2-minus eval losses.

The statistic (PREREGISTRATION_REVISION_V2.md, kept by Amendment 1 of
PREREGISTRATION_REVISION_V3.md) is, per seed s,

    delta_s = loss(S2_MINUS_125M, seed s) - loss(S2_125M, seed s)

on the Books3 32K validation loss. A positive delta means the bridge helps.
The script reports the mean delta, a paired seed-level bootstrap 95% CI and
the margin decision.

Input is an eval summary written by scripts/34_eval_matrix_jax.py (for
example the Books32K summary from scripts/96_run_revision_v3_ladder.py
--eval-after). Seeds are read from run ids ending in -seedNNN. Pass
--variant when the summary holds several bridge variants (for example a
bridge peak-LR sweep), so that each seed has exactly one S2 run.

Outputs, under --out-dir: e1_bridge_effect_analysis.json, a paired-loss CSV
and a short Markdown summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from ttt.research.preregistration import bridge_effect_from_losses


SEED_RE = re.compile(r"-seed(\d{3})$")
S2_STAGE = "S2_125M"
S2_MINUS_STAGE = "S2_MINUS_125M"
SUCCEEDED_RUN_STATUSES = {"succeeded", "resumed_succeeded"}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            seeds.extend(range(int(left), int(right) + 1))
        else:
            seeds.append(int(item))
    return sorted(set(seeds))


def _matches_variant(run_id: str, variant: str) -> bool:
    return bool(re.search(rf"-{re.escape(variant)}-seed\d{{3}}$", run_id))


def paired_losses(
    eval_summary: dict[str, Any],
    *,
    variant: str = "",
    seeds: list[int] | None = None,
    min_seeds: int = 3,
) -> tuple[list[int], list[float], list[float]]:
    """Return (seeds, S2 losses, S2-minus losses) for seeds present in both arms."""

    candidates: dict[str, dict[int, list[dict[str, Any]]]] = {S2_STAGE: {}, S2_MINUS_STAGE: {}}
    for row in eval_summary.get("rows", []):
        if not isinstance(row, dict):
            continue
        stage_id = str(row.get("stage_id", ""))
        if stage_id not in candidates:
            continue
        match = SEED_RE.search(str(row.get("run_id", "")))
        if match is None:
            continue
        candidates[stage_id].setdefault(int(match.group(1)), []).append(row)

    losses: dict[str, dict[int, float]] = {S2_STAGE: {}, S2_MINUS_STAGE: {}}
    for stage_id, by_seed in candidates.items():
        for seed, rows in by_seed.items():
            if variant:
                tagged = [row for row in rows if _matches_variant(str(row["run_id"]), variant)]
                # S2-minus carries the variant tag only when the variant changes it.
                rows = tagged if (tagged or stage_id == S2_STAGE) else rows
            if not rows:
                continue
            if len(rows) > 1:
                run_ids = sorted(str(row["run_id"]) for row in rows)
                raise ValueError(
                    f"Several {stage_id} runs for seed {seed}: {run_ids}. Pass --variant to pick one."
                )
            row = rows[0]
            if str(row.get("status", "")) != "succeeded":
                raise ValueError(f"Cannot analyze a failed eval row: {row}")
            losses[stage_id][seed] = float(row["loss_mean"])

    if seeds is None:
        seeds = sorted(set(losses[S2_STAGE]) & set(losses[S2_MINUS_STAGE]))
        unpaired = sorted(set(losses[S2_STAGE]) ^ set(losses[S2_MINUS_STAGE]))
        if unpaired:
            print(f"Note: seeds without a pair are left out: {unpaired}")
    missing = {stage: sorted(set(seeds) - set(values)) for stage, values in losses.items()}
    if any(missing.values()):
        raise ValueError(f"Missing paired losses: {missing}")
    if len(seeds) < min_seeds:
        raise ValueError(f"E1 needs at least {min_seeds} paired seeds, found {len(seeds)}: {seeds}")
    return seeds, [losses[S2_STAGE][s] for s in seeds], [losses[S2_MINUS_STAGE][s] for s in seeds]


def _gpu_hours(run_summary: dict[str, Any]) -> tuple[float, dict[str, float]]:
    total = 0.0
    by_stage: dict[str, float] = {}
    for row in run_summary.get("rows", []):
        if not isinstance(row, dict) or str(row.get("status", "")) not in SUCCEEDED_RUN_STATUSES:
            continue
        if row.get("gpu_hours") is None:
            continue
        gpu_hours = float(row["gpu_hours"])
        stage_id = str(row["stage_id"])
        total += gpu_hours
        by_stage[stage_id] = by_stage.get(stage_id, 0.0) + gpu_hours
    return total, by_stage


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "s2_loss", "s2_minus_loss", "delta_s2_minus_minus_s2"])
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, analysis: dict[str, Any]) -> None:
    lines = [
        "# E1 bridge-effect analysis",
        "",
        f"Paper run id: `{analysis['paper_run_id']}`. Variant: `{analysis['variant'] or 'none'}`.",
        f"Decision: `{analysis['decision']}`.",
        "",
        "Statistic: `loss(S2_MINUS_s) - loss(S2_s)` on paired seeds. Positive means the bridge lowers the loss.",
        "",
        f"- Mean S2 loss: `{analysis['s2_loss_mean']:.6f}`",
        f"- Mean S2-minus loss: `{analysis['s2_minus_loss_mean']:.6f}`",
        f"- Mean delta: `{analysis['delta_mean']:.6f}`",
        f"- 95% paired seed bootstrap CI: `[{analysis['delta_ci95']['low']:.6f}, {analysis['delta_ci95']['high']:.6f}]`",
        f"- Margin: `{analysis['margin']}`",
        "",
        "| seed | S2 | S2-minus | delta |",
        "|---:|---:|---:|---:|",
    ]
    for row in analysis["paired_rows"]:
        lines.append(
            "| {seed} | {s2_loss:.6f} | {s2_minus_loss:.6f} | {delta_s2_minus_minus_s2:.6f} |".format(**row)
        )
    if analysis["training_gpu_hours"] is not None:
        gpu = analysis["training_gpu_hours"]
        lines += ["", f"Training GPU-hours in the run summary: `{gpu['total']:.3f}`."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-summary", type=Path, required=True, help="Eval summary JSON from 34_eval_matrix_jax.py.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path, default=None, help="Optional ladder summary JSON from 96 (GPU-hours).")
    parser.add_argument("--paper-run-id", default="", help="Defaults to the eval summary's paper_run_id.")
    parser.add_argument("--variant", default="", help="Bridge variant tag in the run ids, e.g. lr1e-3.")
    parser.add_argument("--seeds", default="", help="Seeds to require, e.g. 1-3. Default: every paired seed.")
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--margin", type=float, default=0.10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    eval_summary = _load_json(args.eval_summary)
    seeds, s2_losses, s2_minus_losses = paired_losses(
        eval_summary,
        variant=args.variant.strip(),
        seeds=_parse_seeds(args.seeds) if args.seeds.strip() else None,
        min_seeds=args.min_seeds,
    )
    result = bridge_effect_from_losses(
        s2_minus_losses=s2_minus_losses,
        s2_losses=s2_losses,
        margin=args.margin,
        n_resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    paired_rows = [
        {"seed": seed, "s2_loss": s2, "s2_minus_loss": s2m, "delta_s2_minus_minus_s2": delta}
        for seed, s2, s2m, delta in zip(seeds, s2_losses, s2_minus_losses, result.deltas, strict=True)
    ]
    gpu = None
    if args.run_summary is not None:
        total, by_stage = _gpu_hours(_load_json(args.run_summary))
        gpu = {"total": total, "by_stage": by_stage, "run_summary": str(args.run_summary)}

    analysis = {
        "schema_version": "2.0",
        "paper_run_id": args.paper_run_id.strip() or str(eval_summary.get("paper_run_id", "")),
        "variant": args.variant.strip(),
        "eval_summary": str(args.eval_summary),
        "eval_id": eval_summary.get("eval_id"),
        "preregistered_statistic": "delta_bridge_s = loss(S2_MINUS_s) - loss(S2_s)",
        "margin": args.margin,
        "seeds": seeds,
        "paired_rows": paired_rows,
        "s2_loss_mean": mean(s2_losses),
        "s2_minus_loss_mean": mean(s2_minus_losses),
        "delta_mean": result.mean_delta,
        "delta_ci95": {"low": result.ci95.low, "high": result.ci95.high},
        "decision": result.decision,
        "bootstrap": {
            "n_resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "level": 0.95,
            "unit": "paired training seed delta",
        },
        "training_gpu_hours": gpu,
    }

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e1_bridge_effect_analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(out_dir / "e1_bridge_effect_paired_losses.csv", paired_rows)
    _write_markdown(out_dir / "e1_bridge_effect_analysis.md", analysis)
    print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
