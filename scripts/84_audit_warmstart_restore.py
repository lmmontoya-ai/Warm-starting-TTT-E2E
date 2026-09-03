#!/usr/bin/env python3
"""Audit which parameters a params warm start actually inherits.

Builds the target model for a Hydra experiment on CPU, restores a parent
checkpoint through the same loader the trainer uses, and prints a
parameter-count coverage report grouped by component. Exits non-zero when any
inherited tensor was silently left at fresh initialization.

Examples:

    uv run --exact python scripts/84_audit_warmstart_restore.py \
        --experiment 125m/pretrained/adapt-125m-e2e-8K-from-fa \
        --checkpoint-dir checkpoints/revision_v3_widthpreserving_v1/pretrain-125m-fa

    uv run --exact python scripts/84_audit_warmstart_restore.py \
        --registry configs/research/warmstart_registry.yaml --stage-id S2_MINUS_125M \
        --checkpoint-root ./checkpoints --exp-folder revision_v3_widthpreserving_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
from hydra import compose, initialize_config_dir

from ttt.config import TrainingConfig, register_configs
from ttt.jax_runtime.checkpoint import OrbaxCheckpointer
from ttt.jax_runtime.model.transformer import MetaModel
from ttt.jax_runtime.warmstart_guard import WarmStartValidationError, check_restore_report
from ttt.research.registry import load_registry
from ttt.research.warmstart_validity import parent_stage_for


if not hasattr(jax.monitoring, "record_scalar"):  # pragma: no cover - env compatibility
    jax.monitoring.record_scalar = lambda *a, **k: None


def _compose(repo_root: Path, experiment: str, overrides: list[str]):
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=str(repo_root / "configs")):
        cfg = compose(config_name="config", overrides=["+deploy=interactive", f"+experiment={experiment}", *overrides])
    cfg.training.runtime_mode = "jax_train"
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="", help="Hydra experiment of the child (target) model.")
    parser.add_argument("--override", action="append", default=[], help="Extra Hydra override for the child (repeatable).")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Parent checkpoint dir (contains latest.json).")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--registry", type=Path, default=Path("./configs/research/warmstart_registry.yaml"))
    parser.add_argument("--stage-id", default="", help="Resolve experiment/overrides/parent from a registry stage.")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--exp-folder", default="")
    parser.add_argument("--allow-shape-mismatch", action="store_true")
    parser.add_argument("--max-new-param-fraction", type=float, default=0.25)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    experiment = args.experiment
    overrides = list(args.override)
    checkpoint_dir = args.checkpoint_dir
    label = experiment
    if args.stage_id:
        registry = load_registry(args.registry.expanduser().resolve())
        stage_map = registry.stage_map()
        stage = stage_map[args.stage_id]
        experiment = stage.experiment
        overrides = [*stage.extra_overrides, *overrides]
        label = stage.stage_id
        if checkpoint_dir is None:
            child_cfg = _compose(repo_root, experiment, [o for o in overrides if o.startswith(("training.", "model."))])
            parent = parent_stage_for(stage, stage_map, {"resume_exp_name": str(child_cfg.training.resume_exp_name)})
            if parent is None:
                raise SystemExit(f"Could not resolve the parent stage for {stage.stage_id}; pass --checkpoint-dir.")
            if not args.exp_folder:
                raise SystemExit("--exp-folder is required with --stage-id when --checkpoint-dir is not given.")
            checkpoint_dir = args.checkpoint_root / args.exp_folder / parent.exp_name
    if not experiment:
        raise SystemExit("Provide --experiment or --stage-id.")
    if checkpoint_dir is None:
        raise SystemExit("Provide --checkpoint-dir (or --stage-id with --exp-folder).")

    cfg = _compose(repo_root, experiment, [o for o in overrides if o.startswith(("training.", "model."))])
    model, _ = eqx.nn.make_with_state(MetaModel)(cfg, key=jax.random.PRNGKey(0))
    weights = model.weights()
    checkpointer = OrbaxCheckpointer(checkpoint_dir.expanduser().resolve(), for_saving=False)
    try:
        payload = checkpointer.load(step=args.step, targets={"model_weights": weights}, restore=TrainingConfig.LoadPart.params)
    finally:
        checkpointer.close()
    report = payload.report
    if report is None:
        raise SystemExit("Loader returned no coverage report (legacy checkpoint format?).")

    print(f"target: {label} [{experiment}]")
    print(f"parent: {checkpoint_dir} step={payload.step}")
    for line in report.summary_lines():
        print(line)
    if report.mismatched_paths:
        print("shape-mismatched tensors:")
        for path in report.mismatched_paths:
            print(f"  {path}")
    if report.missed_paths:
        print(f"new tensors (absent from checkpoint): {len(report.missed_paths)}")
        for path in report.missed_paths[:12]:
            print(f"  {path}")

    result: dict[str, Any] = {
        "target_label": label,
        "experiment": experiment,
        "overrides": overrides,
        "checkpoint_dir": str(checkpoint_dir),
        "restore_step": payload.step,
        "report": report.to_dict(),
        "status": "PASS",
    }
    try:
        check_restore_report(
            report,
            allow_shape_mismatch=bool(args.allow_shape_mismatch),
            max_new_param_fraction=float(args.max_new_param_fraction),
            label=label,
        )
        print("WARM-START RESTORE AUDIT: PASS")
        code = 0
    except WarmStartValidationError as exc:
        result["status"] = "FAIL"
        result["error"] = str(exc)
        print(f"WARM-START RESTORE AUDIT: FAIL\n{exc}", file=sys.stderr)
        code = 1
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {args.json_out}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
