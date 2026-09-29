#!/usr/bin/env python3
"""Audit what a params warm start actually inherits, by value and by loss.

Builds the target model for a Hydra experiment on CPU, restores a parent
checkpoint through the same loader and apply function the trainer uses, and
reports three things:

1. coverage: parameter counts restored / shape-mismatched / new, by component;
2. value check: every same-shape tensor is compared with the checkpoint array
   after the restore (the revision-v2 fallback restore reported coverage but
   left every tensor at its fresh initialization);
3. optional loss probe: mean next-token loss of the converted model on a few
   real validation sequences, next to the parent model on the same sequences.

Exits non-zero when any gate fails.

Examples:

    uv run --exact python scripts/84_audit_warmstart_restore.py \
        --experiment 125m/pretrained/adapt-125m-e2e-8K-from-fa \
        --checkpoint-dir checkpoints/revision_v3_widthpreserving_v1/pretrain-125m-fa

    uv run --exact python scripts/84_audit_warmstart_restore.py \
        --registry configs/research/warmstart_registry.yaml --stage-id S2_MINUS_125M \
        --checkpoint-root ./checkpoints --exp-folder revision_v3_widthpreserving_v1 \
        --probe-root /data/books3 --probe-parent
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize_config_dir

from ttt.config import TrainingConfig, register_configs
from ttt.jax_runtime.checkpoint import OrbaxCheckpointer, apply_restored_weights, verify_restored_values
from ttt.jax_runtime.model.transformer import MetaModel
from ttt.jax_runtime.warmstart_guard import (
    WarmStartValidationError,
    check_initial_loss,
    check_restore_report,
    check_restored_values,
)
from ttt.research.registry import load_registry
from ttt.research.warmstart_validity import parent_stage_for


if not hasattr(jax.monitoring, "record_scalar"):  # pragma: no cover - env compatibility
    jax.monitoring.record_scalar = lambda *a, **k: None


def _compose(repo_root: Path, experiment: str, overrides: list[str]):
    register_configs()
    placeholders = ["deploy_paths.data.dclm_filter_8k=/dev/null", "deploy_paths.data.books3=/dev/null"]
    with initialize_config_dir(version_base=None, config_dir=str(repo_root / "configs")):
        cfg = compose(config_name="config", overrides=["+deploy=interactive", f"+experiment={experiment}", *placeholders, *overrides])
    cfg.training.runtime_mode = "jax_train"
    return cfg


def _model_overrides(overrides: list[str]) -> list[str]:
    return [o for o in overrides if o.split("=", 1)[0].lstrip("+").startswith(("training.", "model."))]


def _probe_overrides(context: int, experiment_cfg) -> list[str]:
    """Overrides that let a model score ``context``-token sequences on CPU."""

    out = [f"training.seq_length={context}", f"model.seq_len={context}"]
    mini_batch = int(experiment_cfg.model.mini_batch_size)
    if mini_batch > context or context % mini_batch != 0:
        # Full-attention configs set mini_batch_size to the whole training context.
        out.append(f"model.mini_batch_size={context}")
    return out


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
    parser.add_argument("--probe-root", type=Path, default=None, help="Tokenized dataset root for the loss probe (e.g. books3 or dclm_filter_8k).")
    parser.add_argument("--probe-split", default="val")
    parser.add_argument("--probe-seqs", type=int, default=4)
    parser.add_argument("--probe-context", type=int, default=2048)
    parser.add_argument("--probe-step", type=int, default=0, help="Training step index for the inner-LR schedule (0 = what the trainer sees first).")
    parser.add_argument("--probe-parent", action="store_true", help="Also score the parent model (registry parent experiment or --parent-experiment).")
    parser.add_argument("--parent-experiment", default="", help="Parent Hydra experiment when not using --stage-id.")
    parser.add_argument("--max-probe-loss", type=float, default=7.0, help="Fail when the converted model's probe loss exceeds this (0 disables).")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _probe_batch(cfg, root: Path, split: str, seqs: int, context: int):
    import grain.python as grain

    from ttt.dataloader import lm_dataset

    dataset = lm_dataset(
        path=str(root),
        split=split,
        seq_len=context,
        global_batch_size=seqs,
        bos_token_id=int(cfg.model.bos_token_id),
        eos_token_id=int(cfg.model.eos_token_id),
        seed=0,
        repeat=False,
        shuffle=False,
    )
    return next(iter(dataset.to_iter_dataset(grain.ReadOptions(num_threads=1, prefetch_buffer_size=1))))


def _score(model: MetaModel, state, batch, *, step: int) -> list[float]:
    state = state.set(model.step_index, jnp.array(step, dtype=jnp.int32))
    loss_fn = eqx.filter_jit(lambda m, seq, st: m.loss_for_sequence(seq, st)[0])
    mesh = jax.make_mesh((1, 1), ("data", "state"))
    losses: list[float] = []
    with mesh:
        for i in range(int(batch.input_ids.shape[0])):
            seq = jax.tree.map(lambda x: jnp.asarray(x[i]), batch)
            losses.append(float(loss_fn(model, seq, state)))
    return losses


def _load_into(cfg, checkpoint_dir: Path, step: int | None):
    model, state = eqx.nn.make_with_state(MetaModel)(cfg, key=jax.random.PRNGKey(0))
    checkpointer = OrbaxCheckpointer(checkpoint_dir.expanduser().resolve(), for_saving=False)
    try:
        payload = checkpointer.load(step=step, targets={"model_weights": model.weights()}, restore=TrainingConfig.LoadPart.params)
    finally:
        checkpointer.close()
    model = apply_restored_weights(model, payload.model_weights)
    return model, state, payload


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    experiment = args.experiment
    overrides = list(args.override)
    checkpoint_dir = args.checkpoint_dir
    label = experiment
    parent_experiment = args.parent_experiment
    parent_overrides: list[str] = []
    if args.stage_id:
        registry = load_registry(args.registry.expanduser().resolve())
        stage_map = registry.stage_map()
        stage = stage_map[args.stage_id]
        experiment = stage.experiment
        overrides = [*stage.extra_overrides, *overrides]
        label = stage.stage_id
        child_cfg = _compose(repo_root, experiment, _model_overrides(overrides))
        parent = parent_stage_for(stage, stage_map, {"resume_exp_name": str(child_cfg.training.resume_exp_name)})
        if parent is not None and not parent_experiment:
            parent_experiment = parent.experiment
            parent_overrides = list(parent.extra_overrides)
        if checkpoint_dir is None:
            if parent is None:
                raise SystemExit(f"Could not resolve the parent stage for {stage.stage_id}; pass --checkpoint-dir.")
            if not args.exp_folder:
                raise SystemExit("--exp-folder is required with --stage-id when --checkpoint-dir is not given.")
            checkpoint_dir = args.checkpoint_root / args.exp_folder / parent.exp_name
    if not experiment:
        raise SystemExit("Provide --experiment or --stage-id.")
    if checkpoint_dir is None:
        raise SystemExit("Provide --checkpoint-dir (or --stage-id with --exp-folder).")

    base_cfg = _compose(repo_root, experiment, _model_overrides(overrides))
    cfg = base_cfg
    if args.probe_root is not None:
        cfg = _compose(repo_root, experiment, [*_model_overrides(overrides), *_probe_overrides(args.probe_context, base_cfg)])
    model, state, payload = _load_into(cfg, checkpoint_dir, args.step)
    report = payload.report
    if report is None:
        raise SystemExit("Loader returned no coverage report (legacy checkpoint format?).")
    value_check = verify_restored_values(model.weights(), payload.model_weights)

    print(f"target: {label} [{experiment}]")
    print(f"parent: {checkpoint_dir} step={payload.step}")
    for line in report.summary_lines():
        print(line)
    print(value_check.summary_line())
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
        "value_check": value_check.to_dict(),
        "status": "PASS",
    }
    errors: list[str] = []
    try:
        check_restore_report(
            report,
            allow_shape_mismatch=bool(args.allow_shape_mismatch),
            max_new_param_fraction=float(args.max_new_param_fraction),
            label=label,
        )
    except WarmStartValidationError as exc:
        errors.append(str(exc))
    try:
        check_restored_values(value_check, report=report, label=label)
    except WarmStartValidationError as exc:
        errors.append(str(exc))

    if args.probe_root is not None:
        batch = _probe_batch(cfg, args.probe_root.expanduser().resolve(), args.probe_split, args.probe_seqs, args.probe_context)
        target_losses = _score(model, state, batch, step=args.probe_step)
        probe: dict[str, Any] = {
            "root": str(args.probe_root),
            "split": args.probe_split,
            "context": args.probe_context,
            "seqs": args.probe_seqs,
            "step": args.probe_step,
            "target_losses": target_losses,
            "target_mean": float(np.mean(target_losses)),
        }
        print(f"probe: {label} mean loss {probe['target_mean']:.4f} over {len(target_losses)} x {args.probe_context} tokens {np.round(target_losses, 3).tolist()}")
        if args.probe_parent:
            if not parent_experiment:
                raise SystemExit("--probe-parent needs --stage-id or --parent-experiment.")
            parent_base = _compose(repo_root, parent_experiment, _model_overrides(parent_overrides))
            parent_cfg = _compose(repo_root, parent_experiment, [*_model_overrides(parent_overrides), *_probe_overrides(args.probe_context, parent_base)])
            parent_model, parent_state, parent_payload = _load_into(parent_cfg, checkpoint_dir, args.step)
            parent_check = verify_restored_values(parent_model.weights(), parent_payload.model_weights)
            parent_losses = _score(parent_model, parent_state, batch, step=args.probe_step)
            probe.update({
                "parent_experiment": parent_experiment,
                "parent_value_check": parent_check.to_dict(),
                "parent_losses": parent_losses,
                "parent_mean": float(np.mean(parent_losses)),
                "delta_vs_parent": float(np.mean(target_losses) - np.mean(parent_losses)),
            })
            print(f"probe: parent [{parent_experiment}] mean loss {probe['parent_mean']:.4f}; converted - parent = {probe['delta_vs_parent']:+.4f}")
        result["probe"] = probe
        try:
            check_initial_loss(probe["target_mean"], max_initial_loss=float(args.max_probe_loss), step=args.probe_step, label=f"{label} probe")
        except WarmStartValidationError as exc:
            errors.append(str(exc))

    code = 0
    if errors:
        result["status"] = "FAIL"
        result["errors"] = errors
        print("WARM-START RESTORE AUDIT: FAIL\n" + "\n".join(errors), file=sys.stderr)
        code = 1
    else:
        print("WARM-START RESTORE AUDIT: PASS")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {args.json_out}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
