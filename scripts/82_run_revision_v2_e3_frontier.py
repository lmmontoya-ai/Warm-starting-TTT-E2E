#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from ttt.research.orchestrator import OrchestratorOptions, run_stage
from ttt.research.registry import load_registry
from ttt.research.revision_v2_e3 import (
    BRIDGE_CONTEXT_LENGTH,
    CANONICAL_EXTENSION_STEPS,
    E3_ARMS,
    E3_SEED,
    EXTENSION_CONTEXT_LENGTH,
    EXTENSION_GLOBAL_BATCH_SIZE,
    S2_MINUS_CONTINUATION_ADDITIONAL_STEPS,
    S2_MINUS_PARENT_FINAL_STEP,
    assert_e3_budget_arithmetic,
    assert_e3_registry_stages,
    bridge_tokens_for_steps,
    continuation_target_total_steps,
)
from ttt.research.types import BudgetSpec, CheckpointRef


DEFAULT_LAUNCH_ORDER = ("40", "20", "5", "cont")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _latest_step(checkpoint_dir: Path) -> int | None:
    latest_json = checkpoint_dir / "latest.json"
    if not latest_json.exists():
        return None
    payload = _load_json(latest_json)
    return int(payload["step"])


def _run_succeeded(run_dir: Path) -> bool:
    result_path = run_dir / "run_result.json"
    if not result_path.exists():
        return False
    try:
        payload = _load_json(result_path)
    except Exception:
        return False
    return str(payload.get("status", "")).strip() == "succeeded"


def _parse_order(raw: str) -> list[str]:
    requested = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not requested:
        raise ValueError("At least one E3 launch item is required")
    allowed = set(DEFAULT_LAUNCH_ORDER)
    unknown = [item for item in requested if item not in allowed]
    if unknown:
        raise ValueError(f"Unknown E3 launch item(s): {unknown}; allowed={sorted(allowed)}")
    return requested


def _checkpoint_ref(
    *,
    checkpoint_id: str,
    exp_folder: str,
    exp_name: str,
    checkpoint_path: Path,
    step: int | None,
) -> CheckpointRef:
    return CheckpointRef(
        checkpoint_id=checkpoint_id,
        exp_folder=exp_folder,
        exp_name=exp_name,
        step=step,
        checkpoint_path=str(checkpoint_path.expanduser().resolve()),
    )


def _export_stage(
    *,
    repo_root: Path,
    paper_run_id: str,
    stage_id: str,
    run_id: str,
    repo_id: str,
    exp_dir: Path,
    checkpoint_root: Path,
    checkpoint_exp_folder: str,
    dry_run: bool,
) -> dict[str, Any]:
    cmd = [
        "uv",
        "run",
        "--exact",
        "python",
        "scripts/40_export_stage_to_hf.py",
        "--paper-run-id",
        paper_run_id,
        "--stage-id",
        stage_id,
        "--run-id",
        run_id,
        "--repo-id",
        repo_id,
        "--exp-dir",
        str(exp_dir),
        "--checkpoint-root",
        str(checkpoint_root),
        "--checkpoint-exp-folder",
        checkpoint_exp_folder,
    ]
    if dry_run:
        return {"status": "dry_run", "command": cmd}
    completed = subprocess.run(cmd, cwd=repo_root, check=False)
    return {"status": "succeeded" if completed.returncode == 0 else "failed", "returncode": completed.returncode}


def _opts(
    args: argparse.Namespace,
    *,
    paper_run_id: str,
    exp_folder: str,
    seq_length: int,
) -> OrchestratorOptions:
    return OrchestratorOptions(
        deploy=args.deploy,
        runtime_mode=args.runtime_mode,
        exp_dir=args.exp_dir.expanduser().resolve(),
        checkpoint_root=args.checkpoint_root.expanduser().resolve(),
        profile_root=args.profile_root.expanduser().resolve(),
        dclm_root=args.dclm_root.expanduser().resolve(),
        books_root=args.books_root.expanduser().resolve(),
        exp_folder=exp_folder,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_key=args.wandb_key,
        global_batch_size=args.bridge_global_batch_size,
        ext_global_batch_size=args.ext_global_batch_size,
        seq_length=seq_length,
        save_milestone_freq=args.save_milestone_freq,
        dry_run=args.dry_run,
        paper_run_id=paper_run_id,
        require_dataset_fingerprint=(not args.dry_run),
    )


def _run_or_skip(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    stage_map: dict[str, Any],
    stage_id: str,
    run_id: str,
    paper_run_id: str,
    exp_folder: str,
    budget: BudgetSpec,
    parent_ref: CheckpointRef,
    initial_parent_path: Path,
    target_total_steps: int,
    extra_overrides: list[str],
    extra_tags: dict[str, str],
    seq_length: int,
) -> dict[str, Any]:
    stage = stage_map[stage_id]
    opts = _opts(args, paper_run_id=paper_run_id, exp_folder=exp_folder, seq_length=seq_length)
    checkpoint_dir = opts.checkpoint_root / exp_folder / run_id
    run_dir = opts.exp_dir / paper_run_id / stage_id / run_id
    latest_step = _latest_step(checkpoint_dir)

    explicit_resume = initial_parent_path
    explicit_resume_format = "orbax"
    run_overrides = list(extra_overrides)
    status_prefix = ""
    if latest_step is not None and latest_step >= target_total_steps - 1 and args.skip_complete:
        row = {
            "stage_id": stage_id,
            "run_id": run_id,
            "paper_run_id": paper_run_id,
            "checkpoint_dir": str(checkpoint_dir),
            "latest_step": latest_step,
            "target_total_steps": target_total_steps,
            "status": "skipped_complete_checkpoint",
        }
        if args.export_to_hf and _run_succeeded(run_dir):
            row["hf_export"] = _export_stage(
                repo_root=repo_root,
                paper_run_id=paper_run_id,
                stage_id=stage_id,
                run_id=run_id,
                repo_id=args.hf_repo_id,
                exp_dir=opts.exp_dir,
                checkpoint_root=opts.checkpoint_root,
                checkpoint_exp_folder=exp_folder,
                dry_run=args.dry_run,
            )
        return row

    if latest_step is not None and args.resume_existing:
        explicit_resume = checkpoint_dir
        run_overrides.append("training.load_part=all")
        parent_ref = _checkpoint_ref(
            checkpoint_id=stage_id,
            exp_folder=exp_folder,
            exp_name=run_id,
            checkpoint_path=checkpoint_dir,
            step=latest_step,
        )
        status_prefix = "resumed_"

    result = run_stage(
        stage=stage,
        stage_map=stage_map,
        opts=opts,
        budget=budget,
        eval_spec=stage_map["_eval_spec"],
        repo_root=repo_root,
        run_id=run_id,
        explicit_resume_checkpoint_path=explicit_resume,
        explicit_resume_checkpoint_format=explicit_resume_format,
        parent_refs_override=[parent_ref],
        extra_overrides=run_overrides,
        extra_tags=extra_tags,
    )
    row = {
        "stage_id": stage_id,
        "run_id": run_id,
        "paper_run_id": paper_run_id,
        "checkpoint_dir": str(checkpoint_dir),
        "latest_step_before": latest_step,
        "target_total_steps": target_total_steps,
        "status": status_prefix + result.status,
        "run_dir": result.run_dir,
        "gpu_hours": result.gpu_hours,
        "tokens_seen": result.tokens_seen,
        "error_message": result.error_message,
    }
    if args.export_to_hf and result.status == "succeeded":
        row["hf_export"] = _export_stage(
            repo_root=repo_root,
            paper_run_id=paper_run_id,
            stage_id=stage_id,
            run_id=run_id,
            repo_id=args.hf_repo_id,
            exp_dir=opts.exp_dir,
            checkpoint_root=opts.checkpoint_root,
            checkpoint_exp_folder=exp_folder,
            dry_run=args.dry_run,
        )
    return row


def _validity_checks(stage_map: dict[str, Any]) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    for name, fn in (
        ("e3_budget_arithmetic", assert_e3_budget_arithmetic),
        ("e3_registry_topology", lambda: assert_e3_registry_stages(stage_map)),
    ):
        try:
            fn()
            checks.append({"check": name, "status": "PASS", "reason": ""})
        except Exception as exc:
            checks.append({"check": name, "status": "FAIL", "reason": str(exc)})

    expected_total = continuation_target_total_steps()
    checks.append(
        {
            "check": "s2_minus_continuation_total_steps",
            "status": "PASS" if expected_total == 1920 else "FAIL",
            "reason": f"parent_final_step={S2_MINUS_PARENT_FINAL_STEP}; additional_steps={S2_MINUS_CONTINUATION_ADDITIONAL_STEPS}; total_steps={expected_total}",
        }
    )
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or dry-run revision-v2 E3 bridge frontier arms plus S2-minus continuation."
    )
    parser.add_argument("--registry", type=Path, default=Path("./configs/research/warmstart_registry.yaml"))
    parser.add_argument("--deploy", default="revision_v2_prime_h100_2x")
    parser.add_argument("--runtime-mode", default="jax_train", choices=["simulate", "token_stats", "jax_train"])
    parser.add_argument("--e3-paper-run-id", default="revision_v2_e3_frontier_v1")
    parser.add_argument("--cont-paper-run-id", default="revision_v2_s2minus_cont_v1")
    parser.add_argument("--e3-exp-folder", default="")
    parser.add_argument("--cont-exp-folder", default="")
    parser.add_argument("--shared-parent-exp-folder", default="revision_v2_e1_paired_v1")
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--profile-root", type=Path, default=Path("./artifacts/external_models"))
    parser.add_argument("--dclm-root", type=Path, required=True)
    parser.add_argument("--books-root", type=Path, required=True)
    parser.add_argument("--fa-parent-checkpoint-path", type=Path, default=None)
    parser.add_argument("--s2-minus-parent-checkpoint-path", type=Path, default=None)
    parser.add_argument("--order", default=",".join(DEFAULT_LAUNCH_ORDER))
    parser.add_argument("--ext-steps", type=int, default=CANONICAL_EXTENSION_STEPS)
    parser.add_argument("--bridge-global-batch-size", type=int, default=64)
    parser.add_argument("--ext-global-batch-size", type=int, default=EXTENSION_GLOBAL_BATCH_SIZE)
    parser.add_argument(
        "--bridge-accum-steps",
        type=int,
        default=0,
        help=(
            "Optional training.accum_steps override for E3 bridge/adapt stages only. "
            "Use this to preserve the preregistered global batch on smaller GPU "
            "topologies by reducing the per-device microbatch."
        ),
    )
    parser.add_argument(
        "--ext-accum-steps",
        type=int,
        default=0,
        help="Optional training.accum_steps override for E3 extension stages only.",
    )
    parser.add_argument(
        "--cont-accum-steps",
        type=int,
        default=0,
        help="Optional training.accum_steps override for the S2-minus continuation stage only.",
    )
    parser.add_argument("--seq-length", type=int, default=EXTENSION_CONTEXT_LENGTH)
    parser.add_argument("--save-milestone-freq", type=int, default=30)
    parser.add_argument("--wandb-entity", default="none")
    parser.add_argument("--wandb-project", default="none")
    parser.add_argument("--wandb-key", default="none")
    parser.add_argument("--eval-profile", default="default_longctx")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-complete", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-to-hf", action="store_true")
    parser.add_argument("--hf-repo-id", default="Luxel/ttt-e2e-125m-results")
    parser.add_argument("--summary-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    registry = load_registry(args.registry.expanduser().resolve())
    stage_map: dict[str, Any] = registry.stage_map()
    stage_map["_eval_spec"] = registry.eval_profiles[args.eval_profile]

    validity_checks = _validity_checks(stage_map)
    if any(check["status"] != "PASS" for check in validity_checks):
        for check in validity_checks:
            print(f"{check['status']}: {check['check']} {check['reason']}")
        return 1

    order = _parse_order(args.order)
    e3_exp_folder = args.e3_exp_folder.strip() or args.e3_paper_run_id
    cont_exp_folder = args.cont_exp_folder.strip() or args.cont_paper_run_id
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    fa_parent_path = (
        args.fa_parent_checkpoint_path.expanduser().resolve()
        if args.fa_parent_checkpoint_path is not None
        else checkpoint_root / args.shared_parent_exp_folder / "pretrain-125m-fa"
    )
    s2_minus_parent_path = (
        args.s2_minus_parent_checkpoint_path.expanduser().resolve()
        if args.s2_minus_parent_checkpoint_path is not None
        else checkpoint_root / args.shared_parent_exp_folder / "ext-125m-e2e-32K-from-fa-direct-seed001"
    )

    fa_parent_ref = _checkpoint_ref(
        checkpoint_id="S0_PRETRAIN_FA_125M",
        exp_folder=args.shared_parent_exp_folder,
        exp_name="pretrain-125m-fa",
        checkpoint_path=fa_parent_path,
        step=None,
    )

    rows: list[dict[str, Any]] = []
    failed = False
    arms_by_key = {str(arm.percent): arm for arm in E3_ARMS}
    for item in order:
        if item == "cont":
            parent_step = _latest_step(s2_minus_parent_path)
            if parent_step is None:
                parent_step = S2_MINUS_PARENT_FINAL_STEP
            target_total_steps = continuation_target_total_steps(parent_final_step=parent_step)
            parent_ref = _checkpoint_ref(
                checkpoint_id="S2_MINUS_125M",
                exp_folder=args.shared_parent_exp_folder,
                exp_name="ext-125m-e2e-32K-from-fa-direct-seed001",
                checkpoint_path=s2_minus_parent_path,
                step=parent_step,
            )
            cont_stage = stage_map["S2_MINUS_CONT_125M"]
            row = _run_or_skip(
                args=args,
                repo_root=repo_root,
                stage_map=stage_map,
                stage_id=cont_stage.stage_id,
                run_id=cont_stage.exp_name,
                paper_run_id=args.cont_paper_run_id,
                exp_folder=cont_exp_folder,
                budget=BudgetSpec(budget_id="revision_v2_s2minus_cont", ext_steps=target_total_steps, seed=E3_SEED),
                parent_ref=parent_ref,
                initial_parent_path=s2_minus_parent_path,
                target_total_steps=target_total_steps,
                extra_overrides=[
                    f"training.model_seed={E3_SEED}",
                    f"training.data_seed={E3_SEED}",
                    *(
                        [f"training.accum_steps={int(args.cont_accum_steps)}"]
                        if int(args.cont_accum_steps) > 0
                        else []
                    ),
                ],
                extra_tags={
                    "revision_v2_arm": "s2_minus_continuation",
                    "additional_steps": str(S2_MINUS_CONTINUATION_ADDITIONAL_STEPS),
                    "parent_final_step": str(parent_step),
                },
                seq_length=args.seq_length,
            )
            rows.append(row)
            failed = failed or str(row["status"]).endswith("failed")
            continue

        arm = arms_by_key[item]
        adapt_budget = BudgetSpec(
            budget_id=f"revision_v2_e3_bridge_{arm.percent}pct",
            adapt_steps=arm.bridge_steps,
            seed=E3_SEED,
        )
        adapt_row = _run_or_skip(
            args=args,
            repo_root=repo_root,
            stage_map=stage_map,
            stage_id=arm.adapt_stage_id,
            run_id=arm.adapt_run_id,
            paper_run_id=args.e3_paper_run_id,
            exp_folder=e3_exp_folder,
            budget=adapt_budget,
            parent_ref=fa_parent_ref,
            initial_parent_path=fa_parent_path,
            target_total_steps=arm.bridge_steps,
            extra_overrides=[
                f"training.model_seed={E3_SEED}",
                f"training.data_seed={E3_SEED}",
                *(
                    [f"training.accum_steps={int(args.bridge_accum_steps)}"]
                    if int(args.bridge_accum_steps) > 0
                    else []
                ),
            ],
            extra_tags={
                "revision_v2_arm": f"bridge_{arm.percent}pct_adapt",
                "bridge_percent": str(arm.percent),
                "bridge_steps": str(arm.bridge_steps),
                "bridge_tokens": str(bridge_tokens_for_steps(arm.bridge_steps)),
            },
            seq_length=BRIDGE_CONTEXT_LENGTH,
        )
        rows.append(adapt_row)
        if str(adapt_row["status"]).endswith("failed"):
            failed = True
            break

        adapt_checkpoint_dir = checkpoint_root / e3_exp_folder / arm.adapt_run_id
        adapt_step = _latest_step(adapt_checkpoint_dir)
        if adapt_step is None:
            adapt_step = arm.bridge_steps - 1
        adapt_parent_ref = _checkpoint_ref(
            checkpoint_id=arm.adapt_stage_id,
            exp_folder=e3_exp_folder,
            exp_name=arm.adapt_run_id,
            checkpoint_path=adapt_checkpoint_dir,
            step=adapt_step,
        )
        ext_budget = BudgetSpec(
            budget_id=f"revision_v2_e3_ext_{arm.percent}pct",
            ext_steps=args.ext_steps,
            seed=E3_SEED,
        )
        ext_row = _run_or_skip(
            args=args,
            repo_root=repo_root,
            stage_map=stage_map,
            stage_id=arm.final_stage_id,
            run_id=arm.final_run_id,
            paper_run_id=args.e3_paper_run_id,
            exp_folder=e3_exp_folder,
            budget=ext_budget,
            parent_ref=adapt_parent_ref,
            initial_parent_path=adapt_checkpoint_dir,
            target_total_steps=args.ext_steps,
            extra_overrides=[
                f"training.model_seed={E3_SEED}",
                f"training.data_seed={E3_SEED}",
                *(
                    [f"training.accum_steps={int(args.ext_accum_steps)}"]
                    if int(args.ext_accum_steps) > 0
                    else []
                ),
            ],
            extra_tags={
                "revision_v2_arm": f"bridge_{arm.percent}pct_extension",
                "bridge_percent": str(arm.percent),
                "bridge_steps": str(arm.bridge_steps),
                "bridge_tokens": str(bridge_tokens_for_steps(arm.bridge_steps)),
            },
            seq_length=args.seq_length,
        )
        rows.append(ext_row)
        failed = failed or str(ext_row["status"]).endswith("failed")
        if failed:
            break

    summary_out = (
        args.summary_out.expanduser().resolve()
        if args.summary_out is not None
        else (Path("./reports/revision_v2/e3_frontier") / "run_summary.json").resolve()
    )
    _write_json(
        summary_out,
        {
            "schema_version": "1.0",
            "e3_paper_run_id": args.e3_paper_run_id,
            "cont_paper_run_id": args.cont_paper_run_id,
            "deploy": args.deploy,
            "runtime_mode": args.runtime_mode,
            "order": order,
            "seed": E3_SEED,
            "ext_steps": args.ext_steps,
            "ext_global_batch_size": args.ext_global_batch_size,
            "bridge_accum_steps": int(args.bridge_accum_steps),
            "ext_accum_steps": int(args.ext_accum_steps),
            "cont_accum_steps": int(args.cont_accum_steps),
            "seq_length": args.seq_length,
            "fa_parent_checkpoint_path": str(fa_parent_path),
            "s2_minus_parent_checkpoint_path": str(s2_minus_parent_path),
            "validity_checks": validity_checks,
            "rows": rows,
        },
    )
    print(f"Wrote E3 frontier summary: {summary_out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
