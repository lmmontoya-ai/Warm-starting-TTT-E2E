"""Stage orchestration helpers for warm-start research runs."""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .budget import build_budget_manifest, estimate_gpu_hours_from_wall, estimate_tokens
from .lineage import build_checkpoint_manifest, resolve_checkpoint_ref, resolve_stage_parents, validate_stage_profiles
from .tracking import (
    append_jsonl,
    detect_git_state,
    ensure_run_dir,
    environment_manifest,
    write_command_script,
    write_json,
    write_run_manifest,
    write_run_result,
    write_stage_manifest,
)
from .types import (
    BudgetSpec,
    CheckpointRef,
    DatasetRef,
    EvalSpec,
    RunResult,
    RunSpec,
    StageSpec,
    make_run_result,
    utc_now_iso,
)


@dataclass(frozen=True)
class OrchestratorOptions:
    deploy: str
    runtime_mode: str
    exp_dir: Path
    checkpoint_root: Path
    profile_root: Path
    dclm_root: Path
    books_root: Path
    exp_folder: str
    wandb_entity: str
    wandb_project: str
    wandb_key: str
    global_batch_size: int | None = None
    ext_global_batch_size: int | None = None
    accum_steps: int | None = None
    seq_length: int | None = None
    save_milestone_freq: int = 0
    dummy_dataset: bool = False
    dry_run: bool = False
    paper_run_id: str = "warmstart"
    require_dataset_fingerprint: bool = True



def stage_steps(stage: StageSpec, budget: BudgetSpec) -> int:
    if stage.kind == "pretrain":
        return budget.pretrain_steps
    if stage.kind == "adapt":
        return budget.adapt_steps
    if stage.kind == "ext":
        return budget.ext_steps
    raise ValueError(f"Unknown stage kind={stage.kind} for {stage.stage_id}")



def dataset_refs_for_stage(stage: StageSpec, opts: OrchestratorOptions) -> list[DatasetRef]:
    refs: list[DatasetRef] = []
    for dataset_id in stage.dataset_ids:
        if dataset_id == "dclm_filter_8k":
            refs.append(
                DatasetRef(
                    dataset_id=dataset_id,
                    path=str(opts.dclm_root),
                    split="train",
                )
            )
        elif dataset_id == "books3":
            refs.append(
                DatasetRef(
                    dataset_id=dataset_id,
                    path=str(opts.books_root),
                    split="train",
                )
            )
        else:
            refs.append(DatasetRef(dataset_id=dataset_id, path="", split="train"))
    return refs



def _dataset_fingerprint_path(ref: DatasetRef) -> Path:
    return Path(ref.path).expanduser().resolve() / f"{ref.split}.fingerprint.json"


def _read_dataset_fingerprint(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid dataset fingerprint payload: {path}")
    dataset = payload.get("dataset", payload)
    if not isinstance(dataset, dict):
        raise ValueError(f"Invalid dataset fingerprint dataset payload: {path}")
    return dataset


def _materialize_dataset_refs(
    refs: list[DatasetRef],
    *,
    required: bool,
) -> list[DatasetRef]:
    resolved: list[DatasetRef] = []
    for ref in refs:
        if not ref.path:
            if required:
                raise ValueError(
                    f"Dataset path missing for dataset_id={ref.dataset_id}; cannot validate fingerprint."
                )
            resolved.append(ref)
            continue

        fp_path = _dataset_fingerprint_path(ref)
        if not fp_path.exists():
            if required:
                raise FileNotFoundError(
                    f"Missing dataset fingerprint: {fp_path}. "
                    "Generate it with scripts/13_dataset_fingerprint.py."
                )
            resolved.append(ref)
            continue

        dataset = _read_dataset_fingerprint(fp_path)
        resolved.append(
            DatasetRef(
                dataset_id=str(dataset.get("dataset_id", ref.dataset_id)),
                path=ref.path,
                split=ref.split,
                tokenizer_id=str(dataset.get("tokenizer_id", ref.tokenizer_id)),
                tokenizer_revision=str(
                    dataset.get("tokenizer_revision", ref.tokenizer_revision)
                ),
                num_tokens=int(dataset.get("num_tokens", ref.num_tokens) or 0),
                sha256=str(dataset.get("sha256", ref.sha256)),
            )
        )

    return resolved


def _validate_parent_hash_consistency(
    *,
    run_dir: Path,
    parent_refs: list[CheckpointRef],
) -> None:
    manifest_path = run_dir / "checkpoint_manifest.json"
    if not manifest_path.exists():
        return

    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, dict):
        return

    existing = payload.get("parent_checkpoints", [])
    if not isinstance(existing, list):
        return

    existing_hashes: dict[str, str] = {}
    for item in existing:
        if not isinstance(item, dict):
            continue
        checkpoint_id = str(item.get("checkpoint_id", "")).strip()
        payload_sha = str(item.get("payload_sha256", "")).strip()
        if checkpoint_id and payload_sha:
            existing_hashes[checkpoint_id] = payload_sha

    for parent in parent_refs:
        previous = existing_hashes.get(parent.checkpoint_id)
        if previous and previous != parent.payload_sha256:
            raise ValueError(
                "Parent checkpoint hash mismatch for "
                f"{parent.checkpoint_id}: previous={previous} current={parent.payload_sha256}"
            )


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def observed_tokens_from_runtime_artifacts(
    *,
    metrics_path: Path,
    events_path: Path,
    fallback: int = 0,
) -> int:
    for row in reversed(_read_jsonl_rows(events_path)):
        if str(row.get("event", "")).strip() != "run_finished":
            continue
        tokens = _positive_int(row.get("tokens_seen"))
        if tokens is not None:
            return tokens

    for row in reversed(_read_jsonl_rows(metrics_path)):
        tokens = _positive_int(row.get("tokens_seen"))
        if tokens is not None:
            return tokens

    return max(int(fallback), 0)


def build_train_command(
    *,
    stage: StageSpec,
    opts: OrchestratorOptions,
    steps: int,
    run_id: str,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "--exact",
        "train",
        f"+deploy={opts.deploy}",
        f"+experiment={stage.experiment}",
        f"training.exp_folder={opts.exp_folder}",
        f"training.exp_dir={opts.exp_dir}",
        f"training.exp_name={run_id}",
        f"training.total_steps={steps}",
        f"training.runtime_mode={opts.runtime_mode}",
        f"training.wandb_entity={opts.wandb_entity}",
        f"training.wandb_project={opts.wandb_project}",
        f"training.wandb_key={opts.wandb_key}",
        f"deploy_paths.data.dclm_filter_8k={opts.dclm_root}",
        f"deploy_paths.data.books3={opts.books_root}",
        f"deploy_paths.checkpoint={opts.checkpoint_root}",
        f"training.checkpoint_path={opts.checkpoint_root}",
        f"training.paper_run_id={opts.paper_run_id}",
        f"training.stage_id={stage.stage_id}",
        f"training.run_id={run_id}",
    ]

    if opts.save_milestone_freq > 0:
        cmd.append(f"training.save_milestone_freq={opts.save_milestone_freq}")
    if stage.kind == "ext" and opts.ext_global_batch_size is not None:
        cmd.append(f"training.global_batch_size={opts.ext_global_batch_size}")
    elif opts.global_batch_size is not None:
        cmd.append(f"training.global_batch_size={opts.global_batch_size}")
    if opts.accum_steps is not None:
        cmd.append(f"training.accum_steps={opts.accum_steps}")
    if opts.seq_length is not None:
        cmd.append(f"training.seq_length={opts.seq_length}")
    if opts.dummy_dataset:
        cmd.append("training.dummy_dataset=true")

    for override in stage.extra_overrides:
        cmd.append(override)

    if stage.required_profile_keys:
        # Use first required profile key as default profile path override.
        profile_path = opts.profile_root / stage.required_profile_keys[0] / "model_profile.json"
        cmd.append(f"training.external_profile_path={profile_path}")

    return cmd



def _checkpoint_for_stage(
    *,
    stage: StageSpec,
    run_id: str,
    opts: OrchestratorOptions,
) -> CheckpointRef:
    return resolve_checkpoint_ref(
        checkpoint_root=opts.checkpoint_root,
        exp_folder=opts.exp_folder,
        checkpoint_id=stage.stage_id,
        exp_name=run_id,
    )



def run_stage(
    *,
    stage: StageSpec,
    stage_map: dict[str, StageSpec],
    opts: OrchestratorOptions,
    budget: BudgetSpec,
    eval_spec: EvalSpec,
    repo_root: Path,
    run_id: str | None = None,
) -> RunResult:
    resolved_run_id = run_id or stage.exp_name
    steps = stage_steps(stage, budget)

    parent_refs = resolve_stage_parents(
        stage=stage,
        stage_map=stage_map,
        checkpoint_root=opts.checkpoint_root,
        exp_folder=opts.exp_folder,
        allow_missing=opts.dry_run,
    )
    validate_stage_profiles(stage=stage, profile_root=opts.profile_root)

    run_dir = ensure_run_dir(
        exp_dir=opts.exp_dir,
        paper_run_id=opts.paper_run_id,
        stage_id=stage.stage_id,
        run_id=resolved_run_id,
    )
    _validate_parent_hash_consistency(run_dir=run_dir, parent_refs=parent_refs)

    command = build_train_command(stage=stage, opts=opts, steps=steps, run_id=resolved_run_id)
    datasets = _materialize_dataset_refs(
        dataset_refs_for_stage(stage, opts),
        required=(opts.require_dataset_fingerprint and not opts.dummy_dataset),
    )

    run_spec = RunSpec(
        run_id=resolved_run_id,
        paper_run_id=opts.paper_run_id,
        stage_id=stage.stage_id,
        model_key=stage.model_key,
        path_group=stage.path_group,
        exp_folder=opts.exp_folder,
        exp_name=resolved_run_id,
        command=command,
        config_overrides=stage.extra_overrides,
        checkpoint_parents=parent_refs,
        datasets=datasets,
        budget=budget,
        eval_spec=eval_spec,
        tags={"kind": stage.kind, "train_mode": stage.train_mode},
    )

    write_stage_manifest(run_dir / "stage_manifest.json", stage, repo_root=repo_root)
    write_run_manifest(run_dir / "run_manifest.json", run_spec, repo_root=repo_root)
    write_json(run_dir / "environment_manifest.json", environment_manifest(repo_root=repo_root))
    write_command_script(run_dir / "command.sh", command)
    append_jsonl(
        run_dir / "events.jsonl",
        {"event": "run_started", "created_at_utc": utc_now_iso(), "command": command},
    )

    started = time.perf_counter()
    started_at = utc_now_iso()

    rc = 0
    if not opts.dry_run:
        rc = subprocess.run(command, check=False).returncode

    wall_seconds = max(time.perf_counter() - started, 0.0)
    finished_at = utc_now_iso()

    if rc != 0:
        result = make_run_result(
            run_id=resolved_run_id,
            stage_id=stage.stage_id,
            status="failed",
            run_dir=str(run_dir),
            metrics_path=str(run_dir / "metrics.jsonl"),
            events_path=str(run_dir / "events.jsonl"),
            checkpoint=CheckpointRef(checkpoint_id=stage.stage_id, exp_name=resolved_run_id),
            wall_seconds=wall_seconds,
            gpu_hours=estimate_gpu_hours_from_wall(wall_seconds=wall_seconds),
            tokens_seen=0,
            error_message=f"command exited with rc={rc}",
            started_at_utc=started_at,
            finished_at_utc=finished_at,
        )
        write_run_result(run_dir / "run_result.json", result)
        append_jsonl(
            run_dir / "events.jsonl",
            {
                "event": "run_failed",
                "created_at_utc": finished_at,
                "return_code": rc,
                "wall_seconds": wall_seconds,
            },
        )
        return result

    if opts.dry_run:
        checkpoint_ref = CheckpointRef(
            checkpoint_id=stage.stage_id,
            exp_folder=opts.exp_folder,
            exp_name=resolved_run_id,
        )
    else:
        checkpoint_ref = _checkpoint_for_stage(
            stage=stage,
            run_id=resolved_run_id,
            opts=opts,
        )
    checkpoint_manifest = build_checkpoint_manifest(
        run_checkpoint=checkpoint_ref,
        parent_checkpoints=parent_refs,
    )
    git = detect_git_state(repo_root)
    checkpoint_manifest = {
        **checkpoint_manifest,
        "created_at_utc": utc_now_iso(),
        "git_commit": git.commit,
        "git_branch": git.branch,
        "git_dirty": git.dirty,
    }
    write_json(run_dir / "checkpoint_manifest.json", checkpoint_manifest)

    seq_len = opts.seq_length or 0
    batch_size = opts.global_batch_size or 0
    tokens_planned = estimate_tokens(seq_length=seq_len, global_batch_size=batch_size, total_steps=steps)
    observed_tokens = observed_tokens_from_runtime_artifacts(
        metrics_path=run_dir / "metrics.jsonl",
        events_path=run_dir / "events.jsonl",
        fallback=tokens_planned,
    )
    gpu_hours = estimate_gpu_hours_from_wall(wall_seconds=wall_seconds)
    budget_manifest = build_budget_manifest(
        budget_spec=budget,
        tokens_planned=tokens_planned,
        gpu_hours_planned=0.0,
        tokens_observed=observed_tokens,
        gpu_hours_observed=gpu_hours,
    )
    budget_manifest = {
        **budget_manifest,
        "created_at_utc": utc_now_iso(),
        "git_commit": git.commit,
        "git_branch": git.branch,
        "git_dirty": git.dirty,
    }
    write_json(run_dir / "budget_manifest.json", budget_manifest)

    result = make_run_result(
        run_id=resolved_run_id,
        stage_id=stage.stage_id,
        status="succeeded" if not opts.dry_run else "dry_run",
        run_dir=str(run_dir),
        metrics_path=str(run_dir / "metrics.jsonl"),
        events_path=str(run_dir / "events.jsonl"),
        checkpoint=checkpoint_ref,
        wall_seconds=wall_seconds,
        gpu_hours=gpu_hours,
        tokens_seen=observed_tokens,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
    )
    write_run_result(run_dir / "run_result.json", result)
    append_jsonl(
        run_dir / "events.jsonl",
        {
            "event": "run_finished",
            "created_at_utc": finished_at,
            "wall_seconds": wall_seconds,
            "status": result.status,
            "tokens_seen": observed_tokens,
        },
    )
    return result



def pretty_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)
