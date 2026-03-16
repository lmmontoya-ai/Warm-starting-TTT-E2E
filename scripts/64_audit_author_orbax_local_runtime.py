#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import orbax.checkpoint as ocp
from hydra import compose, initialize_config_dir

from ttt.config import register_configs
from ttt.jax_runtime.checkpoint import fetch_from_eqx_module, unify_dict_with_eqx_module
from ttt.jax_runtime.model.transformer import MetaModel
from ttt.research.author_checkpoints import load_env_file, manifest_path, load_json, select_specs
from ttt.research.types import utc_now_iso


@dataclass(frozen=True)
class AuditTarget:
    target_id: str
    checkpoint_key: str
    experiment: str


AUDIT_TARGETS: tuple[AuditTarget, ...] = (
    AuditTarget(
        target_id="S0",
        checkpoint_key="760m_fa",
        experiment="760m/extension/ext-760m-fa-32K",
    ),
    AuditTarget(
        target_id="S1",
        checkpoint_key="760m_fa",
        experiment="760m/pretrained/ext-760m-swa-32K-from-fa",
    ),
    AuditTarget(
        target_id="S2_ADAPT",
        checkpoint_key="760m_fa",
        experiment="760m/pretrained/adapt-760m-e2e-8K-from-fa",
    ),
    AuditTarget(
        target_id="S3",
        checkpoint_key="760m_e2e",
        experiment="760m/extension/ext-760m-e2e-32K",
    ),
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compose_cfg(repo_root: Path, experiment: str):
    configs_dir = repo_root / "configs"
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=str(configs_dir)):
        cfg = compose(config_name="config", overrides=["+deploy=interactive", f"+experiment={experiment}"])
    cfg.training.runtime_mode = "jax_train"
    return cfg


def _checkpoint_item_dir(artifact_root: Path, checkpoint_key: str) -> Path:
    spec = next(spec for spec in select_specs(checkpoint_key) if spec.key == checkpoint_key)
    manifest = load_json(manifest_path(artifact_root, spec))
    local_raw_step_dir = str(manifest.get("local_raw_step_dir", "")).strip()
    if local_raw_step_dir:
        candidate = Path(local_raw_step_dir).expanduser().resolve()
        if candidate.exists():
            return candidate / "model_weights"
    step = int(manifest["step"])
    fallback = artifact_root / checkpoint_key / "raw_orbax" / str(step) / "model_weights"
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(
        f"Could not resolve raw Orbax item dir for {checkpoint_key} under {artifact_root}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether the author-shared 760M Orbax checkpoints can be restored into "
            "the local runtime target configs used by the paper smoke gates."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, default=Path("./artifacts/author_checkpoints"))
    parser.add_argument(
        "--checkpoint",
        default="all",
        help="Comma-separated checkpoint keys or 'all'. Supported: 760m_fa,760m_e2e",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("./artifacts/author_checkpoints/760m_local_runtime_audit.json"),
    )
    return parser.parse_args()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    load_env_file(repo_root / ".env")
    args = parse_args()

    if not hasattr(jax.monitoring, "record_scalar"):
        jax.monitoring.record_scalar = lambda *a, **kw: None

    requested = {spec.key for spec in select_specs(args.checkpoint)}
    artifact_root = args.artifact_root.expanduser().resolve()
    report_out = args.report_out.expanduser().resolve()

    checkpointer = ocp.StandardCheckpointer()
    rows: list[dict[str, Any]] = []

    for target in AUDIT_TARGETS:
        if target.checkpoint_key not in requested:
            continue

        cfg = _compose_cfg(args.repo_root.expanduser().resolve(), target.experiment)
        model, _ = eqx.nn.make_with_state(MetaModel)(cfg, key=jax.random.PRNGKey(0))
        item_dir = _checkpoint_item_dir(artifact_root, target.checkpoint_key)
        item_metadata = checkpointer.metadata(item_dir)

        row: dict[str, Any] = {
            "target_id": target.target_id,
            "checkpoint_key": target.checkpoint_key,
            "experiment": target.experiment,
            "item_dir": str(item_dir),
            "status": "unknown",
            "compatibility_mode": "unknown",
            "missed_count": 0,
            "mismatched_count": 0,
            "sample_missed": [],
            "sample_mismatched": [],
        }

        try:
            fetch_from_eqx_module(item_metadata, model.weights())
            row["status"] = "ok"
            row["compatibility_mode"] = "exact_target_restore"
        except Exception as exc:
            restored = checkpointer.restore(item_dir)
            _, missed, mismatched = unify_dict_with_eqx_module(
                restored,
                model.weights(),
                allow_shape_mismatch=True,
            )
            row["status"] = "ok"
            row["compatibility_mode"] = "fallback_partial_warmstart"
            row["exact_restore_error"] = f"{type(exc).__name__}: {exc}"
            row["missed_count"] = len(missed)
            row["mismatched_count"] = len(mismatched)
            row["sample_missed"] = missed[:16]
            row["sample_mismatched"] = mismatched[:16]

        rows.append(row)

    payload = {
        "schema_version": "1.0",
        "created_at_utc": utc_now_iso(),
        "requested_checkpoints": sorted(requested),
        "rows": rows,
    }
    _write_json(report_out, payload)
    print(f"Wrote author local-runtime audit: {report_out}")
    for row in rows:
        print(
            f"{row['target_id']}: {row['compatibility_mode']} "
            f"(missed={row['missed_count']} mismatched={row['mismatched_count']})"
        )

    checkpointer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
