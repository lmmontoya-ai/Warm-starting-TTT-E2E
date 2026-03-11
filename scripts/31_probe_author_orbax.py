#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import orbax.checkpoint as ocp

from ttt.research.author_checkpoints import (
    AUTHOR_CHECKPOINTS,
    artifact_root,
    load_env_file,
    load_json,
    manifest_path,
    raw_step_dir,
    select_specs,
    write_json,
)


def _leaf_stats(tree: Any) -> dict[str, Any]:
    stats = {
        "leaf_count": 0,
        "param_count": 0,
        "tensor_count": 0,
        "sample_leaves": [],
    }

    def visit(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                visit(value, path + (str(key),))
            return
        if isinstance(node, (list, tuple)):
            for idx, value in enumerate(node):
                visit(value, path + (str(idx),))
            return

        shape = getattr(node, "shape", None)
        dtype = getattr(node, "dtype", None)
        size = getattr(node, "size", None)
        stats["leaf_count"] += 1
        if shape is not None and dtype is not None:
            stats["tensor_count"] += 1
            try:
                stats["param_count"] += int(size)
            except Exception:
                pass
            if len(stats["sample_leaves"]) < 12:
                stats["sample_leaves"].append(
                    {
                        "path": "/".join(path),
                        "shape": [int(x) for x in shape],
                        "dtype": str(dtype),
                    }
                )

    visit(tree, tuple())
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe locally fetched author Orbax checkpoints. This verifies raw Orbax "
            "readability and writes a probe report without claiming local runtime parity."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default="all",
        help="Comma-separated checkpoint keys or 'all'. Supported: 760m_fa,760m_e2e",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("./artifacts/author_checkpoints"),
    )
    parser.add_argument(
        "--verify-restore",
        action="store_true",
        help="Fully restore model_weights and compute leaf/tensor stats.",
    )
    return parser.parse_args()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    load_env_file(repo_root / ".env")
    args = parse_args()

    artifact_root_path = args.artifact_root.expanduser().resolve()
    if not hasattr(jax.monitoring, "record_scalar"):
        jax.monitoring.record_scalar = lambda *a, **kw: None
    checkpointer = ocp.StandardCheckpointer()

    for spec in select_specs(args.checkpoint):
        manifest_file = manifest_path(artifact_root_path, spec)
        if not manifest_file.exists():
            raise FileNotFoundError(f"Missing artifact manifest: {manifest_file}")
        manifest = load_json(manifest_file)
        step = int(manifest["step"])
        step_dir = raw_step_dir(artifact_root_path, spec, step)
        if not step_dir.exists():
            raise FileNotFoundError(f"Missing raw step dir: {step_dir}")

        root_meta_path = step_dir / "_CHECKPOINT_METADATA"
        item_dir = step_dir / "model_weights"
        if not item_dir.exists():
            raise FileNotFoundError(f"Missing model_weights item dir: {item_dir}")

        item_metadata = checkpointer.metadata(item_dir)
        probe_report: dict[str, Any] = {
            "schema_version": "1.0",
            "checkpoint_key": spec.key,
            "description": AUTHOR_CHECKPOINTS[spec.key].description,
            "step": step,
            "step_dir": str(step_dir),
            "root_checkpoint_metadata_path": str(root_meta_path),
            "root_checkpoint_metadata_json": json.loads(root_meta_path.read_text(encoding="utf-8")),
            "orbax_item": "model_weights",
            "item_metadata_repr": repr(item_metadata),
            "verify_restore": bool(args.verify_restore),
            "restore_status": "not_run",
            "local_runtime_compatibility": {
                "status": "not_verified_for_local_jax_runtime",
                "reason": (
                    "This probe validates raw Orbax readability only. The current local "
                    "jax_runtime model/checkpoint format is still not architecture-parity "
                    "with the author checkpoint tree."
                ),
            },
        }

        if args.verify_restore:
            restored = checkpointer.restore(item_dir)
            probe_report["restore_status"] = "ok"
            probe_report["restored_tree_type"] = type(restored).__name__
            probe_report["restore_stats"] = _leaf_stats(restored)

        report_path = artifact_root(artifact_root_path, spec) / "probe_report.json"
        write_json(report_path, probe_report)
        print(f"Wrote probe report: {report_path}")
        if args.verify_restore:
            stats = probe_report["restore_stats"]
            print(
                "  "
                f"leaf_count={stats['leaf_count']} tensor_count={stats['tensor_count']} "
                f"param_count={stats['param_count']}"
            )

    checkpointer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
