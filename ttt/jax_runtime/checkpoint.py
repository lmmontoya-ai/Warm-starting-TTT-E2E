"""Orbax-native checkpoint manager for the parity JAX runtime."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import orbax.checkpoint as ocp
from orbax.checkpoint import options as ocp_options

from ttt.config import Config, TrainingConfig

if not hasattr(jax.monitoring, "record_scalar"):  # pragma: no cover - env compatibility
    jax.monitoring.record_scalar = lambda *args, **kwargs: None


LATEST_FILENAME = "latest.json"
STEP_METADATA_PREFIX = "step_metadata_"
LEGACY_SIDECAR_PREFIX = "jax_ckpt_step_"


@dataclass
class RestorePayload:
    step: int
    model_weights: Any
    opt_state: Any | None = None
    payload: dict[str, Any] | None = None


def unify_dict_with_eqx_module(d: dict, module, *, allow_shape_mismatch: bool = False):
    from jax._src.lib import pytree

    weights_map = {path: value for path, value in jax.tree.flatten_with_path(d)[0]}
    missed: list[str] = []
    mismatched: list[str] = []

    def find_weight(path, value):
        dict_path = tuple(
            pytree.DictKey(part.name) if isinstance(part, pytree.GetAttrKey) else part
            for part in path
        )
        if dict_path in weights_map:
            new_value = weights_map[dict_path]
            if hasattr(new_value, "shape") and hasattr(value, "shape") and new_value.shape != value.shape:
                if allow_shape_mismatch:
                    mismatched.append(
                        f"{jax.tree_util.keystr(path)}: checkpoint={new_value.shape} target={value.shape}"
                    )
                    return value
                raise ValueError(
                    f"Shape mismatch for {jax.tree_util.keystr(path)}: {new_value.shape} != {value.shape}"
                )
            return new_value
        missed.append(jax.tree_util.keystr(path))
        return value

    return jax.tree.map_with_path(find_weight, module), missed, mismatched


def fetch_from_eqx_module(d: dict, module):
    from jax._src.lib import pytree

    eqx_map = {path: value for path, value in jax.tree.flatten_with_path(module)[0]}
    missed: list[str] = []

    def find_weight(path, value):
        dict_path = tuple(
            pytree.GetAttrKey(part.key) if isinstance(part, pytree.DictKey) else part
            for part in path
        )
        if dict_path in eqx_map:
            new_value = eqx_map[dict_path]
            if hasattr(new_value, "shape") and hasattr(value, "shape") and new_value.shape != value.shape:
                raise ValueError(
                    f"Shape mismatch for {jax.tree_util.keystr(path)}: {new_value.shape} != {value.shape}"
                )
            return new_value
        missed.append(jax.tree_util.keystr(path))
        return value

    return jax.tree.map_with_path(find_weight, d), missed


class LegacyPhase1Loader:
    def __init__(self, checkpoint_dir: Path | str):
        self.checkpoint_dir = Path(checkpoint_dir)

    def _latest_path(self) -> Path:
        return self.checkpoint_dir / LATEST_FILENAME

    def _sidecar_path(self, step: int) -> Path:
        return self.checkpoint_dir / f"{LEGACY_SIDECAR_PREFIX}{step:08d}.json"

    def load(self, step: int | None = None) -> RestorePayload | None:
        latest_path = self._latest_path()
        if step is None:
            if not latest_path.exists():
                return None
            latest = json.loads(latest_path.read_text())
            step = int(latest["step"])
        sidecar_path = self._sidecar_path(step)
        if not sidecar_path.exists():
            return None
        sidecar = json.loads(sidecar_path.read_text())
        state_path = self.checkpoint_dir / sidecar["state_path"]
        bundle = pickle.loads(state_path.read_bytes())
        return RestorePayload(
            step=int(bundle["step"]),
            model_weights=bundle["params"],
            opt_state=bundle.get("opt_state"),
            payload=sidecar,
        )


class OrbaxCheckpointer:
    def __init__(self, checkpoint_dir: Path | str, *, for_saving: bool = True):
        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        if for_saving:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        registry = ocp.DefaultCheckpointHandlerRegistry()
        registry.add("opt_state", ocp.args.StandardRestore, ocp.StandardCheckpointHandler)
        registry.add("opt_state", ocp.args.StandardSave, ocp.StandardCheckpointHandler)
        registry.add("model_weights", ocp.args.StandardRestore, ocp.StandardCheckpointHandler)
        registry.add("model_weights", ocp.args.StandardSave, ocp.StandardCheckpointHandler)
        self.manager = ocp.CheckpointManager(
            self.checkpoint_dir,
            options=ocp.CheckpointManagerOptions(
                multiprocessing_options=ocp_options.MultiprocessingOptions(primary_host=0)
            ),
            handler_registry=registry,
        )

    def latest_step(self) -> int | None:
        return self.manager.latest_step()

    def _latest_path(self) -> Path:
        return self.checkpoint_dir / LATEST_FILENAME

    def _step_metadata_path(self, step: int) -> Path:
        return self.checkpoint_dir / f"{STEP_METADATA_PREFIX}{step:08d}.json"

    def save(
        self,
        *,
        step: int,
        model_weights: Any,
        opt_state: Any | None,
        metrics: dict[str, Any],
        metadata: dict[str, Any],
    ) -> Path:
        args_dict = {"model_weights": ocp.args.StandardSave(model_weights)}
        if opt_state is not None:
            args_dict["opt_state"] = ocp.args.StandardSave(opt_state)
        self.manager.save(step=step, args=ocp.args.Composite(**args_dict))
        self.manager.wait_until_finished()

        step_meta = {
            "step": int(step),
            "checkpoint_format": "orbax",
            "checkpoint_dir": str(self.checkpoint_dir / str(step)),
            "metrics": metrics,
            "metadata": metadata,
            "items": sorted(args_dict.keys()),
        }
        step_meta_path = self._step_metadata_path(step)
        step_meta_path.write_text(json.dumps(step_meta, indent=2, sort_keys=True) + "\n")
        latest = {"step": int(step), "path": str(step), "metadata_path": step_meta_path.name}
        self._latest_path().write_text(json.dumps(latest, indent=2, sort_keys=True) + "\n")
        return step_meta_path

    def load(
        self,
        *,
        step: int | None,
        targets: dict[str, Any],
        restore: TrainingConfig.LoadPart,
    ) -> RestorePayload:
        use_step = self.manager.latest_step() if step is None else int(step)
        if use_step is None:
            raise FileNotFoundError(f"No checkpoints found at {self.checkpoint_dir}")

        item_metadata = self.manager.item_metadata(use_step)
        skipped_mismatched: list[str] = []
        if restore == TrainingConfig.LoadPart.params:
            # Prefer a target-aware restore when the checkpoint and target topology
            # match. This preserves Orbax's native array/sharding reconstruction for
            # same-architecture eval/resume paths. Only fall back to the permissive
            # raw-tree restore when shape mismatches make that impossible, which is
            # required for FA -> SWA/E2E warm-starts.
            try:
                model_target = fetch_from_eqx_module(item_metadata["model_weights"], targets["model_weights"])[0]
                restored = self.manager.restore(
                    use_step,
                    args=ocp.args.Composite(model_weights=ocp.args.StandardRestore(model_target)),
                )
                model_weights = restored["model_weights"]
            except Exception:
                restored = self.manager.restore(
                    use_step,
                    args=ocp.args.Composite(model_weights=ocp.args.StandardRestore(strict=False)),
                )
                model_weights, _, skipped_mismatched = unify_dict_with_eqx_module(
                    restored["model_weights"],
                    targets["model_weights"],
                    allow_shape_mismatch=True,
                )
            opt_state = None
        else:
            model_target = fetch_from_eqx_module(item_metadata["model_weights"], targets["model_weights"])[0]
            args_dict = {"model_weights": ocp.args.StandardRestore(model_target)}
            opt_target = None
            if restore == TrainingConfig.LoadPart.all and "opt_state" in targets and "opt_state" in item_metadata:
                opt_target = fetch_from_eqx_module(item_metadata["opt_state"], targets["opt_state"])[0]
                args_dict["opt_state"] = ocp.args.StandardRestore(opt_target)
            restored = self.manager.restore(use_step, args=ocp.args.Composite(**args_dict))
            model_weights = restored["model_weights"]
            opt_state = restored.get("opt_state")
        meta_path = self._step_metadata_path(use_step)
        payload = json.loads(meta_path.read_text()) if meta_path.exists() else None
        if payload is not None and skipped_mismatched:
            payload = dict(payload)
            payload["skipped_mismatched_params"] = skipped_mismatched
        return RestorePayload(
            step=int(use_step),
            model_weights=model_weights,
            opt_state=opt_state,
            payload=payload,
        )

    def close(self) -> None:
        self.manager.close()


def resolve_resume_checkpoint_dir(cfg: Config, *, current_checkpoint_dir: Path) -> Path | None:
    explicit = str(cfg.training.resume_checkpoint_path).strip()
    if explicit:
        raw = Path(explicit).expanduser().resolve()
        manifest = raw / "artifact_manifest.json"
        if manifest.exists():
            payload = json.loads(manifest.read_text())
            manifest_step = int(payload.get("step", 0) or 0)
            local_raw_step_dir = str(payload.get("local_raw_step_dir", "")).strip()
            if local_raw_step_dir:
                candidate = Path(local_raw_step_dir).expanduser().resolve()
                if candidate.exists():
                    return candidate
            fallback = raw / "raw_orbax" / str(manifest_step)
            if fallback.exists():
                return fallback.resolve()
            raise FileNotFoundError(
                "Artifact manifest exists but neither local_raw_step_dir nor "
                f"raw_orbax/{manifest_step} is available under {raw}"
            )
        if raw.name.isdigit():
            return raw
        latest_json = raw / LATEST_FILENAME
        if latest_json.exists():
            latest = json.loads(latest_json.read_text())
            return raw / str(latest["step"])
        if any(child.name.isdigit() for child in raw.iterdir() if child.is_dir()):
            steps = sorted(int(child.name) for child in raw.iterdir() if child.is_dir() and child.name.isdigit())
            return raw / str(steps[-1])
        return raw

    resume_name = str(cfg.training.resume_exp_name).strip()
    if not resume_name:
        return None
    candidate_root = current_checkpoint_dir.parent / resume_name
    latest_json = candidate_root / LATEST_FILENAME
    if latest_json.exists():
        latest = json.loads(latest_json.read_text())
        return candidate_root / str(latest["step"])
    if candidate_root.is_dir() and candidate_root.name.isdigit():
        return candidate_root
    return candidate_root


def resolve_restore_payload(
    *,
    cfg: Config,
    current_checkpoint_dir: Path,
    targets: dict[str, Any],
) -> RestorePayload | None:
    resume_dir = resolve_resume_checkpoint_dir(cfg, current_checkpoint_dir=current_checkpoint_dir)
    if resume_dir is None:
        return None
    restore_format = str(cfg.training.resume_checkpoint_format or "orbax")
    if restore_format == "phase1_json":
        legacy = LegacyPhase1Loader(resume_dir if resume_dir.is_dir() else resume_dir.parent)
        return legacy.load(step=cfg.training.resume_step)

    orbax_root = resume_dir.parent if resume_dir.name.isdigit() else resume_dir
    loader = OrbaxCheckpointer(orbax_root, for_saving=False)
    restore_mode = cfg.training.load_part
    if str(restore_mode) == "none" or ("opt_state" not in targets):
        restore_mode = TrainingConfig.LoadPart.params
    return loader.load(
        step=int(resume_dir.name) if resume_dir.name.isdigit() else cfg.training.resume_step,
        targets=targets,
        restore=restore_mode,
    )
