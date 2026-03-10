"""Checkpoint manager for native JAX runtime.

The external lineage validator expects a JSON sidecar referenced by
`latest.json`. This module keeps that contract and stores the full model state
in a separate binary payload file.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax


LATEST_FILENAME = "latest.json"
SIDECAR_PREFIX = "jax_ckpt_step_"


@dataclass(frozen=True)
class JaxCheckpointState:
    step: int
    params: Any
    opt_state: Any
    payload: dict[str, Any]


class JaxCheckpointer:
    def __init__(self, checkpoint_dir: Path | str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _latest_path(self) -> Path:
        return self.checkpoint_dir / LATEST_FILENAME

    def _sidecar_path(self, step: int) -> Path:
        return self.checkpoint_dir / f"{SIDECAR_PREFIX}{step:08d}.json"

    def _state_path(self, step: int) -> Path:
        return self.checkpoint_dir / f"jax_state_step_{step:08d}.pkl"

    def latest_step(self) -> int | None:
        latest = self._latest_path()
        if not latest.exists():
            return None
        try:
            payload = json.loads(latest.read_text())
        except json.JSONDecodeError:
            return None
        step = payload.get("step")
        return int(step) if isinstance(step, int) else None

    def save(
        self,
        *,
        step: int,
        params: Any,
        opt_state: Any,
        metrics: dict[str, Any],
        metadata: dict[str, Any],
    ) -> Path:
        sidecar_path = self._sidecar_path(step)
        state_path = self._state_path(step)

        bundle = {
            "step": int(step),
            "params": jax.device_get(params),
            "opt_state": jax.device_get(opt_state),
        }
        with state_path.open("wb") as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

        sidecar = {
            "step": int(step),
            "checkpoint_format": "pickle_pytree",
            "state_path": state_path.name,
            "metrics": metrics,
            "metadata": metadata,
        }
        sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")

        latest = {
            "step": int(step),
            "path": sidecar_path.name,
        }
        self._latest_path().write_text(json.dumps(latest, indent=2, sort_keys=True) + "\n")

        return sidecar_path

    def load(self, step: int | None = None) -> JaxCheckpointState | None:
        if step is None:
            step = self.latest_step()
        if step is None:
            return None

        sidecar_path = self._sidecar_path(step)
        if not sidecar_path.exists():
            # Fall back to latest pointer if step file naming changed.
            latest_path = self._latest_path()
            if not latest_path.exists():
                return None
            latest = json.loads(latest_path.read_text())
            rel_path = latest.get("path")
            if not isinstance(rel_path, str):
                return None
            sidecar_path = self.checkpoint_dir / rel_path
            if not sidecar_path.exists():
                return None

        try:
            sidecar = json.loads(sidecar_path.read_text())
        except json.JSONDecodeError:
            return None
        if not isinstance(sidecar, dict):
            return None

        state_rel = sidecar.get("state_path")
        if not isinstance(state_rel, str):
            return None
        state_path = self.checkpoint_dir / state_rel
        if not state_path.exists():
            return None

        with state_path.open("rb") as f:
            bundle = pickle.load(f)
        if not isinstance(bundle, dict):
            return None

        loaded_step = int(bundle.get("step", step))
        return JaxCheckpointState(
            step=loaded_step,
            params=bundle.get("params"),
            opt_state=bundle.get("opt_state"),
            payload=sidecar,
        )
