"""Phase-1 checkpoint management.

The phase-1 runtime stores lightweight JSON checkpoints that contain loop
position and minimal metadata. This supports warm-start orchestration and
resumption semantics while model internals are still being ported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LATEST_FILENAME = "latest.json"
STEP_PREFIX = "phase1_ckpt_step_"


@dataclass(frozen=True)
class CheckpointState:
    step: int
    payload: dict[str, Any]


class Phase1Checkpointer:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _step_path(self, step: int) -> Path:
        return self.checkpoint_dir / f"{STEP_PREFIX}{step:08d}.json"

    def _latest_path(self) -> Path:
        return self.checkpoint_dir / LATEST_FILENAME

    def latest_step(self) -> int | None:
        latest_path = self._latest_path()
        if not latest_path.exists():
            return None

        try:
            data = json.loads(latest_path.read_text())
        except json.JSONDecodeError:
            return None

        step = data.get("step")
        return int(step) if isinstance(step, int) else None

    def save(self, step: int, payload: dict[str, Any]) -> Path:
        ckpt_path = self._step_path(step)
        ckpt_record = {"step": step, **payload}
        ckpt_path.write_text(json.dumps(ckpt_record, indent=2, sort_keys=True))

        latest_record = {"step": step, "path": ckpt_path.name}
        self._latest_path().write_text(json.dumps(latest_record, indent=2, sort_keys=True))

        return ckpt_path

    def load(self, step: int | None = None) -> CheckpointState | None:
        if step is None:
            step = self.latest_step()

        if step is None:
            return None

        ckpt_path = self._step_path(step)
        if not ckpt_path.exists():
            return None

        try:
            data = json.loads(ckpt_path.read_text())
        except json.JSONDecodeError:
            return None

        if "step" not in data:
            return None

        ckpt_step = int(data.pop("step"))
        return CheckpointState(step=ckpt_step, payload=data)
