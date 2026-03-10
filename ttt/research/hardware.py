"""Hardware probing helpers for run manifests and budget accounting."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareSnapshot:
    num_visible_gpus: int
    gpu_descriptions: list[str]



def count_visible_gpus() -> int:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        # Best effort fallback to nvidia-smi count
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return 0
        if completed.returncode != 0:
            return 0
        lines = [x for x in completed.stdout.splitlines() if x.strip()]
        return len(lines)

    entries = [x.strip() for x in raw.split(",") if x.strip()]
    return len(entries)



def probe_gpu_descriptions() -> list[str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []
    if completed.returncode != 0:
        return []
    return [x.strip() for x in completed.stdout.splitlines() if x.strip()]



def detect_hardware() -> HardwareSnapshot:
    return HardwareSnapshot(
        num_visible_gpus=count_visible_gpus(),
        gpu_descriptions=probe_gpu_descriptions(),
    )



def parse_memory_gb(gpu_description: str) -> float:
    # expected "H100, 81920 MiB"
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*MiB", gpu_description)
    if not match:
        return 0.0
    return float(match.group(1)) / 1024.0
