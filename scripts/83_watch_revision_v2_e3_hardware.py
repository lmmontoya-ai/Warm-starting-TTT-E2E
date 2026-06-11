#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PREFERRED_GPU_TYPES = ("H200_141GB", "H100_80GB")
CONDITIONAL_GPU_TYPES = ("A100_80GB",)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _run_prime_availability(gpu_type: str) -> dict[str, Any]:
    proc = subprocess.run(
        [
            "prime",
            "--plain",
            "availability",
            "list",
            "--gpu-type",
            gpu_type,
            "--gpu-count",
            "8",
            "--no-group-similar",
            "--output",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object for {gpu_type} availability.")
    return payload


def _gpu_type_id(row: dict[str, Any]) -> str:
    raw = str(row.get("gpu_type", "")).upper().replace(" ", "_")
    if raw.startswith("H200"):
        return "H200_141GB"
    if raw.startswith("H100"):
        return "H100_80GB"
    if raw.startswith("A100"):
        return "A100_80GB"
    return raw


def _candidate_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("gpu_resources", [])
    if not isinstance(rows, list):
        raise ValueError("Expected gpu_resources to be a list.")

    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if int(row.get("gpu_count", 0) or 0) != 8:
            continue
        gpu_id = _gpu_type_id(row)
        if gpu_id not in {*PREFERRED_GPU_TYPES, *CONDITIONAL_GPU_TYPES}:
            continue
        if gpu_id == "A100_80GB" and str(row.get("socket", "")).upper() != "SXM4":
            continue
        enriched = dict(row)
        enriched["revision_v2_candidate_class"] = (
            "preferred" if gpu_id in PREFERRED_GPU_TYPES else "conditional-smoke-required"
        )
        enriched["revision_v2_gpu_type_id"] = gpu_id
        candidates.append(enriched)

    candidates.sort(
        key=lambda item: (
            0 if item["revision_v2_candidate_class"] == "preferred" else 1,
            float(item.get("price_value", 0.0) or 0.0),
        )
    )
    return candidates


def _append_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _notify(title: str, message: str) -> None:
    if shutil.which("osascript") is None:
        return
    script = f'display notification "{message}" with title "{title}"'
    subprocess.run(["osascript", "-e", script], check=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Poll Prime for revision-v2 E3-suitable 8-GPU hardware. Preferred: "
            "8x H200 141GB or 8x H100 80GB. Conditional: 8x A100 80GB SXM4 "
            "after a fresh timing smoke. B300 is intentionally excluded under "
            "the pinned JAX runtime."
        )
    )
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("/tmp/revision_v2_e3_prime_hardware_watch.jsonl"),
    )
    parser.add_argument("--notify-title", default="Revision V2 E3 Hardware Available")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpu_types = (*PREFERRED_GPU_TYPES, *CONDITIONAL_GPU_TYPES)
    print(
        f"[{utc_now()}] watching {', '.join(gpu_types)} "
        f"for {args.iterations} iterations every {args.interval_seconds}s",
        flush=True,
    )
    for index in range(args.iterations):
        stamp = utc_now()
        try:
            all_candidates: list[dict[str, Any]] = []
            raw_counts: dict[str, int] = {}
            for gpu_type in gpu_types:
                payload = _run_prime_availability(gpu_type)
                rows = payload.get("gpu_resources", [])
                raw_counts[gpu_type] = len(rows) if isinstance(rows, list) else 0
                all_candidates.extend(_candidate_rows(payload))

            log_row = {
                "timestamp_utc": stamp,
                "iteration": index + 1,
                "raw_counts": raw_counts,
                "match_count": len(all_candidates),
                "matches": all_candidates,
            }
            _append_log(args.log_path, log_row)

            if all_candidates:
                best = all_candidates[0]
                price = best.get("price_per_hour", best.get("price_value"))
                message = (
                    f"{best.get('revision_v2_gpu_type_id')} "
                    f"{best.get('socket')} {price} "
                    f"at {best.get('provider')} {best.get('location')} "
                    f"({best.get('revision_v2_candidate_class')})"
                )
                print(f"[{stamp}] FOUND {len(all_candidates)} candidate(s): {message}")
                _notify(args.notify_title, message)
                return 0

            print(f"[{stamp}] no acceptable revision-v2 E3 hardware found", flush=True)
        except Exception as exc:  # pragma: no cover - operational path
            _append_log(
                args.log_path,
                {"timestamp_utc": stamp, "iteration": index + 1, "error": str(exc)},
            )
            print(f"[{stamp}] watch error: {exc}", file=sys.stderr, flush=True)

        if index + 1 < args.iterations:
            time.sleep(max(1, args.interval_seconds))

    print(f"[{utc_now()}] watch completed with no matches", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

