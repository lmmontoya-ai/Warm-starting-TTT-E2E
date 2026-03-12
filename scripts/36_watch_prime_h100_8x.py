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


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_payload() -> dict[str, Any]:
    proc = subprocess.run(
        ["prime", "availability", "list", "--no-group-similar", "--output", "json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    if not isinstance(payload, dict):
        raise ValueError("Expected Prime availability payload to be a JSON object.")
    return payload


def _matching_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("gpu_resources", [])
    if not isinstance(rows, list):
        raise ValueError("Expected gpu_resources to be a JSON list.")
    matches: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        gpu_type = str(row.get("gpu_type", ""))
        gpu_count = int(row.get("gpu_count", 0) or 0)
        is_spot = bool(row.get("is_spot", False))
        if "H100" not in gpu_type or gpu_count != 8 or is_spot:
            continue
        matches.append(row)
    matches.sort(key=lambda item: float(item.get("price_value", 0.0) or 0.0))
    return matches


def _notify(title: str, message: str) -> None:
    if shutil.which("osascript") is None:
        return
    script = f'display notification "{message}" with title "{title}"'
    subprocess.run(["osascript", "-e", script], check=False)


def _append_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Poll Prime for non-spot 8x H100 inventory and notify when available."
    )
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("/tmp/prime_h100_8x_watch.jsonl"),
    )
    parser.add_argument(
        "--notify-title",
        default="Prime 8x H100 Available",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(
        f"[{utc_now()}] starting watch: iterations={args.iterations} interval={args.interval_seconds}s",
        flush=True,
    )
    for index in range(args.iterations):
        stamp = utc_now()
        try:
            payload = _load_payload()
            matches = _matching_rows(payload)
            log_row = {
                "timestamp_utc": stamp,
                "iteration": index + 1,
                "matches": matches,
                "match_count": len(matches),
            }
            _append_log(args.log_path, log_row)
            if matches:
                best = matches[0]
                price = best.get("price_per_hour", best.get("price_value"))
                provider = best.get("provider", "unknown")
                location = best.get("location", "unknown")
                message = f"{price} at {provider} {location}"
                print(f"[{stamp}] FOUND {len(matches)} match(es): {message}", flush=True)
                _notify(args.notify_title, message)
                return 0
            print(f"[{stamp}] no non-spot 8x H100 rows found", flush=True)
        except Exception as exc:  # pragma: no cover - operational path
            error_row = {
                "timestamp_utc": stamp,
                "iteration": index + 1,
                "error": str(exc),
            }
            _append_log(args.log_path, error_row)
            print(f"[{stamp}] watch error: {exc}", file=sys.stderr, flush=True)

        if index + 1 < args.iterations:
            time.sleep(max(1, args.interval_seconds))

    print(f"[{utc_now()}] watch completed with no matches", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
