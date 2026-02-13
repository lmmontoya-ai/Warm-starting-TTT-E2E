#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_model_id: str
    revision: str
    config_sha256: str

    @property
    def config_url(self) -> str:
        return f"https://huggingface.co/{self.hf_model_id}/resolve/{self.revision}/config.json"


MODEL_SPECS: dict[str, ModelSpec] = {
    "qwen2_5_0_5b": ModelSpec(
        key="qwen2_5_0_5b",
        hf_model_id="Qwen/Qwen2.5-0.5B",
        revision="060db6499f32faf8b98477b0a26969ef7d8b9987",
        config_sha256="479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b",
    ),
    "smollm2_360m": ModelSpec(
        key="smollm2_360m",
        hf_model_id="HuggingFaceTB/SmolLM2-360M",
        revision="f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
        config_sha256="34f7801487078de7e434e19162c497e5cc6ff397080e40e8586627cb68a5168a",
    ),
}


def _download_json(url: str) -> tuple[dict[str, Any], str]:
    req = urllib.request.Request(url, headers={"User-Agent": "codex-phase1-external-setup/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        raw = response.read()
    payload_sha256 = hashlib.sha256(raw).hexdigest()
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object from {url}, got {type(parsed)}")
    return parsed, payload_sha256


def _build_profile(spec: ModelSpec, cfg: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    def get_int(name: str, default: int = 0) -> int:
        val = cfg.get(name, default)
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    def get_float(name: str, default: float = 0.0) -> float:
        val = cfg.get(name, default)
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    profile = {
        "profile_version": 1,
        "model_key": spec.key,
        "hf_model_id": spec.hf_model_id,
        "source_revision": spec.revision,
        "source_config_sha256": config_sha256,
        "model_type": str(cfg.get("model_type", "")),
        "architectures": cfg.get("architectures", []),
        "vocab_size": get_int("vocab_size"),
        "num_hidden_layers": get_int("num_hidden_layers"),
        "hidden_size": get_int("hidden_size"),
        "intermediate_size": get_int("intermediate_size"),
        "num_attention_heads": get_int("num_attention_heads"),
        "num_key_value_heads": get_int("num_key_value_heads"),
        "max_position_embeddings": get_int("max_position_embeddings"),
        "rope_theta": get_float("rope_theta", 10000.0),
        "rms_norm_eps": get_float("rms_norm_eps", 1e-6),
        "bos_token_id": get_int("bos_token_id"),
        "eos_token_id": get_int("eos_token_id"),
        "tie_word_embeddings": bool(cfg.get("tie_word_embeddings", False)),
        "use_sliding_window": bool(cfg.get("use_sliding_window", False)),
        "sliding_window": get_int("sliding_window"),
    }
    return profile


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _prepare_one(spec: ModelSpec, out_root: Path, force: bool) -> tuple[Path, Path]:
    model_dir = out_root / spec.key
    cfg_path = model_dir / "hf_config.json"
    profile_path = model_dir / "model_profile.json"

    if profile_path.exists() and cfg_path.exists() and not force:
        return cfg_path, profile_path

    cfg, observed_sha = _download_json(spec.config_url)
    if observed_sha != spec.config_sha256:
        raise ValueError(
            f"Config digest mismatch for {spec.hf_model_id} @ {spec.revision}: "
            f"expected {spec.config_sha256}, got {observed_sha}"
        )
    profile = _build_profile(spec, cfg, config_sha256=observed_sha)

    _write_json(cfg_path, cfg)
    _write_json(profile_path, profile)

    return cfg_path, profile_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch external model configs and build local profile files for Qwen/Smol experiments."
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", *MODEL_SPECS.keys()],
        help="Which model profile(s) to prepare.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("./artifacts/external_models"),
        help="Destination root for downloaded configs and generated profiles.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download and overwrite existing files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = args.out_root.expanduser().resolve()

    if args.model == "all":
        selected = list(MODEL_SPECS.values())
    else:
        selected = [MODEL_SPECS[args.model]]

    for spec in selected:
        try:
            cfg_path, profile_path = _prepare_one(spec, out_root=out_root, force=args.force)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            print(f"Failed to fetch {spec.hf_model_id}: {exc}")
            return 1

        profile = json.loads(profile_path.read_text())
        print(
            f"Prepared {spec.hf_model_id} -> {profile_path} "
            f"(layers={profile.get('num_hidden_layers')} hidden={profile.get('hidden_size')} "
            f"heads={profile.get('num_attention_heads')} vocab={profile.get('vocab_size')})"
        )
        print(f"  revision: {spec.revision}")
        print(f"  config: {cfg_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
