#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from ttt.config import register_configs
from ttt.infra import Phase1Checkpointer
from ttt.research.types import utc_now_iso


FIELD_MAP: dict[str, str] = {
    "vocab_size": "vocab_size",
    "num_hidden_layers": "num_hidden_layers",
    "hidden_size": "hidden_size",
    "num_attention_heads": "num_attention_heads",
    "num_key_value_heads": "num_key_value_heads",
    "intermediate_size": "intermediate_size",
    "bos_token_id": "bos_token_id",
    "eos_token_id": "eos_token_id",
    "rope_theta": "rope_theta",
    "rms_norm_eps": "rms_norm_eps",
    "tie_word_embeddings": "tie_word_embeddings",
}


KNOWN_PROFILE_KEYS = {
    "profile_version",
    "model_key",
    "hf_model_id",
    "source_revision",
    "source_config_sha256",
    "model_type",
    "architectures",
    "vocab_size",
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
    "rope_theta",
    "rms_norm_eps",
    "bos_token_id",
    "eos_token_id",
    "tie_word_embeddings",
    "use_sliding_window",
    "sliding_window",
}


def _as_dict(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _load_profile(profile_root: Path, model_key: str) -> tuple[Path, dict[str, Any]]:
    profile_path = profile_root / model_key / "model_profile.json"
    if not profile_path.exists():
        raise FileNotFoundError(
            f"Missing profile for {model_key}: {profile_path}. "
            "Run scripts/07_prepare_external_models.py first."
        )
    return profile_path, _as_dict(profile_path)


def _compose_model_cfg(configs_dir: Path, deploy: str, experiment: str) -> dict[str, Any]:
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=str(configs_dir)):
        cfg = compose(
            config_name="config",
            overrides=[f"+deploy={deploy}", f"+experiment={experiment}"],
        )
    payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("Composed Hydra config is not a mapping")
    model_cfg = payload.get("model", {})
    if not isinstance(model_cfg, dict):
        raise ValueError("Composed Hydra model config is not a mapping")
    return model_cfg


def _load_model_cfg(model_cfg_path: Path) -> dict[str, Any]:
    cfg = OmegaConf.to_container(OmegaConf.load(model_cfg_path), resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError(f"Model config must be a mapping: {model_cfg_path}")
    return cfg


def _load_checkpoint_payload(
    *,
    checkpoint_root: Path,
    exp_folder: str,
    exp_name: str,
) -> tuple[int, dict[str, Any]]:
    checkpointer = Phase1Checkpointer(checkpoint_root / exp_folder / exp_name)
    restored = checkpointer.load(step=None)
    if restored is None:
        raise FileNotFoundError(
            f"Missing checkpoint for {exp_folder}/{exp_name} under {checkpoint_root}"
        )
    return restored.step, restored.payload


def _numeric_equal(a: Any, b: Any, tol: float = 1e-9) -> bool:
    try:
        af = float(a)
        bf = float(b)
        return abs(af - bf) <= tol
    except (TypeError, ValueError):
        return a == b


def _compat_report(profile: dict[str, Any], model_cfg: dict[str, Any]) -> dict[str, Any]:
    field_rows: list[dict[str, Any]] = []
    matched = 0
    present = 0

    for profile_key, model_key in FIELD_MAP.items():
        pval = profile.get(profile_key)
        mval = model_cfg.get(model_key)
        status = "missing"
        if pval is not None and mval is not None:
            present += 1
            status = "match" if _numeric_equal(pval, mval) else "mismatch"
            if status == "match":
                matched += 1
        field_rows.append(
            {
                "profile_key": profile_key,
                "model_key": model_key,
                "profile_value": pval,
                "model_value": mval,
                "status": status,
            }
        )

    unresolved_profile_keys = sorted(k for k in profile.keys() if k not in KNOWN_PROFILE_KEYS)
    missing_model_keys = sorted(
        model_key
        for model_key in FIELD_MAP.values()
        if model_cfg.get(model_key) is None
    )

    return {
        "field_rows": field_rows,
        "field_count": len(FIELD_MAP),
        "field_present": present,
        "field_matched": matched,
        "coverage_ratio": (float(matched) / float(present)) if present > 0 else 0.0,
        "unresolved_profile_keys": unresolved_profile_keys,
        "missing_model_keys": missing_model_keys,
    }


def _checkpoint_state_report(payload: dict[str, Any], vocab_size: int) -> dict[str, Any]:
    model_state = payload.get("model_state")
    if not isinstance(model_state, dict):
        return {
            "has_model_state": False,
            "token_count_entries": 0,
            "min_token_id": None,
            "max_token_id": None,
            "out_of_vocab_entries": 0,
        }

    token_counts = model_state.get("token_counts", {})
    if not isinstance(token_counts, dict):
        token_counts = {}

    ids: list[int] = []
    oov = 0
    for key in token_counts.keys():
        try:
            token_id = int(key)
        except (TypeError, ValueError):
            continue
        ids.append(token_id)
        if token_id < 0 or token_id >= vocab_size:
            oov += 1

    return {
        "has_model_state": True,
        "token_count_entries": len(ids),
        "min_token_id": min(ids) if ids else None,
        "max_token_id": max(ids) if ids else None,
        "out_of_vocab_entries": oov,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit profile/config/checkpoint compatibility for warm-start imports. "
            "Writes key-coverage and unresolved-key reports."
        )
    )
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--profile-root", type=Path, default=Path("./artifacts/external_models"))

    parser.add_argument(
        "--experiment",
        default="",
        help="Hydra experiment path to compose model config (e.g. external/qwen2_5_0_5b/pretrain-fa-import-8K).",
    )
    parser.add_argument("--deploy", default="interactive")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Optional direct model config YAML path (used when --experiment is omitted).",
    )

    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--exp-folder", default="external_phase1")
    parser.add_argument("--exp-name", default="")

    parser.add_argument(
        "--on-unresolved",
        choices=["ignore", "error"],
        default="error",
        help="Policy for unresolved profile keys and missing model keys.",
    )
    parser.add_argument("--report-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    configs_dir = repo_root / "configs"
    profile_root = args.profile_root.expanduser().resolve()

    profile_path, profile = _load_profile(profile_root=profile_root, model_key=args.model_key)

    if args.experiment:
        model_cfg = _compose_model_cfg(
            configs_dir=configs_dir,
            deploy=args.deploy,
            experiment=args.experiment,
        )
        model_cfg_source = f"hydra:{args.experiment}"
    else:
        model_cfg_path = args.model_config
        if model_cfg_path is None:
            model_cfg_path = configs_dir / "model" / f"{args.model_key}.yaml"
        model_cfg_path = model_cfg_path.expanduser().resolve()
        if not model_cfg_path.exists():
            raise FileNotFoundError(f"Missing model config file: {model_cfg_path}")
        model_cfg = _load_model_cfg(model_cfg_path)
        model_cfg_source = str(model_cfg_path)

    compat = _compat_report(profile=profile, model_cfg=model_cfg)

    ckpt_state: dict[str, Any] | None = None
    checkpoint_step: int | None = None
    if args.exp_name:
        checkpoint_root = args.checkpoint_root.expanduser().resolve()
        checkpoint_step, payload = _load_checkpoint_payload(
            checkpoint_root=checkpoint_root,
            exp_folder=args.exp_folder,
            exp_name=args.exp_name,
        )
        ckpt_state = _checkpoint_state_report(
            payload=payload,
            vocab_size=int(model_cfg.get("vocab_size", 0) or 0),
        )

    unresolved_findings = list(compat["unresolved_profile_keys"]) + list(compat["missing_model_keys"])
    status = "ok"
    if unresolved_findings and args.on_unresolved == "error":
        status = "failed_unresolved"
    elif unresolved_findings:
        status = "warning_unresolved"

    report = {
        "schema_version": "1.0",
        "created_at_utc": utc_now_iso(),
        "status": status,
        "model_key": args.model_key,
        "profile_path": str(profile_path),
        "model_config_source": model_cfg_source,
        "checkpoint": {
            "exp_folder": args.exp_folder,
            "exp_name": args.exp_name,
            "step": checkpoint_step,
            "state": ckpt_state,
        },
        "compatibility": compat,
        "policy": {
            "on_unresolved": args.on_unresolved,
        },
    }

    if args.report_out is None:
        report_out = repo_root / "artifacts" / "compat" / f"{args.model_key}_compat_report.json"
    else:
        report_out = args.report_out.expanduser().resolve()
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"Wrote compatibility report: {report_out}")
    print(
        "Coverage: "
        f"matched={compat['field_matched']}/{compat['field_present']} "
        f"({compat['coverage_ratio']:.3f})"
    )

    if status == "failed_unresolved":
        print("Unresolved compatibility findings (policy=error):")
        for item in unresolved_findings:
            print(f"  - {item}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
