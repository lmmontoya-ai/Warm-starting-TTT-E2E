from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_revision_v2_prime_profile_pins_data_two_topology() -> None:
    profiles = _load_yaml(REPO_ROOT / "configs" / "research" / "vast_runtime_profiles.yaml")
    profile = profiles["profiles"]["revision_v2_prime_h100_2x"]

    assert profile["provider"] == "prime_intellect"
    assert profile["cloud"] == "massedcompute"
    assert profile["gpu"] == "H100_80GB"
    assert profile["gpu_count"] == 2
    assert profile["hydra_deploy"] == "revision_v2_prime_h100_2x"
    assert profile["topology"]["backend_num_devices"] == 2
    assert profile["topology"]["training_n_data_parallel"] == 2
    assert profile["topology"]["training_n_state_parallel"] == 1
    assert profile["env"]["HF_HUB_DISABLE_XET"] == "1"


def test_revision_v2_prime_a100_profile_pins_data_eight_topology() -> None:
    profiles = _load_yaml(REPO_ROOT / "configs" / "research" / "vast_runtime_profiles.yaml")
    profile = profiles["profiles"]["revision_v2_prime_a100_8x"]

    assert profile["provider"] == "prime_intellect"
    assert profile["cloud"] == "datacrunch"
    assert profile["gpu"] == "A100_80GB"
    assert profile["gpu_count"] == 8
    assert profile["hydra_deploy"] == "revision_v2_prime_a100_8x"
    assert profile["topology"]["backend_num_devices"] == 8
    assert profile["topology"]["training_n_data_parallel"] == 8
    assert profile["topology"]["training_n_state_parallel"] == 1
    assert profile["env"]["REVISION_V2_EXPECTED_JAX_DEVICES"] == "8"


def test_revision_v2_prime_deploy_configs_match_runtime_profiles() -> None:
    profiles = _load_yaml(REPO_ROOT / "configs" / "research" / "vast_runtime_profiles.yaml")[
        "profiles"
    ]
    deploy_names = ["revision_v2_prime_h100_2x", "revision_v2_prime_a100_8x"]

    for name in deploy_names:
        profile = profiles[name]
        deploy = _load_yaml(REPO_ROOT / "configs" / "deploy" / f"{name}.yaml")

        assert deploy["backend"]["num_devices"] == profile["topology"]["backend_num_devices"]
        assert deploy["training"]["n_data_parallel"] == profile["topology"]["training_n_data_parallel"]
        assert deploy["training"]["n_state_parallel"] == profile["topology"]["training_n_state_parallel"]
