# 125M Ablation Vast Runtime

Use the pinned Vast runtime profile in `configs/research/vast_runtime_profiles.yaml` for `125M` continuation-ablation retries.

Current pinned profile:
- profile: `h200_125m_ablation_v1`
- image: `nvidia/cuda:12.4.0-devel-ubuntu22.04`
- hardware target: `8x H200`
- driver floor: `560.35.03`
- bootstrap: `scripts/62_bootstrap_vast_125m_ablation_runtime.sh`

Why this exists:
- the earlier continuation-ablation debugging used drifting container surfaces
- that made it too easy to conflate repo/runtime changes with image drift
- future retries should hold the base image fixed and let the repo lockfile define the Python userspace stack

Required rule:
- do not use floating images such as `pytorch/pytorch:latest` for `125M` continuation-ablation retries

Suggested boot sequence on a fresh Vast node:
```bash
cd /root/Reproducing-TTT-E2E
bash scripts/62_bootstrap_vast_125m_ablation_runtime.sh /root/Reproducing-TTT-E2E
```

This profile only removes image drift. It does not claim that the `32K` continuation issue is solved; it makes the next retry reproducible.
