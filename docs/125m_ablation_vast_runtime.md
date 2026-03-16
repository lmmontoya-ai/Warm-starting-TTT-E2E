# 125M Ablation Vast Runtime

Use the pinned Vast runtime profile in `configs/research/vast_runtime_profiles.yaml` for `125M` continuation-ablation retries.

Current pinned profile:
- profile: `h200_125m_ablation_v1`
- image: `nvidia/cuda:12.4.0-devel-ubuntu22.04`
- hardware target: `8x H200`
- driver floor: `560.35.03`
- bootstrap: `scripts/62_bootstrap_vast_125m_ablation_runtime.sh`

Extension-debug profile:
- profile: `h200_125m_ext32k_legacy_v1`
- image: `nvidia/cuda:12.9.0-devel-ubuntu22.04`
- hardware target: `8x H200`
- driver floor: `560.35.03`
- requirements: `requirements/125m_ext32k_runtime.txt`
- bootstrap: `scripts/63_bootstrap_vast_125m_ext32k_runtime.sh`

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

For `S2` / `S3` `32K` continuation debugging, prefer the extension-debug profile first. It recreates the extension-era JAX/CUDA/Orbax userspace surface from commit `6ee3e98`, which is closer to the runtime that originally completed the canonical `32K` extension stages.
