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

Observed working recipe for the direct `S2` continuation gate on Vast `8x H200`:
- bootstrap with `scripts/63_bootstrap_vast_125m_ext32k_runtime.sh`
- use the packaged Books root mirrored at:
  - `s3://$B2_BUCKET/$B2_DATASET_PREFIX/paper_budget_125m_val-full/books3`
  - not the generic `s3://$B2_BUCKET/$B2_DATASET_PREFIX/books3`
- materialize the canonical `S2_125M` checkpoint locally under:
  - `checkpoints/protocol_r_125m_main_v1/ext-125m-e2e-32K-from-fa-bridge`
- for direct one-host probes, force:
  - `backend.distributed=false`
  - `+deploy_paths.checkpoint=/root/Reproducing-TTT-E2E/checkpoints`
  - `+deploy_paths.data.books3=/root/ttt-e2e-data/books3`

Validated direct probe:
- host family: `317686`
- driver: `570.148.08`
- image: `nvidia/cuda:12.9.0-devel-ubuntu22.04`
- probe: resume canonical `S2_125M` step `479` with `load_part=all` and `total_steps=482`
- result:
  - checkpoint restore succeeded
  - step `480` completed
  - step `481` completed
  - new checkpoint `481` written under the ablation run root
