# External Models Runbook (Qwen2.5-0.5B + SmolLM2-360M)

This runbook describes everything needed to launch scratch and adapter training
paths for the external-pretrained follow-up track.

## Scope
- Primary: `Qwen/Qwen2.5-0.5B`
- Secondary: `HuggingFaceTB/SmolLM2-360M`
- Workflows covered:
  - scratch FA path,
  - imported FA controls,
  - SWA bridge,
  - TTT bridge and direct 32K extension.

## 1) Prepare external model profiles

```bash
uv run --exact python scripts/07_prepare_external_models.py --model all
```

Outputs:
- `./artifacts/external_models/qwen2_5_0_5b/hf_config.json`
- `./artifacts/external_models/qwen2_5_0_5b/model_profile.json`
- `./artifacts/external_models/smollm2_360m/hf_config.json`
- `./artifacts/external_models/smollm2_360m/model_profile.json`

## 2) Inspect launch matrix

```bash
uv run --exact python scripts/08_external_matrix.py \
  --model all \
  --path all \
  --runtime-mode token_stats
```

## 3) Run short-budget pilot

```bash
uv run --exact python scripts/09_external_pilot.py \
  --model all \
  --path all \
  --runtime-mode token_stats \
  --bootstrap-token-data \
  --exp-folder external_phase1_pilot \
  --pretrain-steps 8 \
  --adapt-steps 4 \
  --ext-steps 4 \
  --skip-existing
```

Outputs:
- manifest JSON in `./experiments/*external_manifest.json`
- CSV/JSON report in `./experiments/*external_report.{csv,json}`

## 4) Run scratch-only or adapter-only

Scratch-only:

```bash
uv run --exact python scripts/09_external_pilot.py \
  --model qwen2_5_0_5b \
  --path scratch \
  --bootstrap-token-data
```

Adapter-only:

```bash
uv run --exact python scripts/09_external_pilot.py \
  --model smollm2_360m \
  --path adapter \
  --bootstrap-token-data
```

## 5) Config map

### Qwen configs
- Model: `configs/model/qwen2_5_0_5b.yaml`
- Training: `configs/training/qwen2_5_0_5b/*.yaml`
- Experiments: `configs/experiment/external/qwen2_5_0_5b/*.yaml`

### Smol configs
- Model: `configs/model/smollm2_360m.yaml`
- Training: `configs/training/smollm2_360m/*.yaml`
- Experiments: `configs/experiment/external/smollm2_360m/*.yaml`

## 6) Notes for comparability
- Use same stage split (8K short stage + 32K extension).
- Keep FA imported controls when evaluating adapter variants.
- Report both token-matched and wall-clock-matched comparisons.
- Use `scripts/05_phase1_report.py` outputs as the primary run ledger.
