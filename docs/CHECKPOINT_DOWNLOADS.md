# Checkpoint Downloads

This document names the external checkpoint artifacts used by the revision
experiments and shows how to restore them into the local layout expected by the
warm-start registry and evaluation scripts.

Large checkpoints are not committed to git. The public result mirrors are:

| Scale | Hugging Face repo | Visibility | Use |
| --- | --- | --- | --- |
| 125M | `Luxel/ttt-e2e-125m-results` | public | E1/E2a baselines, E3 parents, current-pipeline re-eval |
| 760M | `Luxel/ttt-e2e-760m-results` | public | 760M E2a baselines and paper-quality re-eval |

The separate raw author-seed repo `Luxel/ttt-e2e-author-760m-orbax` is not part
of the public result mirror. It is only needed if a future plan re-runs 760M
training from the original author-provided upstream seeds.

## Local Layout

Use the repo defaults unless a cloud runner requires another root:

```bash
export EXP_DIR=./experiments
export CHECKPOINT_ROOT=./checkpoints
```

The restore helper writes:

```text
experiments/<target_paper_run_id>/<stage_id>/<run_id>/
checkpoints/<target_paper_run_id>/<run_id>/
```

The checkpoint directory contains `latest.json`, optional step metadata, and
the latest Orbax step directory. The experiment directory contains the resolved
config, run manifests, and the `hf_restore_manifest.json` sidecar.

## Restore Helper

Use `scripts/46_restore_stage_from_hf.py` for stage exports. Public repos do
not require a token.

```bash
uv run --with huggingface_hub python scripts/46_restore_stage_from_hf.py \
  --repo-id <hf_repo> \
  --source-paper-run-id <source_paper_run_id> \
  --source-stage-id <source_stage_id> \
  --source-run-id <source_run_id> \
  --target-paper-run-id <target_paper_run_id>
```

Add `--overwrite` only when deliberately replacing a local restore. For a
preview that does not download files, add `--dry-run`.

For large checkpoint downloads, setting `HF_HUB_DISABLE_XET=1` is recommended
because it avoids Xet finalization stalls observed on multi-GB Orbax shards:

```bash
export HF_HUB_DISABLE_XET=1
```

## What To Restore By Experiment

| Task | Required checkpoints |
| --- | --- |
| E1 paired training | 125M `S0_PRETRAIN_FA_125M` and `S2_ADAPT_125M`, restored once into the shared E1 artifact folder; S2/S2-minus seed outputs are produced by the new runs |
| E1 current-pipeline baseline re-eval | 125M `S1_125M`, `S2_125M`, `S3_125M`, restored into the same shared E1 artifact folder; restore `S0_125M` too if reporting the full S0-S3 table |
| E2a 125M retrieval | 125M `S2_125M`, `S3_125M`, and the newly produced `S2_MINUS_125M`; add `S0_125M`/`S1_125M` for the full comparison table |
| E2a 760M retrieval | 760M `S2`, `S3`; restore `S2_ADAPT` only for bridge/8K diagnostics |
| E3 bridge-budget ablation | 125M `S0_PRETRAIN_FA_125M` as the FA parent for training; restore completed `S2_BRIDGE_5PCT_125M`, `S2_BRIDGE_20PCT_125M`, `S2_BRIDGE_40PCT_125M`, and `S2_MINUS_CONT_125M` for analysis |
| 760M paper re-eval | 760M `S2`, `S3`; restore `S2_ADAPT` for DCLM-8K bridge-surface checks |

## Canonical 125M Stages

Source repo: `Luxel/ttt-e2e-125m-results`

Source paper run: `protocol_r_125m_main_v1`

| Stage | Run id | Latest step | Use |
| --- | --- | ---: | --- |
| `S0_PRETRAIN_FA_125M` | `pretrain-125m-fa` | 4799 | FA seed parent for E1/E3 |
| `S0_125M` | `ext-125m-fa-32K` | 479 | full-attention 32K baseline |
| `S1_125M` | `ext-125m-swa-32K-from-fa` | 479 | SWA-only 32K baseline |
| `S2_ADAPT_125M` | `adapt-125m-e2e-8K-from-fa` | 479 | original 8K bridge |
| `S2_125M` | `ext-125m-e2e-32K-from-fa-bridge` | 479 | warm-started TTT-E2E baseline |
| `S3_PRETRAIN_E2E_125M` | `pretrain-125m-e2e` | 4799 | scratch TTT-E2E seed |
| `S3_125M` | `ext-125m-e2e-32K` | 479 | scratch TTT-E2E 32K baseline |

Restore the E1 shared parents once. Use the same target name for
`--paper-run-id` and `--exp-folder` during E1 so all seed runs resolve the same
read-only parent copies and `scripts/40_export_stage_to_hf.py` can export the
results without path remapping.

```bash
export HF_HUB_DISABLE_XET=1
export E1_PAPER_RUN_ID=revision_v2_e1_paired_v1

for spec in \
  "S0_PRETRAIN_FA_125M pretrain-125m-fa" \
  "S2_ADAPT_125M adapt-125m-e2e-8K-from-fa"
do
  set -- $spec
  uv run --with huggingface_hub python scripts/46_restore_stage_from_hf.py \
    --repo-id Luxel/ttt-e2e-125m-results \
    --source-paper-run-id protocol_r_125m_main_v1 \
    --source-stage-id "$1" \
    --source-run-id "$2" \
    --target-paper-run-id "$E1_PAPER_RUN_ID"
done
```

For the preregistered five paired E1 seeds, do not restore parents per seed.
Every seed must read:

```text
checkpoints/revision_v2_e1_paired_v1/pretrain-125m-fa
checkpoints/revision_v2_e1_paired_v1/adapt-125m-e2e-8K-from-fa
```

Restore the current-pipeline baseline re-eval set:

```bash
export HF_HUB_DISABLE_XET=1
export TARGET_RUN=revision_v2_e1_paired_v1

for spec in \
  "S1_125M ext-125m-swa-32K-from-fa" \
  "S2_125M ext-125m-e2e-32K-from-fa-bridge" \
  "S3_125M ext-125m-e2e-32K"
do
  set -- $spec
  uv run --with huggingface_hub python scripts/46_restore_stage_from_hf.py \
    --repo-id Luxel/ttt-e2e-125m-results \
    --source-paper-run-id protocol_r_125m_main_v1 \
    --source-stage-id "$1" \
    --source-run-id "$2" \
    --target-paper-run-id "$TARGET_RUN"
done
```

Add `S0_125M ext-125m-fa-32K` to that loop if the full S0-S3 table is being
regenerated.

## Revision V2 E3 Output Stages

Source repo: `Luxel/ttt-e2e-125m-results`

Completed E3 bridge-budget frontier:

| Source paper run | Stage | Run id | Latest step | Use |
| --- | --- | --- | ---: | --- |
| `revision_v2_e3_frontier_v1` | `S2_BRIDGE_5PCT_125M` | `ext-125m-e2e-32K-from-fa-bridge5pct-seed001` | 479 | 5% bridge frontier point |
| `revision_v2_e3_frontier_v1` | `S2_BRIDGE_20PCT_125M` | `ext-125m-e2e-32K-from-fa-bridge20pct-seed001` | 479 | 20% bridge frontier point |
| `revision_v2_e3_frontier_v1` | `S2_BRIDGE_40PCT_125M` | `ext-125m-e2e-32K-from-fa-bridge40pct-seed001` | 479 | 40% bridge frontier point |
| `revision_v2_s2minus_cont_v1` | `S2_MINUS_CONT_125M` | `ext-125m-e2e-32K-from-fa-direct-cont1440-seed001` | 1919 | S2-minus +1440 continuation |

Restore the completed E3 analysis set:

```bash
export HF_HUB_DISABLE_XET=1

for spec in \
  "revision_v2_e3_frontier_v1 S2_BRIDGE_5PCT_125M ext-125m-e2e-32K-from-fa-bridge5pct-seed001 revision_v2_e3_frontier_v1" \
  "revision_v2_e3_frontier_v1 S2_BRIDGE_20PCT_125M ext-125m-e2e-32K-from-fa-bridge20pct-seed001 revision_v2_e3_frontier_v1" \
  "revision_v2_e3_frontier_v1 S2_BRIDGE_40PCT_125M ext-125m-e2e-32K-from-fa-bridge40pct-seed001 revision_v2_e3_frontier_v1" \
  "revision_v2_s2minus_cont_v1 S2_MINUS_CONT_125M ext-125m-e2e-32K-from-fa-direct-cont1440-seed001 revision_v2_s2minus_cont_v1"
do
  set -- $spec
  uv run --with huggingface_hub python scripts/46_restore_stage_from_hf.py \
    --repo-id Luxel/ttt-e2e-125m-results \
    --source-paper-run-id "$1" \
    --source-stage-id "$2" \
    --source-run-id "$3" \
    --target-paper-run-id "$4"
done
```

The HF stage export includes checkpoint and top-level experiment sidecars. The
local E3 report/eval bundle copied from the production pod is preserved under
`reports/revision_v2/e3_frontier/remote_bundle/`; because reports are generated
artifacts, the tracked source-of-truth summary is `CANONICAL_RESULTS.md`.

## Canonical 760M Stages

Source repo: `Luxel/ttt-e2e-760m-results`

Source paper run: `protocol_r_760m_author_seed_v1`

| Stage | Run id | Latest step | Use |
| --- | --- | ---: | --- |
| `S2_ADAPT` | `adapt-760m-e2e-8K-from-fa` | 23199 | 8K bridge checkpoint |
| `S2` | `ext-760m-e2e-32K-from-fa-bridge` | 11599 | warm-started TTT-E2E 32K baseline |
| `S3` | `ext-760m-e2e-32K` | 11599 | scratch TTT-E2E 32K baseline |

Restore the 760M E2a/paper re-eval set:

```bash
export HF_HUB_DISABLE_XET=1
export TARGET_RUN=revision_v2_760m_current_eval

for spec in \
  "S2 ext-760m-e2e-32K-from-fa-bridge" \
  "S3 ext-760m-e2e-32K"
do
  set -- $spec
  uv run --with huggingface_hub python scripts/46_restore_stage_from_hf.py \
    --repo-id Luxel/ttt-e2e-760m-results \
    --source-paper-run-id protocol_r_760m_author_seed_v1 \
    --source-stage-id "$1" \
    --source-run-id "$2" \
    --target-paper-run-id "$TARGET_RUN"
done
```

Restore the 760M bridge checkpoint only when needed:

```bash
export HF_HUB_DISABLE_XET=1

uv run --with huggingface_hub python scripts/46_restore_stage_from_hf.py \
  --repo-id Luxel/ttt-e2e-760m-results \
  --source-paper-run-id protocol_r_760m_author_seed_v1 \
  --source-stage-id S2_ADAPT \
  --source-run-id adapt-760m-e2e-8K-from-fa \
  --target-paper-run-id revision_v2_760m_current_eval
```

## Verification

After any restore, verify the local stage has a latest checkpoint:

```bash
python - <<'PY'
from pathlib import Path

root = Path("checkpoints")
for latest in sorted(root.glob("*/**/latest.json")):
    print(latest)
PY
```

For a stricter stage check, inspect the restore manifest:

```bash
python - <<'PY'
import json
from pathlib import Path

for path in sorted(Path("experiments").glob("*/**/hf_restore_manifest.json")):
    payload = json.loads(path.read_text())
    print(
        payload["target_paper_run_id"],
        payload["target_stage_id"],
        payload["target_run_id"],
        payload.get("latest_step"),
    )
PY
```

Before interpreting E1/E2a results, remember the preregistration requirement:
baseline losses must be re-evaluated through the current checkpoint-based
float32 pipeline, not read directly from the historical ledger.
