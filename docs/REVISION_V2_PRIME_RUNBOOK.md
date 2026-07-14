# Revision V2 Prime GPU Runbook

This runbook is the execution checklist for the revision-v2 E1 program on the
Prime Intellect / Massed Compute VM:

```text
pod id: 86b060fa01364fd58b305d88c82e150b
name: illegal-dexterous-chital
gpu: H100_80GB x2
image: ubuntu_22_cuda_12
topology: data=2, state=1
```

The committed runtime profile is `revision_v2_prime_h100_2x`; use
`+deploy=revision_v2_prime_h100_2x` for all revision-v2 GPU runs. Do not use
`+deploy=interactive`, which is an older 8-device interactive profile.

## Constants

```bash
export REPO_ROOT=/home/ubuntu/Warm-starting-TTT-E2E
export DATA_ROOT=/home/ubuntu/ttt-e2e-data
export DCLM_ROOT=$DATA_ROOT/dclm_filter_8k
export BOOKS_ROOT=$DATA_ROOT/books3
export PAPER_RUN_ID=revision_v2_e1_paired_v1
export EXP_FOLDER=revision_v2_e1_paired_v1
export HF_125M_RESULTS_REPO=Luxel/ttt-e2e-125m-results
export HF_HUB_DISABLE_XET=1
```

Use local `.env.hf` and `.env.backblaze` files on the VM for credentials. They
must remain untracked.

## 1. Clone And Bootstrap

```bash
git clone https://github.com/lmmontoya-ai/Warm-starting-TTT-E2E.git "$REPO_ROOT"
cd "$REPO_ROOT"
git checkout release-hardening

scripts/78_bootstrap_revision_v2_prime_runtime.sh "$REPO_ROOT"
```

The bootstrap must report:

```text
backend=gpu
device_count=2
local_device_count=2
```

## 2. Restore Shared Parents And Baselines

Restore parents and current-eval baselines into the same shared artifact folder:

```bash
cd "$REPO_ROOT"
export HF_HUB_DISABLE_XET=1
export PAPER_RUN_ID=revision_v2_e1_paired_v1

for spec in \
  "S0_PRETRAIN_FA_125M pretrain-125m-fa" \
  "S2_ADAPT_125M adapt-125m-e2e-8K-from-fa" \
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
    --target-paper-run-id "$PAPER_RUN_ID"
done
```

The expected checkpoint roots are:

```text
checkpoints/revision_v2_e1_paired_v1/pretrain-125m-fa
checkpoints/revision_v2_e1_paired_v1/adapt-125m-e2e-8K-from-fa
checkpoints/revision_v2_e1_paired_v1/ext-125m-swa-32K-from-fa
checkpoints/revision_v2_e1_paired_v1/ext-125m-e2e-32K-from-fa-bridge
checkpoints/revision_v2_e1_paired_v1/ext-125m-e2e-32K
```

## 3. Stage Tokenized Data

Fetch the existing tokenized datasets from Backblaze. Do not attempt to
re-download Books3 from raw public sources.

```bash
cd "$REPO_ROOT"
uv run --exact python scripts/28_fetch_b2_dataset.py \
  --env-file .env.backblaze \
  --dest-root "$DATA_ROOT" \
  --datasets dclm_filter_8k,books3 \
  --splits train,val
```

Generate or refresh fingerprints:

```bash
for dataset in dclm_filter_8k books3; do
  for split in train val; do
    uv run --exact python scripts/13_dataset_fingerprint.py \
      --dataset-id "$dataset" \
      --path "$DATA_ROOT/$dataset" \
      --split "$split"
  done
done
```

## 4. E1 Preflight

```bash
cd "$REPO_ROOT"
uv run --exact python scripts/76_e1_preflight.py \
  --exp-folder "$EXP_FOLDER" \
  --checkpoint-root ./checkpoints \
  --dclm-root "$DCLM_ROOT" \
  --books-root "$BOOKS_ROOT" \
  --canonical-dclm-root "$DCLM_ROOT" \
  --canonical-books-root "$BOOKS_ROOT" \
  --output-dir ./reports/revision_v2/e1_preflight
```

Required result:

```text
PREFLIGHT: PASS
```

## 5. Calibration And Kill-Resume

Run a short S2-minus calibration with frequent checkpoints:

```bash
cd "$REPO_ROOT"
uv run --exact python scripts/79_run_revision_v2_e1_pairs.py \
  --paper-run-id revision_v2_calibration_v1 \
  --exp-folder "$EXP_FOLDER" \
  --deploy revision_v2_prime_h100_2x \
  --dclm-root "$DCLM_ROOT" \
  --books-root "$BOOKS_ROOT" \
  --seeds 1 \
  --arms S2_MINUS_125M \
  --ext-steps 60 \
  --ext-global-batch-size 8 \
  --save-milestone-freq 5
```

For the kill-resume test, interrupt the process after the first checkpoint is
written, then rerun the exact command. The launcher should detect
`checkpoints/$EXP_FOLDER/ext-125m-e2e-32K-from-fa-direct-seed001/latest.json`
and resume with `training.load_part=all`.

Record steps/sec from `metrics.jsonl` and extrapolate the five-pair wall-clock
before launching production seeds.

## 6. Current-Pipeline Baseline Re-Eval

```bash
cd "$REPO_ROOT"
uv run --exact python scripts/18_eval_matrix.py \
  --paper-run-id "$PAPER_RUN_ID" \
  --exp-folder "$EXP_FOLDER" \
  --checkpoint-root ./checkpoints \
  --dclm-root "$DCLM_ROOT" \
  --books-root "$BOOKS_ROOT" \
  --stages S1_125M,S2_125M,S3_125M \
  --contexts 32768 \
  --datasets books3 \
  --eval-batches 8 \
  --strict
```

Export the re-evaluated baseline stages:

```bash
for spec in \
  "S1_125M ext-125m-swa-32K-from-fa" \
  "S2_125M ext-125m-e2e-32K-from-fa-bridge" \
  "S3_125M ext-125m-e2e-32K"
do
  set -- $spec
  uv run --exact python scripts/40_export_stage_to_hf.py \
    --paper-run-id "$PAPER_RUN_ID" \
    --stage-id "$1" \
    --run-id "$2" \
    --repo-id "$HF_125M_RESULTS_REPO" \
    --checkpoint-exp-folder "$EXP_FOLDER" \
    --require-eval-success
done
```

## 7. Launch Five Paired E1 Seeds

```bash
cd "$REPO_ROOT"
uv run --exact python scripts/79_run_revision_v2_e1_pairs.py \
  --paper-run-id "$PAPER_RUN_ID" \
  --exp-folder "$EXP_FOLDER" \
  --deploy revision_v2_prime_h100_2x \
  --dclm-root "$DCLM_ROOT" \
  --books-root "$BOOKS_ROOT" \
  --seeds 1-5 \
  --arms S2_125M,S2_MINUS_125M \
  --ext-steps 480 \
  --ext-global-batch-size 8 \
  --save-milestone-freq 30 \
  --export-to-hf \
  --hf-repo-id "$HF_125M_RESULTS_REPO"
```

This phase produces numbers only. Interpretation is deferred to the
preregistered decision gate in `PREREGISTRATION_REVISION_V2.md`.

## 8. Tear Down Discipline

After the five paired seeds and exports complete:

1. Sync `reports/revision_v2/`, `reports/paper/$PAPER_RUN_ID/`, and run
   summaries off the VM.
2. Verify HF exports for every seed arm.
3. Stop the VM before any advisor interpretation pause.
