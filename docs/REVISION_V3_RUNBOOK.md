# Revision V3 Runbook: Width-Preserving 125M Ladder On 8x H100

Purpose: rerun the 125M warm-start ladder with the width-preserving conversion
(`PREREGISTRATION_REVISION_V3.md`) so every "warm start" actually inherits the
seed's feed-forward weights. Everything below is one GPU session.

```text
box:        8x H100 80GB SXM (Vast.ai or Prime Intellect), Ubuntu 22.04, CUDA 12
topology:   data=8, state=1, accum_steps=1
profile:    +deploy=revision_v3_vast_h100_8x
paper run:  revision_v3_widthpreserving_v1
```

Budget (observed GPU-hours from the revision-v2 runs, per GPU):

| Stage | GPU-h each | Count | Total |
| --- | ---: | ---: | ---: |
| S0 full-attention 32K | 0.5 | 1 | 0.5 |
| S1 sliding-window | 1.5 | 1 | 1.5 |
| Bridge (10%) | 2.1 | 1 | 2.1 |
| S2 extension | 1.4 | 5 seeds | 7.0 |
| No-bridge control | 1.9 | 5 seeds | 9.5 |
| Total | | | ~21 |

On 8 GPUs that is under 3 hours of training. Plan 5 to 6 wall-clock hours for
data staging, evals, and export. Do not use Blackwell (B200/B300) parts; the
pinned JAX stack failed there in June.

## 0. Before Renting Anything (Mac, $0)

```bash
uv run --exact python -m pytest -q tests/test_warmstart_guard.py tests/test_warmstart_validity.py tests/test_jax_checkpoint_partial_restore.py
uv run --exact python scripts/85_check_registry_warmstart_shapes.py --json-out reports/revision_v3/registry_warmstart_shape_check.json
```

Both must pass. The shape check must print `WARM-START SHAPE CHECK: PASS`
with no `[ERROR]` or `[WARNING]` rows (only `[INFO]` rows for the added prime
modules).

## 1. Constants On The Box

```bash
export REPO_ROOT=$HOME/Warm-starting-TTT-E2E
export DATA_ROOT=$HOME/ttt-e2e-data
export DCLM_ROOT=$DATA_ROOT/dclm_filter_8k
export BOOKS_ROOT=$DATA_ROOT/books3
export PAPER_RUN_ID=revision_v3_widthpreserving_v1
export EXP_FOLDER=revision_v3_widthpreserving_v1
export HF_125M_RESULTS_REPO=Luxel/ttt-e2e-125m-results
export HF_HUB_DISABLE_XET=1
export REVISION_V2_EXPECTED_JAX_DEVICES=8
```

Keep `.env.hf` and `.env.backblaze` untracked on the box.

## 2. Clone And Bootstrap

```bash
git clone https://github.com/lmmontoya-ai/Warm-starting-TTT-E2E.git "$REPO_ROOT"
cd "$REPO_ROOT" && git checkout release-hardening
scripts/78_bootstrap_revision_v2_prime_runtime.sh "$REPO_ROOT"
```

Expected: `backend=gpu device_count=8 local_device_count=8`.

## 3. Restore The Shared Parent And The Scratch Reference

Only the FA seed is a training parent now. Restore S3 as well so the scratch
reference is re-evaluated through the same pipeline as the new runs.

```bash
cd "$REPO_ROOT"
for spec in \
  "S0_PRETRAIN_FA_125M pretrain-125m-fa" \
  "S3_125M ext-125m-e2e-32K"
do
  set -- $spec
  uv run --with huggingface_hub python scripts/46_restore_stage_from_hf.py \
    --repo-id "$HF_125M_RESULTS_REPO" \
    --source-paper-run-id protocol_r_125m_main_v1 \
    --source-stage-id "$1" \
    --source-run-id "$2" \
    --target-paper-run-id "$PAPER_RUN_ID"
done
```

Expected checkpoint roots:

```text
checkpoints/revision_v3_widthpreserving_v1/pretrain-125m-fa
checkpoints/revision_v3_widthpreserving_v1/ext-125m-e2e-32K
```

## 4. Stage Tokenized Data And Fingerprints

```bash
# The tokenized 125M package lives under this prefix in the TTTE2E bucket
# (about 40 GB: dclm train 10.4, dclm val 20.4, books train 0.8, books val 8.4).
# The bucket-level default prefix in .env.backblaze points at an empty path, so
# pass the package prefix explicitly.
uv run --exact python scripts/28_fetch_b2_dataset.py \
  --env-file .env.backblaze --dest-root "$DATA_ROOT" \
  --b2-prefix ttt-e2e-datasets/paper_budget_125m_val-full \
  --datasets dclm_filter_8k,books3 --splits train,val
for dataset in dclm_filter_8k books3; do
  for split in train val; do
    uv run --exact python scripts/13_dataset_fingerprint.py \
      --dataset-id "$dataset" --path "$DATA_ROOT/$dataset" --split "$split"
  done
done
```

## 5. Restore-Coverage Audit On The Box (CPU or GPU, minutes)

This is the check that was missing in revision v2. Run it for every
conversion before training:

```bash
for stage in S0_125M S1_125M S2_ADAPT_125M S2_MINUS_125M; do
  uv run --exact python scripts/84_audit_warmstart_restore.py \
    --stage-id "$stage" --checkpoint-root ./checkpoints --exp-folder "$EXP_FOLDER" \
    --json-out "reports/revision_v3/$PAPER_RUN_ID/restore_audit_$stage.json"
done
```

Expected for the E2E conversions: `restored=183.47M (94.1%)`, `shape-mismatch
0.00M`, `new 11.51M` (the prime MLPs only), and `WARM-START RESTORE AUDIT:
PASS`. S0 and S1 must show `shape-mismatch 0.00M`.

## 6. Smoke: 20 Steps Of The Bridge

```bash
uv run --exact python scripts/86_run_revision_v3_ladder.py \
  --paper-run-id revision_v3_smoke_v1 --exp-folder "$EXP_FOLDER" \
  --deploy revision_v3_vast_h100_8x \
  --dclm-root "$DCLM_ROOT" --books-root "$BOOKS_ROOT" \
  --stages S2_ADAPT_125M --adapt-steps 20 --save-milestone-freq 10
```

Read `experiments/revision_v3_smoke_v1/S2_ADAPT_125M/adapt-125m-e2e-8K-from-fa/metrics.jsonl`.
The step-0 loss must be far below `7.0` (the trainer aborts otherwise). Note
the steady-state seconds per step and extrapolate the full session. Delete the
smoke checkpoint directory afterwards so it is not mistaken for the real bridge:

```bash
rm -rf "checkpoints/$EXP_FOLDER/adapt-125m-e2e-8K-from-fa"
```

## 7. Launch The Ladder

```bash
uv run --exact python scripts/86_run_revision_v3_ladder.py \
  --paper-run-id "$PAPER_RUN_ID" --exp-folder "$EXP_FOLDER" \
  --deploy revision_v3_vast_h100_8x \
  --dclm-root "$DCLM_ROOT" --books-root "$BOOKS_ROOT" \
  --seeds 1-5 --adapt-steps 480 --ext-steps 480 --ext-global-batch-size 8 \
  --save-milestone-freq 30 \
  --export-to-hf --hf-repo-id "$HF_125M_RESULTS_REPO"
```

Order: S0, S1, bridge, then (S2, S2-minus) for seeds 1 to 5. The launcher
refuses to start if the registry shape check fails or the FA parent is
missing. Each warm-started run writes `restore_report.json` next to its
`metrics.jsonl`; a run that fails the coverage or initial-loss gate stops the
ladder. Rerunning the same command resumes from the last milestone.

## 8. Checkpoint-Based 64-Batch Evaluation

Same surface as revision v2: float32 restore, Books3 validation, context
`32768`, `64` batches of `8`.

```bash
export XLA_FLAGS=--xla_gpu_autotune_level=3   # S0 full attention at 32K OOMed during autotuning without this
uv run --exact python scripts/18_eval_matrix.py \
  --paper-run-id "$PAPER_RUN_ID" --exp-folder "$EXP_FOLDER" \
  --checkpoint-root ./checkpoints \
  --dclm-root "$DCLM_ROOT" --books-root "$BOOKS_ROOT" \
  --eval-id jax_parity_eval64 --contexts 32768 --datasets books3 \
  --eval-batches 64 --eval-batch-size 8 \
  --stages S0_125M,S1_125M,S2_ADAPT_125M,S2_125M,S2_MINUS_125M,S3_125M \
  --summary-json "reports/revision_v3/$PAPER_RUN_ID/books32k_eval64_summary.json" \
  --summary-csv  "reports/revision_v3/$PAPER_RUN_ID/books32k_eval64_summary.csv" \
  --strict
```

The script discovers every run directory under
`experiments/$PAPER_RUN_ID/<stage>/<run_id>` and filters by stage, so the
seed-suffixed S2 and S2-minus runs are picked up automatically.

Also evaluate the bridge output on DCLM-8K (context `8192`, dataset
`dclm_filter_8k`) so the short-context quality after the bridge is on record
next to the seed's `3.5242`.

## 9. Preregistered E1 Analysis On The New Pairs

```bash
uv run --exact python scripts/80_analyze_revision_v2_e1.py \
  --eval-summary "reports/revision_v3/$PAPER_RUN_ID/books32k_eval64_summary.json" \
  --run-summary  "reports/revision_v3/$PAPER_RUN_ID/ladder_summary.json" \
  --baseline-summary "reports/revision_v3/$PAPER_RUN_ID/books32k_eval64_summary.json" \
  --out-dir "reports/revision_v3/$PAPER_RUN_ID/e1_pairs_eval64"
```

Interpretation follows `PREREGISTRATION_REVISION_V2.md` (margin `0.10`, paired
seed bootstrap) and the sanity expectations in `PREREGISTRATION_REVISION_V3.md`.

## 10. Tear Down

1. Verify every stage exported to HF (`hf_export.status == succeeded` in
   `ladder_summary.json`).
2. Copy `reports/revision_v3/` and the `restore_report.json` files off the box.
3. Destroy the instance and confirm with `vastai show instances --raw` (or
   `prime pods list`).
4. Append the session cost to `docs/REVISION_V2_COST_LEDGER.md` under a
   revision-v3 heading.
