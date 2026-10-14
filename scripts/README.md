# Scripts

Run every script from the repository root, for example
`uv run --exact python scripts/96_run_revision_v3_ladder.py --help`.
Each Python script prints its inputs and outputs with `--help`.

Scripts marked "entry point" are the supported way in.
The others are helpers that entry points call, or tools for one specific check.
Numbers are historical and have gaps where withdrawn or provider-specific scripts were removed.
Git history keeps the removed scripts.

## Host setup

- `bootstrap_gpu_host.sh` (entry point) installs system packages and uv on a Linux GPU host, syncs the locked environment, and checks that JAX sees the expected number of GPUs.

## Data

- `00_data_fetch.py` (entry point) downloads the original authors' DCLM 8K and Books3 token datasets from their Requester Pays GCS buckets into `<dest-root>/<dataset>/<split>`. It needs `--billing-project`. `--budget 125m` copies only the 125M token budget plus the val splits, about 40 GB.
- `13_dataset_fingerprint.py` (entry point) writes the `<split>.fingerprint.json` sidecar that the ladder requires next to each dataset split.
- `02_data_validate.py` checks a Zarr dataset's layout and how `seq_len` slices it.
- `04_make_token_data.py` writes small synthetic token streams for `token_stats` pilot runs.
- `14_dataset_card.py` builds a CSV and JSON dataset card from fingerprint files.
- `55_audit_compact_books3.py` checks that a budget Books3 train split is a prefix of a larger copy and samples how its tokens compare with val.

## Core ladder

- `96_run_revision_v3_ladder.py` (entry point) runs the revision-v3 125M ladder per seed: S0, S1, the S3 re-extension, the bridge, S2 and S2-minus, with optional parameter-matched scratch runs. Every run passes the trainer's restore, value, initial-loss and LR-schedule gates. `--eval-after` chains the Books32K evaluation.
- `23_warmstart_registry.py` runs any stage ladder defined in `configs/research/warmstart_registry.yaml` and writes lineage manifests.
- `25_run_b1_b2_real.py` runs the 760M B1 and B2 baselines (full-attention and TTT-E2E pretrain, then 32K extension) followed by evaluation and tables.
- `01_config_inspect.py` prints how Hydra resolves an experiment config.
- `03_pretrained_matrix.py` prints the launch commands for the pretrained warm-start matrix.
- `05_phase1_report.py` summarizes runs: tokens, wall-clock time and checkpoint lineage.
- `06_phase1_pilot.py` runs the short B1, B2, P1 and P2 pilot matrix and its reports.
- `59_backfill_run_metadata.py` rewrites `run_result.json` and `budget_manifest.json` from the tokens a run's metrics actually record.

## Checkpoint transfer

- `40_export_stage.py` (entry point) exports one finished stage to a local directory (`--dest-dir`) or to your Hugging Face repo (`--repo-id`), in the stage-export layout `<paper_run_id>/stages/<stage_id>/<run_id>/`.
- `46_restore_stage.py` (entry point) restores one exported stage from a local directory (`--source-dir`) or a Hugging Face repo (`--repo-id`) into `experiments/` and `checkpoints/`, and records the source in `restore_manifest.json`. Use it to install the 125M seeds before running `96`.

## External models (Qwen, SmolLM)

- `07_prepare_external_models.py` fetches public Qwen and SmolLM configs and writes local model profiles.
- `08_external_matrix.py` prints launch commands for the scratch and adapter runs on those models.
- `09_external_pilot.py` runs short scratch and adapter pilots on Qwen2.5-0.5B and SmolLM2-360M.
- `10_seed_external_import_checkpoints.py` creates the import checkpoints that the adapter runs start from.
- `11_external_eval.py` evaluates external pilot runs on loss against context, efficiency and NIAH and decode proxies.
- `12_external_e2e_research.py` chains 07, 10, 09 and 11, then tables, figures and a bundle.
- `15_import_hf_checkpoint.py` converts a Hugging Face model into an import checkpoint for adapter warm starts.
- `16_audit_checkpoint_compat.py` reports which profile and checkpoint keys an import covers.
- `17_probe_warmstart_init.py` measures the initial loss and early loss slope of a warm start in the `token_stats` runtime.

## Evaluation

- `34_eval_matrix_jax.py` (entry point) evaluates every run under `experiments/<paper_run_id>/<stage>/<run>` in float32 at the given contexts and datasets, and writes eval manifests plus a summary JSON and CSV.
- `80_analyze_e1_bridge_effect.py` (entry point) computes the preregistered E1 bridge effect, its paired seed bootstrap interval and the margin decision from a `34` summary.
- `18_eval_matrix.py` runs `ttt.research.eval_runner` over a run tree and writes eval manifests and a summary.
- `19_eval_aggregate.py` turns eval manifests into stage-delta and quality-against-compute tables.
- `24_eval_ruler.py` adds RULER-style retrieval metrics to eval manifests from their `eval_raw.csv`.
- `73_summarize_per_position_nll.py` plots per-position NLL curves and writes band tables from a `34` summary.
- `74_eval_niah_jax.py` scores a needle-in-a-haystack proxy with the JAX runtime.
- `81_e2a_proxy.py` is the paired retrieval proxy of the retired E2a experiment. It stays for the record.

## Tables and figures

- `20_make_paper_tables.py` writes deterministic CSV tables from run and eval manifests.
- `21_make_paper_figures.py` draws PNG figures from those tables.
- `22_make_artifact_bundle.py` packages configs, manifests, tables and figures into a tarball with a manifest.

## Warm-start safety checks

- `95_check_registry_warmstart_shapes.py` (entry point) checks without a GPU that every warm-started registry stage keeps its parent's tensor shapes.
- `84_audit_warmstart_restore.py` (entry point) restores a parent checkpoint into a child model on CPU, compares every tensor by value, and can score both models on a few validation sequences.

## Tier-1 harness

The Tier-1 targets other than the FA seed are pending a dated addendum that names revision-v3 checkpoints. Until then every runner below refuses those conditions.

- `91_run_tier1_registry.py` (entry point) runs the Tier-1 stage registry after a fail-closed preflight. Restore stages read from `--restore-source-dir` or `--restore-repo-id`.
- `85_prepare_t1a_zero_shot.py` builds the pinned T1a zero-shot task manifest.
- `86_run_t1a_zero_shot_jax.py` scores one T1a condition.
- `87_run_t1b_fast_weight_probe.py` prepares the T1b manifest or records fast-weight update norms for one condition.
- `88_prepare_t1c_ruler.py` generates and tokenizes the pinned RULER subset for T1c.
- `89_validate_t1c_ruler_harness.py` runs the T1c harness validation gate.
- `90_run_t1c_ruler_jax.py` runs T1c generation and scoring for one condition.
- `92_make_tier1_reports.py` builds the Tier-1 tables, statistics and the T1b figure.
- `93_record_tier1_session.py` appends a finished GPU session to the Tier-1 cost ledger.
- `94_export_tier1_to_hf.py` uploads finished Tier-1 artifacts to Hugging Face repos you name.

## Storage helpers for your own buckets

- `28_fetch_s3_dataset.py` pulls dataset splits from an S3-compatible bucket you name (`--bucket`, `--prefix`, `--endpoint-url`) into the layout `00` writes.
- `30_fetch_author_orbax.py` downloads the original authors' 760M checkpoints from their Requester Pays bucket. It needs `--billing-project`.
- `31_probe_author_orbax.py` checks that downloaded author checkpoints open and writes a probe report.
- `32_upload_author_orbax_to_hf.py` uploads downloaded author checkpoints to a Hugging Face repo you name.
- `33_download_author_orbax_from_hf.py` downloads them back from that repo.

## Upstream parity checks

These run the original authors' code. Clone it first:
`git clone https://github.com/test-time-training/e2e ttte2e_reference/e2e`,
or pass `--reference-root`. Without the checkout they exit with that instruction.

- `41_run_reference_125m_32k_fa_smoke.py` runs the upstream 125M 32K full-attention extension smoke from an FA seed.
- `50_run_reference_125m_32k_swa_smoke.py` runs the upstream 125M 32K sliding-window smoke that mirrors S1.
- `56_run_reference_125m_s3_pretrain_smoke.py` runs a 2-step upstream 125M TTT-E2E pretrain smoke.
- `58_run_reference_entrypoint_no_wandb.py` runs an upstream entrypoint with W&B replaced by a no-op logger. The three smokes call it.
