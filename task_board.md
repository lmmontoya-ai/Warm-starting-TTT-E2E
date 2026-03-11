# Task Board

Operational execution board for the warm-start TTT-E2E study.

Scope:
- Core paper path: `125M + 760M` in-family causal ladder
- Validation path: `Qwen 0.5B`
- Appendix-only path: `SWAA`

References:
- `research_plan.md`
- `research_protocol/warmstart_preregistered_plan.yaml`
- `configs/research/warmstart_registry.yaml`
- `AGENTS.md`

## Completed

1. Probed the Requester Pays GCS Zarr layout.
   Result: direct subset export is feasible; full-bucket mirroring is unnecessary for the paper budgets.
   Evidence:
   - `scripts/29_probe_gcs_zarr.py`
   - source roots are `gs://llama3-dclm-filter-8k/data.zarr` and `gs://llama3-books3/data.zarr`
   - chunks are contiguous numeric objects under `<split>/c/<chunk_id>` with `100M` tokens per chunk

2. Froze the compact export protocol for `125M`.
   Result: deterministic contiguous-prefix chunk selection plus `val=full`, with `export_manifest.json` recorded in the package root.

3. Exported the `125M` compact package locally.
   Artifact: `~/ttt-e2e-datasets/paper_budget_125m_val-full`
   Observed size: about `38G`

4. Generated all four `125M` split fingerprints.
   Artifacts:
   - `dclm_filter_8k/train.fingerprint.json`
   - `dclm_filter_8k/val.fingerprint.json`
   - `books3/train.fingerprint.json`
   - `books3/val.fingerprint.json`

5. Built the `125M` dataset card, uploaded the package to Backblaze B2, and validated one restore.
   Result: the compact `125M` package is now durable outside the local workstation.

6. Exported the `760M` compact package locally.
   Artifact: `~/ttt-e2e-datasets/paper_budget_760m_val-full`
   Observed size: about `86G`

7. Generated all four `760M` split fingerprints.
   Artifacts:
   - `dclm_filter_8k/train.fingerprint.json`
   - `dclm_filter_8k/val.fingerprint.json`
   - `books3/train.fingerprint.json`
   - `books3/val.fingerprint.json`

8. Built the `760M` dataset card and uploaded the package to Backblaze B2.
   Result: both in-family paper-scale packages are now persisted remotely.

## Now

1. Restore the `760M` package from Backblaze B2 onto a clean destination once.
   Goal: verify the second paper-scale package is fetchable and complete before freeing local disk.
   Tool: `scripts/28_fetch_b2_dataset.py`

2. Verify author-shared `760M` FA and E2E checkpoints against the current lineage/checkpoint flow.
   Goal: confirm they can drive `S0`, `S1`, `S2`, and `S3` in the local runtime.

3. Dry-run the registry with the local runtime.
   Goal: validate stage selection, lineage, manifests, and fingerprint enforcement.
   Tool: `scripts/23_warmstart_registry.py`

4. Decide benchmark policy for the paper.
   Options:
   - keep current `NIAH` / `RULER` results explicitly proxy-only
   - implement a real `NIAH` harness before publication claims

5. Wire 125M reporting integration.
   Goal: ensure model-scoped stage IDs `S0_125M`, `S1_125M`, `S2_125M`, `S3_125M` aggregate cleanly.
   Tools: `scripts/19_eval_aggregate.py`, `scripts/20_make_paper_tables.py`, `scripts/21_make_paper_figures.py`

## Next

1. Run cheap pilots with `simulate` and `token_stats`.
   Goal: validate sequencing before expensive JAX runs.
   Tools: `scripts/06_phase1_pilot.py`, `scripts/05_phase1_report.py`

2. Run real JAX baseline-only validation for `760M`.
   Goal: prove `jax_train` / `jax_eval` are stable on `B1/B2`.
   Tool: `scripts/25_run_b1_b2_real.py`

3. Execute the full `125M` ladder.
   Stages:
   - `S0_PRETRAIN_FA_125M`
   - `S0_125M`
   - `S1_125M`
   - `S2_ADAPT_125M`
   - `S2_125M`
   - `S3_PRETRAIN_E2E_125M`
   - `S3_125M`

4. If 125M validates the story, execute the `760M` ladder.
   Stages:
   - `S0_PRETRAIN_FA`
   - `S0`
   - `S1`
   - `S2_ADAPT`
   - `S2`
   - `S3_PRETRAIN_E2E`
   - `S3`

5. Run evaluation and aggregation immediately after each run group.
   Tools:
   - `scripts/18_eval_matrix.py`
   - `scripts/19_eval_aggregate.py`
   - `scripts/20_make_paper_tables.py`
   - `scripts/21_make_paper_figures.py`
   - `scripts/22_make_artifact_bundle.py`

## Blocked

1. `760M` scale-up is blocked until author checkpoint compatibility is verified.

2. Publication-grade benchmark claims are blocked until `NIAH` / `RULER` policy is resolved.

3. Clean `125M` paper tables are blocked until reporting handles model-scoped 125M stage IDs.

4. GPU execution is blocked until the H100/Vast environment, storage, and resume strategy are ready.

5. Full-scale dataset consumption is blocked until we either:
   - derive smaller token-budget-matched dataset roots from the mirrored source data, or
   - replace the current full-array dataloader with a streaming / chunked loader for large Zarr splits.

## Later

1. Run confirmatory 3-seed subset from the prereg plan.
   Subset:
   - `125M`: `S1`, `S2@10%`, `S2@20%`, `S3`
   - `760M`: `S1`, `S2@10%`, `S3`

2. Run `Qwen 0.5B` compact cross-architecture validation.
   Tools:
   - `scripts/07_prepare_external_models.py`
   - `scripts/15_import_hf_checkpoint.py`
   - `scripts/16_audit_checkpoint_compat.py`
   - `scripts/17_probe_warmstart_init.py`

3. Add mechanistic analysis after the main ladder results exist.
   Candidates:
   - per-position NLL
   - CKA
   - prime weight divergence
   - gradient flow during adaptation

4. Run `Qwen 1.5B` only if `Qwen 0.5B` is successful and budget remains.

5. Run `SWAA` only as appendix work after the core paper path is complete.

## Notes

- The preregistration now treats the full sweep as exploratory single-seed work and the reduced subset as confirmatory 3-seed work.
- The live registry now covers both `in_family_125m` and `in_family_760m`.
- 125M warm-start configs live under `configs/experiment/125m/pretrained/`.
- The storage plan and the runnable-data plan are not the same thing: mirroring `DCLM` / `Books3` to B2 solves access and persistence, but not the current loader's memory behavior.
- `scripts/13_dataset_fingerprint.py` now fingerprints exported Zarr roots by streaming split metadata and chunk bytes, instead of loading full token lists into RAM.
