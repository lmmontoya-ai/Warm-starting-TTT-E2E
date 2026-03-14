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

9. Restored and validated the `760M` package from Backblaze B2.
   Result: the durable storage path is now verified for both `125M` and `760M`.

10. Fetched both author-shared `760M` Orbax checkpoints into standardized local raw-artifact roots.
    Artifacts:
    - `artifacts/author_checkpoints/760m_fa`
    - `artifacts/author_checkpoints/760m_e2e`
    Result: requester-pays GCS intake is working and manifests are recorded locally.

11. Probed both author checkpoints with Orbax restore.
    Artifacts:
    - `artifacts/author_checkpoints/760m_fa/probe_report.json`
    - `artifacts/author_checkpoints/760m_e2e/probe_report.json`
    Result: the raw author checkpoints are readable as Orbax `model_weights` items.
    Caveat: this validates artifact transport/readability, not local runtime warm-start parity.

12. Uploaded the standardized author checkpoint artifacts to Hugging Face and verified a round-trip download.
    Repo:
    - `Luxel/ttt-e2e-author-760m-orbax`
    Result: the remote HF copy matches the local artifact tree aside from local `.DS_Store` and HF cache metadata.

13. Added the first-class 125M parity eval path and 125M launch wrapper.
    Tools:
    - `scripts/34_eval_matrix_jax.py`
    - `scripts/35_run_125m_ladder.py`
    Result: `125M` can now use `jax_eval` as the authoritative paper eval path instead of the old proxy external-eval flow.

14. Updated reporting and bundling for model-scoped 125M stages and parity artifacts.
    Tools:
    - `scripts/19_eval_aggregate.py`
    - `scripts/20_make_paper_tables.py`
    - `scripts/22_make_artifact_bundle.py`
    Result: `S0_125M/S1_125M/S2_125M/S3_125M` now flow cleanly into paper tables, and parity eval artifacts are included in the bundle.

15. Extended `jax_eval` monitoring outputs.
    Result:
    - `metrics.jsonl` now records eval wall time and coarse per-position NLL summaries
    - W&B eval logging now includes checkpoint step, eval wall time, and NLL head/tail summaries

16. Validated the new 125M dry-run path end to end.
    Evidence:
    - `scripts/23_warmstart_registry.py --dry-run`
    - `scripts/34_eval_matrix_jax.py --dry-run`
    - `scripts/20_make_paper_tables.py` with 125M stage overrides
    Result: the registry, parity eval, and stage-aware table generation all execute successfully together.

17. Added targeted 32K-FA OOM diagnosis tooling.
    Tools:
    - `scripts/40_diagnose_125m_32k_fa_oom.py`
    - `scripts/41_run_reference_125m_32k_fa_smoke.py`
    Result:
    - the local runtime now has a deterministic compile-memory harness that records resolved config, tree/sharding summaries, lowered IR, compile logs, and compile/execute results
    - the reference smoke now has a one-command wrapper for params-only restore from the FA 8K seed

18. Confirmed the 125M 32K FA extension OOM is not specific to H100.
    Evidence:
    - `artifacts/prime_smokes/prime_125m_a100_32k_fa_smoke_20260312a/a100_32k_fa_smoke.log`
    - prior `8x H100 80GB` smoke logs under `artifacts/prime_smokes/`
    Result:
    - both `8x H100 80GB` and `8x A100 80GB` hit the same late compile-time allocation at ~`103.48 GB` per GPU
    - this remains a runtime-parity issue until the reference smoke proves otherwise

19. Added a deterministic local-vs-reference graph-boundary comparison report for the failing 125M 32K FA path.
    Tools:
    - `scripts/42_compare_local_reference_125m_32k_fa.py`
    Artifact:
    - `artifacts/oom_diagnosis/compare_125m_32k_fa_20260312T201949Z/`
    Result:
    - experiment and deploy configs are confirmed identical between local and reference
    - model-sharding config is not the primary gap
    - the highest-probability remaining culprits are still in the local train loop, batch loading path, and remat utility surface

20. Froze the `125M` dual-protocol framing for the paper.
    Artifacts:
    - `docs/125m_dual_protocol.md`
    - `docs/125m_ladder_runbook.md`
    Result:
    - Protocol F is now the faithful feasibility result
    - Protocol R is now the smallest explicit runnable revision, defined as an extension-only batch-size change with token-budget preservation

21. Added first-class Protocol R execution surfaces.
    Tools:
    - `scripts/44_search_reference_125m_32k_fa_batch.py`
    - `scripts/45_gate_local_125m_protocol_r.py`
    - `scripts/35_run_125m_ladder.py` with extension-only batch override and protocol manifest
    Result:
    - the reference batch search is now scripted
    - the local Protocol R acceptance gate is now scripted
    - the full ladder launcher can now freeze and record a revised matched protocol

22. Froze the `125M` Protocol R batch size on reference hardware.
    Result:
    - reference `32K` FA extension search selected `B*=8`
    - `32`, `24`, and `16` OOMed on `8x H200 141GB`
    - `8` passed the reference train-fit gate
    Evidence:
    - `artifacts/protocol_r/reference_batch_search_20260312T231848Z/reference_protocol_r_batch_search.json`

23. Passed the clean local Protocol R gate on `8x H200`.
    Result:
    - local `2`-step smoke passed at `global_batch_size=8`
    - local parity eval passed
    - local `8`-step benchmark passed
    Evidence:
    - `reports/paper/protocol_r_local_gate_h200_b8_clean/protocol_r_local_gate_summary.json`

## Now

1. Launch the full `125M` ladder under frozen Protocol R.
   Goal: complete the first fully runnable, paper-defensible in-family ladder.
   Frozen protocol:
   - `--protocol revised`
   - `--ext-global-batch-size 8`
   - `--preserve-ext-token-budget`
   - `--base-ext-global-batch-size 32`
   Status:
   - reference batch search selected `B*=8`
   - clean local H200 gate passed
   Tools:
   - `scripts/35_run_125m_ladder.py`
   - `scripts/39_prime_125m_ladder_controller.py`
   - `scripts/40_export_stage_to_hf.py`

2. Execute the `125M` split lineages independently under the canonical `protocol_r_125m_main_v1` lineage.
   Goal: keep progress moving while the `32K` SWA path is diagnosed separately.
   Current split:
   - `h200_s0`: `S0_125M`
   - `h100_b`: `S2_ADAPT_125M`, `S3_PRETRAIN_E2E_125M`
   - `h200_s1_diag`: reference + local `S1_125M` diagnostics, then full `S1_125M` only if safe
   - `h200_c`: `S2_125M`, `S3_125M`

3. Validate stage durability during the real Protocol R run.
   Goal: avoid losing completed stages or filling the workstation disk during long ladder runs.
   Current status:
   - the old controller mirrored full checkpoint histories locally and still left the final FA checkpoint incomplete
   - the controller now mirrors only the latest checkpoint snapshots locally, keeping one fallback step
   - completed stages are now exported directly from the pod to HF via `scripts/40_export_stage_to_hf.py`
   - next acceptance gate is a successful HF export of one completed stage during the real Protocol R run

4. Validate the rebased `jax_runtime` against author checkpoints and paper configs.
   Goal: confirm the new in-repo parity runtime can consume raw author Orbax checkpoints and resolved `125M` / `760M` configs without shape hacks.
   Current status:
   - Orbax is now the native local checkpoint format
   - the parity runtime writes `latest.json` + step metadata sidecars
   - tiny `jax_train` -> `jax_eval` smoke path is working end-to-end
   - direct CE parity, split-state parity, and SWA state-update unit tests are now green
   - author-checkpoint transport artifacts remain the source of truth

5. Decide benchmark policy for the paper.
   Options:
   - keep current `NIAH` / `RULER` results explicitly proxy-only
   - implement a real `NIAH` harness before publication claims

6. Configure W&B in the execution environment.
   Goal: make the first real 125M launch observable through the standard group naming convention `paper_run_id/stage_id/run_id/runtime_mode`.
   Required env vars:
   - `WANDB_API_KEY`
   - `WANDB_ENTITY`
   - `WANDB_PROJECT`

## Next

1. If `GB200/B200` hardware appears, run exactly one faithful reference gate there.
   Goal: preserve the Blackwell-faithful branch without waiting indefinitely.
   Tool: `scripts/41_run_reference_125m_32k_fa_smoke.py`

2. If `125M` validates the story under Protocol R, execute the `760M` ladder.
   Stages:
   - `S0_PRETRAIN_FA`
   - `S0`
   - `S1`
   - `S2_ADAPT`
   - `S2`
   - `S3_PRETRAIN_E2E`
   - `S3`

3. Run parity evaluation and aggregation immediately after each run group.
   Tools:
   - `scripts/34_eval_matrix_jax.py`
   - `scripts/19_eval_aggregate.py`
   - `scripts/20_make_paper_tables.py`
   - `scripts/21_make_paper_figures.py`
   - `scripts/22_make_artifact_bundle.py`

## Blocked

1. `760M` scale-up is blocked on parity validation of the rebased `jax_runtime`, not on raw artifact transport anymore.

2. Publication-grade benchmark claims are blocked until `NIAH` / `RULER` policy is resolved.

3. Large-pod execution remains economically sensitive; on-demand `8x H200` is the current viable lane, while `GB200/B200` remains opportunistic rather than required for Protocol R.

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
- `jax_runtime` now uses a Zarr + Grain dataloader path for `jax_train` / `jax_eval`; `simulate` and `token_stats` still use the lighter phase-1 iterator path.
- `scripts/13_dataset_fingerprint.py` now fingerprints exported Zarr roots by streaming split metadata and chunk bytes, instead of loading full token lists into RAM.
- The first-class `125M` paper eval path is now `scripts/34_eval_matrix_jax.py`, not `scripts/18_eval_matrix.py`.
- Full-ladder dry-run now tolerates missing parent checkpoints by using placeholder lineage refs only in `dry_run`; real runs remain strict.
