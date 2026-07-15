# Tier-1 Execution Runbook

This runbook executes T1a, T1b, and T1c without training. The production
orchestrator is `scripts/91_run_tier1_registry.py`; individual scripts exist so
that a failed stage can be diagnosed, but paper runs must go through the
registry and its stage markers.

## Before renting a GPU

1. Commit the preregistration and implementation. Production preflight fails if
   `PREREGISTRATION_TIER12.md`, the machine plan, Tier-1 config, or Tier-1
   registry is untracked or dirty. This ensures the read happens after the
   reporting rules are fixed.
2. Install the pinned-capability environment:

   ```bash
   uv sync --extra tier1
   ```

3. Make the stored Books32K validation surface available. Its
   `val.fingerprint.json` sidecar is mandatory; there is no production bypass.
   On a host with the project Backblaze credentials, fetch and fingerprint only
   the validation split before renting the GPU:

   ```bash
   uv run python scripts/28_fetch_b2_dataset.py \
     --dest-root /data --datasets books3 --splits val
   uv run python scripts/13_dataset_fingerprint.py \
     --dataset-id books3 --path /data/books3 --split val \
     --tokenizer-id meta-llama/Meta-Llama-3-8B \
     --tokenizer-revision 315b20096dc791d381d514deb5f8bd9c8d6d3061
   ```

   If a curated `val.fingerprint.json` ships with the stored surface, keep it
   and verify it instead of overwriting it. A newly generated sidecar must be
   retained with the Tier-1 manifests.
4. The tokenizer comes from the public pinned
   `NousResearch/Meta-Llama-3-8B` mirror; no gated Meta model access is needed.
   Its four file hashes are preregistered. Set `HF_TOKEN` only if required for
   the public result checkpoints, which come from
   `Luxel/ttt-e2e-125m-results` and `Luxel/ttt-e2e-760m-results`. Set
   `HF_HUB_DISABLE_XET=1` if the host has the known Xet transport issue.
5. Start the T1c three-working-day clock when harness integration begins. Pass
   that date through `--started-at YYYY-MM-DD`; expiration writes a dated
   deferral note and prevents real-checkpoint scoring.

The registry downloads only experiment metadata and model weights. S2 and
S2-minus request all publicly exported model-weight milestones so T1b can
report a saved-checkpoint gradient trajectory. If only the final checkpoint is
present, the report is automatically fenced as `final_checkpoint_only`.

## CPU preparation and fail-closed dry run

Use the real Books path in both commands:

```bash
uv run python scripts/91_run_tier1_registry.py \
  --include-optional \
  --books-root /data/books3 \
  --started-at 2026-07-14
```

Without `--execute`, this only materializes the complete stage plan and writes
`reports/revision_v2/tier1/preflight/tier1_registry_preflight.json`. Every check
must say `PASS`; skipped checks are not accepted.

For a CPU-only preparation pass, execute through `T1C_MANIFEST` plus the three
manifest stages. The T1c sequence is official-generator smoke → tokenizer and
scorer gate → full 600-example manifest. Real checkpoint stages are downstream
of that gate.

## GPU session

Use one 2×H100 80GB pod. The runtime derives its mesh from
`jax.device_count()`; `--batch-size 2` gives one 32K sequence per device for
T1b/T1c. Set this for the whole session because a full-attention checkpoint can
otherwise fail during XLA autotuning:

```bash
export XLA_FLAGS=--xla_gpu_autotune_level=3
export HF_HUB_DISABLE_XET=1
uv run python scripts/91_run_tier1_registry.py \
  --execute --resume --include-optional \
  --books-root /data/books3 \
  --started-at 2026-07-14 \
  --batch-size 2
```

`--include-optional` adds the 5%/40% T1b arms and the fenced 760M RULER rows.
The 760M production stages cannot run until all three 125M RULER conditions and
a non-paper 760M checkpoint smoke have succeeded. Omit the flag to run the
required 125M evidence only.

The aggregate stage enforces identical manifest, config, scorer, and harness
hashes across compared conditions. It emits Table T1a, the T1b layer and
trajectory tables, the external RULER table/counts, paired diagnostics, result
language constrained by the preregistration, and Figure F5 through the
canonical paper plot script.

## Close-out

1. Inspect only the aggregate artifacts, then fill the pending rows in
   `research_protocol/tier1_claims_audit_template.csv` into `CLAIMS_AUDIT.md`
   (the repository file is `docs/CLAIMS_AUDIT.md`)
   before adding prose to the manuscript.
2. Dry-run and then execute `scripts/94_export_tier1_to_hf.py` to mirror reports,
   manifests, and Figure F5 to the public 125M/760M result repositories. Pass
   `--include-optional-t1b` and `--include-760m` when those fenced arms were
   actually completed.
3. Terminate/destroy the pod the same day. Record its observed duration and
   closed state with `scripts/93_record_tier1_session.py`; the script refuses a
   record that would reduce the reviewer-response reserve below $450.
4. Verify that Prime reports no active pod or that Vast reports no active
   instance. The cost record is not a substitute for provider-side teardown.
