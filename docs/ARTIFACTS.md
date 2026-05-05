# Artifacts

This release keeps large artifacts outside git. The repository tracks code,
configs, tests, documentation, paper figures, and curated small result summaries
only.

## Not Committed

- raw or tokenized dataset shards
- Orbax checkpoints and model weights
- author-provided upstream checkpoints
- W&B run directories
- cloud execution logs and local machine manifests
- reference snapshots: `og_repo/`, `ttte2e_reference/`, `swaa_reference/`

## Expected Local Layout

Use these default local roots when running full experiments:

| Purpose | Default/local example |
| --- | --- |
| generated runs | `./experiments/<paper_run_id>/` |
| checkpoints | `./checkpoints/<exp_folder>/` |
| generated reports | `./reports/paper/<paper_run_id>/` |
| external model profiles | `./artifacts/external_models/<model_key>/model_profile.json` |
| author/import checkpoints | external storage, restored under `./artifacts/author_checkpoints/` if needed |

All of these paths are ignored by default.

## Public Result Artifacts

Curated paper outputs that remain tracked are small enough for git and are used
for figure/table regeneration. They include:

- final manuscript PDFs in `paper/plots/figures/`
- 125M and 760M scalar summaries under `reports/paper/protocol_r_*`
- final plot data and rendered figures under `reports/paper/warmstart_paper_v1/`

If a future run produces a large or private artifact, upload it to external
storage and document the retrieval instructions here rather than committing it.
