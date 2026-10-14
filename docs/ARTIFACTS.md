# Data, checkpoints, and other artifacts

Git holds code, configs, tests, and documentation. Datasets, checkpoints, run
directories, and generated reports live outside it, and this repository does
not redistribute any of them.

## Where to get each artifact

| Artifact | Source | How |
| --- | --- | --- |
| Tokenized DCLM (8K filter) and Books3 | TTT-E2E authors (Tandon et al.) | Requester Pays GCS buckets. See [DATASETS.md](DATASETS.md). |
| Upstream TTT-E2E checkpoints (125M, 1B, 3B) | TTT-E2E authors | Their Requester Pays bucket, listed in the [upstream README](https://github.com/test-time-training/e2e#model-checkpoints). For other checkpoints from their paper, the authors ask you to email them. |
| 125M full-attention seed (`pretrain-125m-fa`, step 4799) | This project | On request from the maintainer. |
| 125M scratch TTT-E2E pretrain (`pretrain-125m-e2e`, step 4799) | This project | On request from the maintainer. |
| 760M full-attention and TTT-E2E pretrain seeds | TTT-E2E authors | The authors shared these on request. Ask them. This project does not redistribute them. |
| Revision-v3 checkpoints | This project | None exist yet. |

To request a checkpoint from this project, open a GitHub issue that says which
checkpoint you need and what you plan to do with it. The maintainer shares it
directly, as a bundle written by `scripts/40_export_stage.py --dest-dir`.

## Withdrawn checkpoints

Checkpoints from the protocol-R, revision-v2, and 760M runs are not
distributed. Every warm-started stage (S1, the 8K bridges, S2, S2-minus, the
E3 arms, and 760M S2) inherited no weights from its seed, and S0 also started
from a random initialization. The scratch S3 extensions restored correctly but
ran on a learning-rate schedule that reached its floor a quarter of the way
through, so their numbers do not compare with runs on the fixed recipe.
`PREREGISTRATION_REVISION_V3.md` (Amendment 1) has the details.

The two 125M pretrain checkpoints in the table above are not affected. Neither
one went through a warm-start restore, and each ran its full 4,800-step
learning-rate schedule.

## Local layout

Run outputs go under these roots. All of them are gitignored. The ladder
scripts take `--exp-dir` and `--checkpoint-root` to move them.

| Purpose | Default path |
| --- | --- |
| run directories, with `run_manifest.json`, `train.log` and `hydra/` | `./experiments/<paper_run_id>/<stage>/<run>/` |
| registry ladder summary | `./experiments/<paper_run_id>/registry_run_summary.json` |
| checkpoints | `./checkpoints/<exp_folder>/<run>/` |
| generated tables, figures and audits | `./reports/` |
| external model profiles | `./artifacts/external_models/<model_key>/model_profile.json` |

A warm-started stage expects its parent's checkpoint under
`./checkpoints/<exp_folder>/<parent run>/`. Requested seeds arrive as a bundle
in the stage-export layout. Extract it and install each seed with
`scripts/46_restore_stage.py --source-dir <bundle>`, setting
`--target-paper-run-id` to the ladder's paper run id. Step 3 of
[REVISION_V3_RUNBOOK.md](REVISION_V3_RUNBOOK.md) has the exact commands.

## What never goes into git

- raw or tokenized dataset shards
- Orbax checkpoints and model weights, including upstream checkpoints
- W&B run directories
- cloud provider logs, `.env` files, and machine-local manifests
- the reference snapshots `og_repo/`, `ttte2e_reference/`, and `swaa_reference/`
