# Warm-starting TTT-E2E

TTT-E2E ([Tandon et al.](https://test-time-training.github.io/e2e.pdf)) is a
sliding-window Transformer that keeps learning at test time. It compresses the
context it reads into the weights of small "prime" MLPs. The published recipe
trains it from scratch. This repository asks whether you can skip that:
take a short-context full-attention Transformer you already have, convert it
to TTT-E2E, adapt it briefly at 8K, and extend it to 32K.

The code is a JAX implementation of the full experiment ladder: conversion,
warm-start restore with value checks, training, checkpoint evaluation, and a
preregistered decision procedure.

## Status

There are no results yet.

Earlier versions of this repository reported 125M and 760M results. Those
results are withdrawn. A bug in the restore path meant that no "warm-started"
model inherited any weights from its seed. Separately, every 32K extension ran
on a learning-rate schedule that hit its floor a quarter of the way through.
[`PREREGISTRATION_REVISION_V3.md`](PREREGISTRATION_REVISION_V3.md)
(Amendment 1) documents both bugs, the fixes, and the revised run design.

The trainer now refuses both failure modes. It compares every restored tensor
with the checkpoint by value, and it rejects a learning-rate schedule that does
not fit the run.

The revision-v3 runs have not been launched.

## The comparison

| Stage | Path |
| --- | --- |
| S0 | full-attention seed, extended directly to 32K |
| S1 | full-attention seed, converted to sliding-window attention, extended to 32K |
| S2 | full-attention seed, converted to TTT-E2E, adapted at 8K (the bridge), extended to 32K |
| S2-minus | like S2 without the 8K bridge |
| S3 | TTT-E2E trained from scratch at 8K, extended to 32K |
| S3W | S3 at S2's parameter count |

The registry in
[`configs/research/warmstart_registry.yaml`](configs/research/warmstart_registry.yaml)
defines every stage, its parent, and its data. The decision rules (the bridge
effect S2 minus S2-minus, a `0.10` loss margin, a paired seed bootstrap) are
fixed in the preregistrations before any run.

## Install

You need [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone https://github.com/lmmontoya-ai/Warm-starting-TTT-E2E.git
cd Warm-starting-TTT-E2E
uv sync
```

On Linux, `uv sync` installs JAX with CUDA 12. On macOS and Windows it installs
CPU JAX, which is enough for the tests and the quick start.

## Quick start on CPU

Two commands run the whole stage ladder on synthetic tokens:

```bash
uv run --exact python scripts/04_make_token_data.py --out /tmp/warmstart_tokens
uv run --exact python scripts/23_warmstart_registry.py \
  --paper-run-id warmstart_smoke --exp-folder warmstart_smoke \
  --dclm-root /tmp/warmstart_tokens --books-root /tmp/warmstart_tokens \
  --runtime-mode token_stats
```

The first writes 70,000 random tokens per split, with fingerprint sidecars.
The second runs all 24 stages in the registry. The `token_stats` runtime swaps
the network for a token-frequency model, so no real model trains. What gets
tested is the plumbing: stage order, the parent checks, checkpoint hand-offs,
and the run manifests. It takes a few minutes on a CPU.

Each stage writes its run directory to
`experiments/warmstart_smoke/<stage>/<run>/` and its checkpoint to
`checkpoints/warmstart_smoke/<run>/`. The summary is
`experiments/warmstart_smoke/registry_run_summary.json`.

A few rules:

- `--paper-run-id` and `--exp-folder` are folder names, not paths. Set the
  locations with `--exp-dir` and `--checkpoint-root`.
- Running the ladder again under the same ids needs `--skip-existing`, or new
  ids.
- `--runtime-mode simulate` runs the same ladder with made-up losses and
  reads no data.
- On Windows, `/tmp/warmstart_tokens` resolves to `\tmp` on the current drive.
  Any other folder works too.

## Running the real experiments

The 125M ladder needs an 8x H100 80GB host. The Phase 1 pilot in Amendment 1
is about 16 GPU-hours of training. [`docs/REVISION_V3_RUNBOOK.md`](docs/REVISION_V3_RUNBOOK.md)
walks through it: the restore audit, a smoke run, the pilot, evaluation, and
the gates that decide whether Phase 2 runs.

## Data and checkpoints

Nothing large is in git, and this repository redistributes no data or weights.

- **Datasets.** The Llama-3-tokenized DCLM (8K filter) and Books3 sets come
  from the TTT-E2E authors' Requester Pays buckets on Google Cloud Storage.
  [`docs/DATASETS.md`](docs/DATASETS.md) shows how to download the subset the
  125M ladder uses.
- **Seeds.** The 125M full-attention seed and the 125M scratch TTT-E2E
  pretrain were trained in this project. Open an issue to request them. The
  760M seeds and any other upstream checkpoint come from the TTT-E2E authors.
- **Trained models.** None are published. Revision-v3 checkpoints do not exist
  yet, and the earlier ones are defective.

[`docs/ARTIFACTS.md`](docs/ARTIFACTS.md) has the details and the local
directory layout.

## Repository layout

| Path | Contents |
| --- | --- |
| `ttt/` | runtime: JAX model, trainer, evaluator, checkpointing, warm-start and schedule guards, research orchestration |
| `configs/` | Hydra configs for models, training, experiments, deploy profiles, and the stage registry |
| `scripts/` | command-line entry points, indexed in [`scripts/README.md`](scripts/README.md) |
| `tests/` | CPU tests |
| `research_protocol/` | machine-readable preregistered plans |
| `docs/` | runbook, datasets, artifacts, reproducibility notes |
| `PREREGISTRATION_*.md` | dated preregistrations. Revision v3 is the active one. |

## Tests

```bash
uv run pytest -q
```

The suite runs on CPU in a few minutes. CI runs it on every push.

## Citation

[`CITATION.cff`](CITATION.cff) describes how to cite this code. If you use
TTT-E2E itself, cite Tandon et al.

## License

MIT. See [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md) for third-party
data and code.
