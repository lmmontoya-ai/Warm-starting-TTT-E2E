# Contributing

Issues and pull requests are welcome. Open an issue first for anything larger
than a bug fix, so we can agree on the approach before you write it.

## Setup and checks

```bash
uv sync
uv run pytest -q
uv run pre-commit run --all-files
```

The tests run on CPU. CI runs the same two commands on every push and pull
request.

## Rules

- Keep datasets, checkpoints, W&B runs, generated reports, and cloud logs out
  of git. `docs/ARTIFACTS.md` lists what stays out.
- Never commit credentials. `SECURITY.md` has a pre-push scan.
- Cover runtime changes with focused tests.
- Add any new script to `scripts/README.md`.
- Do not add files under `og_repo/`, `ttte2e_reference/`, or `swaa_reference/`.
  They are local reference snapshots and stay untracked.
- Do not edit a preregistration after the fact. Add a dated amendment instead.
