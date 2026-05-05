# Contributing

Contributions are welcome when they preserve the public-release boundaries of
this repository.

## Ground Rules

- Keep generated artifacts, datasets, checkpoints, W&B runs, and cloud logs out
  of git.
- Do not add files under `og_repo/`, `ttte2e_reference/`, or `swaa_reference/`.
- Keep runtime changes covered by focused tests.
- Document any new public script in `README.md` or `docs/`.

## Local Checks

```bash
uv sync
uv run pytest -q
uv run pre-commit run --all-files
```

Before opening a pull request, run a quick safety scan:

```bash
rg -n "/Users/|AWS_SECRET|HF_TOKEN=.*[^$]|WANDB_API_KEY=.*[^$]" .
```
