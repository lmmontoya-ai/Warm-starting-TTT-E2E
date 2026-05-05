# Datasets

The paper uses two language-modeling surfaces:

- `dclm_filter_8k` for short-context pretraining and bridge adaptation.
- `books3` for 32K long-context extension and primary validation.

Dataset shards are not committed to this repository. Place local tokenized data
under external paths and pass them with `--dclm-root` and `--books-root`.

## Fingerprints

The research orchestrator expects dataset fingerprint sidecars by default:

```bash
uv run --exact python scripts/13_dataset_fingerprint.py \
  --dataset-id dclm_filter_8k \
  --path /path/to/dclm_filter_8k \
  --split train
```

```bash
uv run --exact python scripts/13_dataset_fingerprint.py \
  --dataset-id books3 \
  --path /path/to/books3 \
  --split train
```

Use `--allow-missing-fingerprints` only for exploratory debugging.

## Dataset Cards

Generate a compact CSV/JSON card from fingerprint sidecars:

```bash
uv run --exact python scripts/14_dataset_card.py \
  --fingerprints /path/to/dclm_filter_8k/train.fingerprint.json /path/to/books3/train.fingerprint.json \
  --json-out ./reports/paper/dataset_card.json \
  --csv-out ./reports/paper/dataset_card.csv
```

Generated cards under `reports/` are ignored by default unless explicitly
curated for a release.
