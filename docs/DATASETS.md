# Datasets

The ladder reads two Llama-3-tokenized corpora:

| Dataset | Used for | Source bucket |
| --- | --- | --- |
| `dclm_filter_8k` | 8K pretraining and the 8K bridge | `gs://llama3-dclm-filter-8k` |
| `books3` | 32K extension and the main evaluation | `gs://llama3-books3` |

The TTT-E2E authors publish both, as Zarr arrays, in Requester Pays buckets on
Google Cloud Storage. This repository does not redistribute them. See the
[upstream README](https://github.com/test-time-training/e2e#dataset-download)
for their terms, and read the Books3 note in [`NOTICE.md`](../NOTICE.md).

## Download

Requester Pays means your Google Cloud project pays for every read, so you need
a project with billing enabled. Pass it with `--billing-project` or set
`GCS_BILLING_PROJECT`. The script uses `gcloud storage` (or `gsutil`), so
install the Google Cloud CLI and run `gcloud auth login` first.

The 125M ladder needs only a slice of each bucket. `--budget 125m` downloads
the first train chunks up to the 125M token budget (2.52B DCLM tokens, 126M
Books3 tokens) and the full validation splits, about 40 GB in total:

```bash
uv run --exact python scripts/00_data_fetch.py \
  --dest-root /data/ttt-e2e --budget 125m \
  --billing-project <your-gcp-project>
```

Add `--dry-run` to print the copy commands without running them. The selection
is the one used for this project's runs, and the script records it in
`<dest-root>/export_manifest.json`. `--budget 760m` applies the same rule at the
760M budget, and `--budget full` copies whole buckets, which are far larger.

The result has the layout the loaders expect:

```text
/data/ttt-e2e/dclm_filter_8k/{zarr.json,train/,val/}
/data/ttt-e2e/books3/{zarr.json,train/,val/}
```

Pass `--dclm-root /data/ttt-e2e/dclm_filter_8k --books-root /data/ttt-e2e/books3`
to the ladder scripts.

To set up several GPU hosts, download once, copy the tree to an S3-compatible
bucket you control, and pull it on each host with
`scripts/28_fetch_s3_dataset.py`. That avoids paying Requester Pays egress per
host.

## Fingerprints

The orchestrator refuses a dataset split that has no fingerprint sidecar
(`<split>.fingerprint.json`). A fingerprint records digests and token counts,
so a run manifest can prove which data it read. Generate one per split:

```bash
for dataset in dclm_filter_8k books3; do
  for split in train val; do
    uv run --exact python scripts/13_dataset_fingerprint.py \
      --dataset-id "$dataset" --path "/data/ttt-e2e/$dataset" --split "$split"
  done
done
```

`--allow-missing-fingerprints` turns the check off. Use it only for debugging,
never for a run you intend to report.

## Dataset cards

`scripts/14_dataset_card.py` turns fingerprint sidecars into a CSV and JSON
summary:

```bash
uv run --exact python scripts/14_dataset_card.py \
  --fingerprints /data/ttt-e2e/dclm_filter_8k/train.fingerprint.json \
                 /data/ttt-e2e/books3/train.fingerprint.json \
  --json-out ./reports/dataset_card.json \
  --csv-out ./reports/dataset_card.csv
```

`reports/` is gitignored.
