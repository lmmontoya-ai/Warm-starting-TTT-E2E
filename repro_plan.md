# Reproduction Plan (Code‑First Walkthrough)

This plan is a guided path to understand and reproduce TTT‑E2E by reading and running the code with intent (not just executing commands).

**Goal**
- Build a working mental model of the training/eval pipeline and then reproduce the scaled targets (TTT‑E2E vs full attention, 8K–32K scaling, NIAH gap).

**Repo Map (Entry Points + Configs)**
- Entry point: `train` → `ttt.train:main` in `pyproject.toml`.
- Hydra root config: `configs/config.yaml`.
- Experiment configs: `configs/experiment/`.
- Deploy configs: `configs/deploy/interactive.yaml` and `configs/deploy/submitit.yaml`.

**Data Pipeline (What the loader expects)**
- Loader: `ttt/dataloader/lm_dataset.py`.
- Data format: Zarr arrays with splits at `path/{split}`; each sample is `seq_len + 1` tokens.
- The code assumes data is already tokenized (Llama‑3 tokenization per repo README).
- Split names are controlled by `training.data_split` and `training.eval_split`.

**Model + Training Loop (Core Mechanics)**
- Model: `ttt/model/transformer.py`.
- Sequence modeling block is selected via `model.seq_modeling_block` (`self_attention` for full attention, `SWA` for sliding‑window attention).
- TTT‑E2E meta‑learning is controlled by `training.train_mode = meta` and `training.spec_inner` (inner‑loop trainable parameters).
- Outer loop training: `ttt/train.py`.
- Inner loop logic: `MetaModel.loss_for_sequence` in `ttt/model/transformer.py`.

**Evaluation (Built‑in vs Missing)**
- Built‑in eval logs loss and token‑level NLL: `Evaluator` in `ttt/model/loop.py`.
- Missing in repo: RULER/NIAH, latency scripts, Mamba baselines (need external implementation or custom scripts).

---

## Step‑By‑Step Coding Path

0. **Data download + storage layout**.
- Use a low‑cost CPU instance to download the tokenized datasets into a persistent volume.
- Expected buckets (per repo README): `gs://llama3-dclm-filter-8k/` and `gs://llama3-books3/`.
- Mount that volume on the GPU instance and point `deploy_paths.data.dclm_filter_8k` and `deploy_paths.data.books3` at those folders.
- Verify splits exist as Zarr arrays at `path/{split}` (e.g., `/train`, `/val`).
 - Helper: `scripts/00_data_fetch.py` (download + basic layout check).

1. Confirm entry point and config graph by reading `pyproject.toml` and `configs/config.yaml`, and by tracing how `training.dataset_path` is resolved from `deploy_paths`.
 - Helper: `scripts/01_config_inspect.py` (composes Hydra config and prints key paths).
2. Inspect dataset loader assumptions in `ttt/dataloader/lm_dataset.py`, focusing on Zarr layout, `seq_len + 1` slicing, and sharding.
 - Helper: `scripts/02_data_validate.py` (opens Zarr splits and checks sample shape).
3. Trace the model stack in `ttt/model/transformer.py`, especially `MetaModel.loss_for_sequence`, `spec_inner`, and suffix/prime layers.
4. Understand attention variants in `ttt/model/attention.py` and how `model.seq_modeling_block` switches between SWA and full attention.
5. Walk the training loop in `ttt/train.py`, including optimizer setup, sharding, and checkpointing in `ttt/infra/checkpoint.py`.
6. Review built‑in eval in `ttt/model/loop.py` and note where to hook in additional metrics.

---

## Minimal Knobs You Must Set for Any Run

**Required runtime config values**
- `training.wandb_entity`, `training.wandb_project`, `training.wandb_key`.
- `deploy_paths.data.dclm_filter_8k` and `deploy_paths.data.books3`.
- `deploy_paths.checkpoint` or `training.checkpoint_path`.

**Key experiment switches**
- Model size by experiment config, for example `+experiment=760m/pretrain/pretrain-760m-e2e`.
- TTT‑E2E vs full attention by choosing `ext-*-e2e-*` vs `ext-*-fa-*`.
- Context length via `training.seq_length` or the extension configs (32K/128K).
- Sliding window via `model.sliding_window_size` (default 8K).

---

## Reproduction Flow (Scaled 760M @ 32K)

1. Pretrain at 8K on DCLM: `+experiment=760m/pretrain/pretrain-760m-e2e`.
2. Extend to 32K on Books3: `+experiment=760m/extension/ext-760m-e2e-32K`.
3. Run full‑attention baseline: `+experiment=760m/extension/ext-760m-fa-32K`.
4. Evaluate built‑in loss and token‑NLL; add NIAH and latency scripts externally.

---

## Gaps You Must Implement Yourself

- RULER/NIAH evaluation pipeline.
- Latency benchmarking scripts.
- Mamba baseline (not implemented in repo).

---

## Next Action (Recommended)

Pick one deep‑dive track next:
- Data ingestion & Zarr layout.
- Inner‑loop mechanics (`spec_inner`, suffix blocks, prime FFNs).
- Evaluation hooks (token‑level NLL, NIAH, latency).
