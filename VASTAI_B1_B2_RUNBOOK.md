# B1/B2 Baseline Runs on Vast.ai 8xH100 — Complete Runbook

## 1. Goal

Run the two foundational baselines for the warm-start TTT-E2E research paper using the **original reference code** (`ttte2e_reference/e2e/`):

| Baseline | Pipeline | What it proves |
|----------|----------|----------------|
| **B2** (FA baseline) | `pretrain-125m-fa` (8K) → `ext-125m-fa-32K` (32K) | Quality ceiling with full attention |
| **B1** (TTT-E2E from scratch) | `pretrain-125m-e2e` (8K) → `ext-125m-e2e-32K` (32K) | TTT-E2E gold standard to compare warm-start against |

Both use the 125M model architecture with the Llama-3 tokenizer (vocab=128,256).

---

## 2. What the Reference Code Does

The reference code lives at `ttte2e_reference/e2e/`. It is a complete JAX/Equinox training framework.

### Architecture (125M)
- 12 transformer blocks, hidden_size=768, 12 attention heads
- SwiGLU MLPs with RMSNorm (pre + post)
- RoPE positional encoding (theta=500,000)
- Tied word embeddings (vocab=128,256)
- For E2E: last 3 blocks are "suffix blocks" with a second "prime" MLP that gets updated during test-time training. `intermediate_size` is reduced from 2048 to 1664 to match parameter count.

### Training modes
- **`pretrain` mode**: Standard next-token prediction. Forward → loss → backward → optimizer step. Uses FlashAttention when `force_flash=True`.
- **`meta` mode**: Meta-learning through the inner loop. The sequence is split into 1024-token mini-batches. For each mini-batch, the prime MLP params are updated via SGD (inner loop). The outer loss is the average loss across all mini-batches AFTER inner updates. Gradients flow through the inner loop via `jax.value_and_grad` (gradients-of-gradients). This is 3.4x slower than pretrain mode at 8K context.

### Data format
- **Zarr arrays** on disk. The dataloader reads from `/{split}` path inside a Zarr store.
- Each dataset directory (e.g., `llama3-dclm-filter-8k/`) contains Zarr arrays with tokenized data.
- The data is pre-tokenized with the Llama-3 tokenizer.
- The dataloader uses Google Grain for batching, sharding, and shuffling.

### Checkpoint format
- Uses **Orbax** (`orbax-checkpoint`) for checkpoint management.
- Saves: model weights, optimizer state, data iterator state.
- Resume: `load_part=params` loads only model weights (for warm-start), `load_part=all` loads everything (for crash resume).

### Multi-GPU
- Pure data parallelism via JAX named sharding mesh.
- `interactive.yaml` sets `num_devices: 8` (single node).
- Each GPU gets `global_batch_size / 8` samples.

---

## 3. Experiment Configs (Exact Values)

### B2: Full-Attention Baseline

**Stage 1: Pretrain FA at 8K** (`pretrain-125m-fa`)
```
train_mode: pretrain
seq_modeling_block: self_attention (full attention)
force_flash: True
seq_length: 8192
global_batch_size: 64  (8 per GPU)
total_steps: 4800
mini_batch_size: 8192  (whole sequence = 1 chunk)
rope_theta: 500000
dataset: dclm_filter_8k (DCLM, filtered docs >8K tokens)
optimizer: AdamW (lr=3e-3, warmup=480, decay=4800, b1=0.9, b2=0.95, wd=0.1, clip=1.0)
Total tokens: 4800 * 64 * 8192 = 2.52B tokens
```

**Stage 2: Extend FA to 32K** (`ext-125m-fa-32K`)
```
train_mode: pretrain
seq_modeling_block: self_attention
force_flash: True
seq_length: 32768
global_batch_size: 32  (4 per GPU)
total_steps: 120
mini_batch_size: 32768
rope_theta: 2000000  (extended for 32K)
dataset: books3
optimizer: AdamW (lr=4e-4, warmup=12, decay=120, wd=0.1, clip=1.0)
load_part: params (from pretrain-125m-fa checkpoint)
Total tokens: 120 * 32 * 32768 = 125.8M tokens
```

### B1: TTT-E2E From Scratch

**Stage 1: Pretrain E2E at 8K** (`pretrain-125m-e2e`)
```
train_mode: meta
seq_modeling_block: SWA (sliding window attention)
sliding_window_size: 8192
force_flash: False
seq_length: 8192
global_batch_size: 64
total_steps: 4800
mini_batch_size: 1024  (8 chunks per sequence for inner loop)
rope_theta: 500000
prime: True
suffix_len: 3  (last 3 of 12 blocks have prime MLPs)
intermediate_size: 1664  (reduced from 2048 for param matching)
dataset: dclm_filter_8k
inner optimizer: SGD (lr=1.0, clip=1.0)
ilr_init: 0.1  (inner LR warmup from 0.1 to 1.0)
ilr_warmup_steps: 480
spec_inner: ["language_model.**.suffix_blocks.feed_forward_prime.**"]
outer optimizer: AdamW (lr=3e-3, warmup=480, decay=4800, b1=0.9, b2=0.95, wd=0.1, clip=1.0)
Total tokens: 2.52B tokens (same as FA, but 3.4x slower to process)
```

**Stage 2: Extend E2E to 32K** (`ext-125m-e2e-32K`)
```
train_mode: meta
seq_modeling_block: SWA
sliding_window_size: 8192
seq_length: 32768
global_batch_size: 32
total_steps: 120
mini_batch_size: 1024  (32 chunks per sequence)
prime: True, suffix_len: 3, intermediate_size: 1664
ilr_init: 1  (no warmup for extension)
dataset: books3
load_part: params (from pretrain-125m-e2e checkpoint)
Total tokens: 125.8M tokens
```

---

## 4. Time Estimates (8xH100 SXM)

| Stage | Tokens | Mode | Est. time |
|-------|-------:|:----:|----------:|
| pretrain-125m-fa | 2.52B | pretrain | ~50 min |
| ext-125m-fa-32K | 126M | pretrain | ~5 min |
| pretrain-125m-e2e | 2.52B | meta (3.4x) | ~2.8 hrs |
| ext-125m-e2e-32K | 126M | meta (~2x at 32K) | ~5 min |
| **Total training** | | | **~4 hrs** |
| XLA compilation (first step of each stage) | | | ~5-15 min/stage |
| Evaluation (end of each ext stage) | | | ~10-20 min |
| **Total wall-clock** | | | **~5-6 hrs** |

Add data download + setup: **~6-8 hours total** for the full session.

---

## 5. Data Preparation (Do This BEFORE Renting the Instance)

The reference code expects pre-tokenized Zarr arrays. The original authors host them on Google Cloud Storage:

```bash
# DCLM (pre-training data, filtered for docs >8K tokens)
gcloud storage cp -r gs://llama3-dclm-filter-8k/ llama3-dclm-filter-8k

# Books3 (extension data)
gcloud storage cp -r gs://llama3-books3/ llama3-books3
```

> **Note**: These buckets may be Requester Pays. You need a GCP account with billing enabled. See https://docs.cloud.google.com/storage/docs/requester-pays

### Strategy: Stage data to HuggingFace Hub first
To avoid slow downloads on Vast.ai:

1. Download datasets to a local machine or cheap cloud instance with GCP access.
2. Upload to a private HuggingFace dataset repo:
   ```bash
   huggingface-cli upload your-username/ttt-e2e-data ./llama3-dclm-filter-8k --repo-type dataset
   huggingface-cli upload your-username/ttt-e2e-data ./llama3-books3 --repo-type dataset
   ```
3. On the Vast.ai instance, download from HF (datacenter speeds):
   ```bash
   huggingface-cli download your-username/ttt-e2e-data --repo-type dataset --local-dir /data/ttt-e2e-data
   ```

### Alternative: Use `--dummy-dataset` for pipeline validation
Before spending money on real data, validate the full pipeline with dummy data:
```bash
# Add to any training command:
training.dummy_dataset=true
```
This generates random token sequences and runs through the full training loop. Use this to verify:
- Environment is set up correctly
- All 8 GPUs are visible and utilized
- Checkpointing works
- W&B logging works

---

## 6. Vast.ai Instance Setup

### Instance requirements
- **GPU**: 8x H100 SXM 80GB (or 8x H100 NVL)
- **RAM**: >=256GB system RAM
- **Disk**: >=500GB SSD (for datasets + checkpoints)
- **Image**: CUDA 12.4+ with Python 3.12
- **Networking**: High bandwidth for data download

### Recommended Vast.ai template search
Search for: `8x H100`, sort by price. Expect ~$20-28/hr.

Preferred Docker image: `nvidia/cuda:12.4.0-devel-ubuntu22.04` or any image with CUDA 12.x and Python 3.12.

---

## 7. Instance Bootstrap Script

Run this after SSH-ing into the Vast.ai instance. Save as `setup.sh`:

```bash
#!/bin/bash
set -euo pipefail

echo "=== [1/7] System packages ==="
apt-get update && apt-get install -y git curl build-essential

echo "=== [2/7] Install uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "=== [3/7] Clone repo ==="
cd /workspace
git clone https://github.com/YOUR_USERNAME/Reproducing-TTT-E2E.git
cd Reproducing-TTT-E2E/ttte2e_reference/e2e

echo "=== [4/7] Install Python dependencies ==="
uv sync

echo "=== [5/7] Verify JAX sees all GPUs ==="
uv run python -c "
import jax
print(f'JAX version: {jax.__version__}')
print(f'Devices: {jax.devices()}')
print(f'Device count: {jax.device_count()}')
assert jax.device_count() == 8, f'Expected 8 GPUs, got {jax.device_count()}'
print('All 8 GPUs detected.')
"

echo "=== [6/7] Download data ==="
# Option A: From HuggingFace (recommended)
pip install huggingface_hub
huggingface-cli login --token $HF_TOKEN
mkdir -p /data
huggingface-cli download YOUR_USERNAME/ttt-e2e-data --repo-type dataset --local-dir /data/ttt-e2e-data

# Option B: From GCS directly (slower, requires gcloud)
# gcloud storage cp -r gs://llama3-dclm-filter-8k/ /data/llama3-dclm-filter-8k
# gcloud storage cp -r gs://llama3-books3/ /data/llama3-books3

echo "=== [7/7] Verify data ==="
python -c "
import zarr, zarr.storage
for name, split in [('dclm-filter-8k', 'train'), ('books3', 'train')]:
    path = f'/data/ttt-e2e-data/llama3-{name}'
    store = zarr.storage.LocalStore(path, read_only=True)
    arr = zarr.open_array(store, path=f'/{split}')
    print(f'{name}/{split}: shape={arr.shape}, dtype={arr.dtype}')
print('Data verified.')
"

echo "=== Setup complete ==="
```

---

## 8. Running the Experiments

All commands are run from inside `ttte2e_reference/e2e/`.

### 8.1 Validate with dummy data first (~10 min)

```bash
cd /workspace/Reproducing-TTT-E2E/ttte2e_reference/e2e

# Quick FA pretrain smoke test (dummy data, 10 steps)
uv run --exact train \
  +deploy=interactive \
  +experiment=125m/pretrain/pretrain-125m-fa \
  training.exp_folder=smoke \
  training.exp_name=smoke-fa \
  training.checkpoint_path=/workspace/checkpoints \
  training.total_steps=10 \
  training.save_milestone_freq=5 \
  training.dummy_dataset=true \
  training.log_wandb=false \
  deploy_paths.data.dclm_filter_8k=/tmp/dummy \
  deploy_paths.data.books3=/tmp/dummy \
  training.wandb_entity=none \
  training.wandb_project=none \
  training.wandb_key=none

echo "FA smoke test passed."

# Quick E2E pretrain smoke test (dummy data, 10 steps)
uv run --exact train \
  +deploy=interactive \
  +experiment=125m/pretrain/pretrain-125m-e2e \
  training.exp_folder=smoke \
  training.exp_name=smoke-e2e \
  training.checkpoint_path=/workspace/checkpoints \
  training.total_steps=10 \
  training.save_milestone_freq=5 \
  training.dummy_dataset=true \
  training.log_wandb=false \
  deploy_paths.data.dclm_filter_8k=/tmp/dummy \
  deploy_paths.data.books3=/tmp/dummy \
  training.wandb_entity=none \
  training.wandb_project=none \
  training.wandb_key=none

echo "E2E smoke test passed."
```

**What to check**: Both runs should complete 10 steps, log loss values, and save checkpoints. The E2E run should be noticeably slower per step due to meta-learning overhead.

### 8.2 B2: Full-Attention Baseline (real data)

```bash
# Set common paths
export DATA_ROOT=/data/ttt-e2e-data
export CKPT_ROOT=/workspace/checkpoints
export WANDB_ENTITY=your-wandb-entity    # or "none" to disable
export WANDB_PROJECT=ttt-e2e-125m
export WANDB_KEY=your-wandb-api-key      # or "none"

# --- B2 Stage 1: FA Pretrain at 8K ---
uv run --exact train \
  +deploy=interactive \
  +experiment=125m/pretrain/pretrain-125m-fa \
  training.exp_folder=b1_b2 \
  training.exp_name=pretrain-125m-fa \
  training.checkpoint_path=$CKPT_ROOT \
  training.save_milestone_freq=500 \
  deploy_paths.data.dclm_filter_8k=$DATA_ROOT/llama3-dclm-filter-8k \
  deploy_paths.data.books3=$DATA_ROOT/llama3-books3 \
  training.wandb_entity=$WANDB_ENTITY \
  training.wandb_project=$WANDB_PROJECT \
  training.wandb_key=$WANDB_KEY

# --- B2 Stage 2: FA Extension to 32K ---
uv run --exact train \
  +deploy=interactive \
  +experiment=125m/extension/ext-125m-fa-32K \
  training.exp_folder=b1_b2 \
  training.exp_name=ext-125m-fa-32K \
  training.checkpoint_path=$CKPT_ROOT \
  training.save_milestone_freq=30 \
  training.load_part=params \
  training.resume_exp_name=pretrain-125m-fa \
  deploy_paths.data.dclm_filter_8k=$DATA_ROOT/llama3-dclm-filter-8k \
  deploy_paths.data.books3=$DATA_ROOT/llama3-books3 \
  training.wandb_entity=$WANDB_ENTITY \
  training.wandb_project=$WANDB_PROJECT \
  training.wandb_key=$WANDB_KEY
```

### 8.3 B1: TTT-E2E From Scratch (real data)

```bash
# --- B1 Stage 1: E2E Pretrain at 8K (this is the expensive one: ~2.8 hrs) ---
uv run --exact train \
  +deploy=interactive \
  +experiment=125m/pretrain/pretrain-125m-e2e \
  training.exp_folder=b1_b2 \
  training.exp_name=pretrain-125m-e2e \
  training.checkpoint_path=$CKPT_ROOT \
  training.save_milestone_freq=200 \
  deploy_paths.data.dclm_filter_8k=$DATA_ROOT/llama3-dclm-filter-8k \
  deploy_paths.data.books3=$DATA_ROOT/llama3-books3 \
  training.wandb_entity=$WANDB_ENTITY \
  training.wandb_project=$WANDB_PROJECT \
  training.wandb_key=$WANDB_KEY

# --- B1 Stage 2: E2E Extension to 32K ---
uv run --exact train \
  +deploy=interactive \
  +experiment=125m/extension/ext-125m-e2e-32K \
  training.exp_folder=b1_b2 \
  training.exp_name=ext-125m-e2e-32K \
  training.checkpoint_path=$CKPT_ROOT \
  training.save_milestone_freq=30 \
  training.load_part=params \
  training.resume_exp_name=pretrain-125m-e2e \
  deploy_paths.data.dclm_filter_8k=$DATA_ROOT/llama3-dclm-filter-8k \
  deploy_paths.data.books3=$DATA_ROOT/llama3-books3 \
  training.wandb_entity=$WANDB_ENTITY \
  training.wandb_project=$WANDB_PROJECT \
  training.wandb_key=$WANDB_KEY
```

---

## 9. Checkpoint Backup to HuggingFace Hub

Vast.ai instances can be preempted. Checkpoints MUST be backed up externally.

### Automatic backup script

Save as `backup_checkpoints.sh` and run in a `tmux` background pane:

```bash
#!/bin/bash
# Run this in a separate tmux pane: tmux new -s backup
# It watches the checkpoint directory and uploads new checkpoints to HF Hub.

set -euo pipefail

HF_REPO="YOUR_USERNAME/ttt-e2e-125m-checkpoints"
CKPT_ROOT="/workspace/checkpoints/b1_b2"
UPLOAD_INTERVAL=300  # seconds between checks

pip install huggingface_hub 2>/dev/null

echo "Watching $CKPT_ROOT for new checkpoints..."

LAST_UPLOADED=""

while true; do
    # Find the most recent checkpoint directory
    LATEST=$(find $CKPT_ROOT -name "default" -type d 2>/dev/null | sort | tail -1)

    if [ -n "$LATEST" ] && [ "$LATEST" != "$LAST_UPLOADED" ]; then
        STAGE_DIR=$(dirname $(dirname "$LATEST"))
        STAGE_NAME=$(basename "$STAGE_DIR")
        echo "$(date): Uploading checkpoint for $STAGE_NAME..."

        huggingface-cli upload "$HF_REPO" \
            "$STAGE_DIR" \
            "checkpoints/$STAGE_NAME" \
            --repo-type model \
            --quiet \
            || echo "Upload failed, will retry next cycle"

        LAST_UPLOADED="$LATEST"
        echo "$(date): Upload complete for $STAGE_NAME"
    fi

    sleep $UPLOAD_INTERVAL
done
```

### Manual backup after each stage

```bash
# After each training stage completes:
huggingface-cli upload YOUR_USERNAME/ttt-e2e-125m-checkpoints \
  /workspace/checkpoints/b1_b2/pretrain-125m-fa \
  checkpoints/pretrain-125m-fa \
  --repo-type model

huggingface-cli upload YOUR_USERNAME/ttt-e2e-125m-checkpoints \
  /workspace/checkpoints/b1_b2/ext-125m-fa-32K \
  checkpoints/ext-125m-fa-32K \
  --repo-type model

# ... repeat for e2e stages
```

### Resuming from a backed-up checkpoint

If the instance is killed and you start a new one:

```bash
# Download checkpoints from HF
huggingface-cli download YOUR_USERNAME/ttt-e2e-125m-checkpoints \
  --repo-type model \
  --local-dir /workspace/checkpoints/b1_b2

# Re-run the same training command. The reference code will detect
# existing checkpoints and resume automatically when wandb detects
# a preexisting run (or you can use load_part=all explicitly).
```

---

## 10. Crash Resume

The reference code (`ttte2e_reference/e2e/ttt/train.py` lines 182-247) handles resume automatically:

1. If W&B detects a preexisting run (`wandb_logger.preexisting`) AND a checkpoint exists in the checkpoint directory, it loads `load_part=all` (weights + optimizer + data iterator position).
2. The data iterator resumes from the exact batch it left off at (via Grain checkpoint state).
3. Training continues from `start_step` onward.

**For this to work across Vast.ai instances:**
- The checkpoint directory must be restored from HF backup (see Section 9).
- The W&B run must be the same (same `exp_name` and `wandb_project`).
- Use the exact same training command.

**Manual resume (safer)**:
If automatic resume doesn't trigger, explicitly add:
```
training.load_part=all
training.resume_exp_name=pretrain-125m-e2e
```

---

## 11. Monitoring

### W&B Dashboard
If W&B is enabled, monitor at `https://wandb.ai/YOUR_ENTITY/ttt-e2e-125m`.

Key metrics to watch:
- `loss`: Should decrease steadily. For FA pretrain, expect loss to drop from ~11 to ~4-5 over 4800 steps.
- `gradient_norm`: Should stay below the clip threshold (1.0) after warmup.
- `outer_learning_rate`: Should follow cosine schedule (warmup → peak → decay).

### GPU utilization
```bash
# In a separate tmux pane:
watch -n 5 nvidia-smi
```

All 8 GPUs should show high utilization (>80%) and high memory usage during training.

### Training logs
Hydra writes logs to `outputs/YYYY-MM-DD/HH-MM-SS/train.log` by default.

---

## 12. Expected Outputs

After completing all four stages, you should have:

```
/workspace/checkpoints/b1_b2/
├── pretrain-125m-fa/          # B2 Stage 1 checkpoint
│   └── default/
│       └── <step>/            # Orbax checkpoint dirs
├── ext-125m-fa-32K/           # B2 Stage 2 checkpoint (final B2 result)
│   └── default/
│       └── <step>/
├── pretrain-125m-e2e/         # B1 Stage 1 checkpoint
│   └── default/
│       └── <step>/
└── ext-125m-e2e-32K/          # B1 Stage 2 checkpoint (final B1 result)
    └── default/
        └── <step>/
```

Each checkpoint contains: model weights, optimizer state, data iterator state.

The extension stages also run evaluation at the final step (built into the reference code at `train.py` line 350), which produces:
- Validation loss on the eval split
- Token-level NLL breakdown (saved as `.npy` files in the experiment log directory)

---

## 13. Post-Run: Upload Everything to HF

```bash
# Upload final checkpoints
huggingface-cli upload YOUR_USERNAME/ttt-e2e-125m-checkpoints \
  /workspace/checkpoints/b1_b2 \
  checkpoints/ \
  --repo-type model

# Upload experiment logs and metrics
huggingface-cli upload YOUR_USERNAME/ttt-e2e-125m-checkpoints \
  /workspace/Reproducing-TTT-E2E/ttte2e_reference/e2e/outputs \
  logs/ \
  --repo-type model

# Upload W&B logs if needed
huggingface-cli upload YOUR_USERNAME/ttt-e2e-125m-checkpoints \
  /workspace/Reproducing-TTT-E2E/ttte2e_reference/e2e/wandb \
  wandb/ \
  --repo-type model
```

---

## 14. Cost Summary

| Item | Estimated cost |
|------|---------------:|
| Vast.ai 8xH100 (~$24/hr x 8 hrs) | **$192** |
| Vast.ai buffer for setup/issues (+4 hrs) | **$96** |
| GCS egress for data (if applicable) | **$0-50** |
| HF Hub storage (free tier) | **$0** |
| **Total estimate** | **$240-340** |

---

## 15. Runbook Checklist

### Before renting the instance
- [ ] GCS data downloaded locally or to a staging machine
- [ ] Data uploaded to HuggingFace Hub (private dataset repo)
- [ ] HuggingFace token ready (`HF_TOKEN`)
- [ ] W&B account and API key ready (or decide to disable with `log_wandb=false`)
- [ ] Repo pushed to GitHub (so you can clone on the instance)
- [ ] This runbook saved and accessible from the instance

### On the instance
- [ ] Run `setup.sh` — all 8 GPUs detected
- [ ] Run dummy-data smoke test — both FA and E2E complete 10 steps
- [ ] Start `backup_checkpoints.sh` in tmux
- [ ] Run B2 Stage 1 (pretrain-125m-fa) — verify loss decreasing
- [ ] Upload B2 Stage 1 checkpoint to HF
- [ ] Run B2 Stage 2 (ext-125m-fa-32K) — verify eval runs
- [ ] Upload B2 Stage 2 checkpoint to HF
- [ ] Run B1 Stage 1 (pretrain-125m-e2e) — verify loss decreasing, ~3.4x slower
- [ ] Upload B1 Stage 1 checkpoint to HF
- [ ] Run B1 Stage 2 (ext-125m-e2e-32K) — verify eval runs
- [ ] Upload B1 Stage 2 checkpoint to HF
- [ ] Upload all logs and experiment outputs
- [ ] Tear down instance

### After the run
- [ ] Verify all checkpoints downloadable from HF
- [ ] Compare B1 vs B2 32K validation loss
- [ ] Document results in experiment tracking

---

## 16. Troubleshooting

### "CUDA out of memory"
The 125M model should fit comfortably on 8xH100. If OOM occurs:
- Check that `global_batch_size` matches the config (64 for pretrain, 32 for ext).
- Verify no other processes are using GPU memory (`nvidia-smi`).
- For E2E meta mode, the inner loop requires extra memory for gradients-of-gradients. Try reducing `global_batch_size` to 32 and adjusting `accum_steps` to 2.

### "Zarr array not found"
The dataloader expects Zarr arrays at `/{split}` inside the dataset directory. Verify:
```python
import zarr, zarr.storage
store = zarr.storage.LocalStore("/data/ttt-e2e-data/llama3-dclm-filter-8k", read_only=True)
arr = zarr.open_array(store, path="/train")
print(arr.shape, arr.dtype)
```

### "XLA compilation taking forever"
The first training step compiles the full compute graph. This is normal and takes 5-15 minutes for 125M. Subsequent steps are fast. The compilation cache (`/tmp/jax_cache`) prevents recompilation on restart.

### "W&B authentication failed"
Either set `WANDB_API_KEY` environment variable or pass the key directly:
```
training.wandb_key=your-api-key
```
To disable W&B entirely: `training.log_wandb=false`

### Instance preempted mid-training
1. Rent a new instance.
2. Download checkpoints from HF Hub.
3. Re-run the same command — it will resume from the latest checkpoint.

### "Requester Pays" error downloading from GCS
You need a GCP project with billing enabled:
```bash
gcloud storage cp -r gs://llama3-dclm-filter-8k/ /data/ --billing-project=your-gcp-project
```

---

## 17. Key File Paths Reference

| File | Purpose |
|------|---------|
| `ttte2e_reference/e2e/ttt/train.py` | Main training entrypoint |
| `ttte2e_reference/e2e/ttt/model/transformer.py` | Model architecture (MetaModel, CausalLM, Block) |
| `ttte2e_reference/e2e/ttt/model/loop.py` | Training loop (train_on_sequence, Evaluator) |
| `ttte2e_reference/e2e/ttt/model/attention.py` | Attention implementations (SWA, Full) |
| `ttte2e_reference/e2e/ttt/infra/checkpoint.py` | Orbax checkpoint manager |
| `ttte2e_reference/e2e/ttt/dataloader/lm_dataset.py` | Zarr/Grain data loading |
| `ttte2e_reference/e2e/ttt/config.py` | Full config schema |
| `ttte2e_reference/e2e/ttt/optimizers.py` | Optimizer construction |
| `ttte2e_reference/e2e/configs/experiment/125m/` | All 125M experiment configs |
| `ttte2e_reference/e2e/configs/training/125m/` | 125M training hyperparameters |
| `ttte2e_reference/e2e/configs/model/125m.yaml` | 125M architecture definition |
| `ttte2e_reference/e2e/configs/deploy/interactive.yaml` | Single-node 8-GPU deploy config |
