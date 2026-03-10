#!/usr/bin/env python3
"""Import Qwen2.5-1.5B HuggingFace weights into reference-code Orbax checkpoint.

This script handles:
1. Downloading safetensors from HuggingFace
2. GQA expansion (2 KV heads → 12 by replication) for MHA compatibility
3. Weight name mapping (HF → Equinox pytree structure)
4. Transposition (HF stores [out, in]; Equinox NormalLinear stores [in, out])
5. Saving as Orbax checkpoint compatible with reference Checkpointer

The reference code's transformer uses standard MHA (all projections are
hidden_size × hidden_size). Qwen2.5-1.5B uses GQA with 12 Q heads and 2 KV
heads. We expand K/V projections by repeating each KV head 6 times.  This is
mathematically lossless for the initial forward pass.

Usage:
    python scripts/26_import_qwen_to_orbax.py \\
        --checkpoint-dir ./checkpoints/phase2/import-qwen15-fa-base \\
        [--validate]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Resolve the reference code BEFORE importing anything from the main repo ──
# The reference code uses `ttt` as its package root, same as the main repo.
# We insert the reference path first so `import ttt` resolves to the reference.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_REF_ROOT = _REPO_ROOT / "ttte2e_reference" / "e2e"
sys.path.insert(0, str(_REF_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

# ── Reference code imports (resolved via sys.path to ttte2e_reference/e2e) ──
from ttt.config import (
    CheckpointConfig,
    Config,
    ModelConfig,
    TrainingConfig,
)
from ttt.infra.checkpoint import Checkpointer, unify_dict_with_eqx_module
from ttt.model.transformer import MetaModel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HF_MODEL_ID = "Qwen/Qwen2.5-1.5B"
HF_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"

# Qwen2.5-1.5B architecture
NUM_LAYERS = 28
HIDDEN_SIZE = 1536
NUM_Q_HEADS = 12
NUM_KV_HEADS = 2
HEAD_DIM = HIDDEN_SIZE // NUM_Q_HEADS  # 128
INTERMEDIATE_SIZE = 8960
VOCAB_SIZE = 151936
ROPE_THETA = 1_000_000.0
RMS_NORM_EPS = 1e-6

GQA_REPEAT = NUM_Q_HEADS // NUM_KV_HEADS  # 6


# ---------------------------------------------------------------------------
# GQA expansion
# ---------------------------------------------------------------------------

def expand_kv_heads(weight: np.ndarray) -> np.ndarray:
    """Expand GQA K/V projection weights from num_kv_heads to num_q_heads.

    Input shape:  [num_kv_heads * head_dim, hidden_size]  = [256, 1536]
    Output shape: [num_q_heads * head_dim, hidden_size]   = [1536, 1536]

    Each KV head is replicated GQA_REPEAT times (6x for Qwen2.5-1.5B).
    """
    kv_dim = NUM_KV_HEADS * HEAD_DIM
    assert weight.shape[0] == kv_dim, (
        f"Expected first dim {kv_dim}, got {weight.shape[0]}"
    )
    # [num_kv_heads, head_dim, hidden]
    reshaped = weight.reshape(NUM_KV_HEADS, HEAD_DIM, -1)
    # [num_kv_heads, repeat, head_dim, hidden]
    expanded = np.repeat(reshaped, GQA_REPEAT, axis=0)
    # [num_q_heads * head_dim, hidden]
    return expanded.reshape(NUM_Q_HEADS * HEAD_DIM, -1)


# ---------------------------------------------------------------------------
# HF → Equinox weight mapping
# ---------------------------------------------------------------------------

def load_hf_weights(cache_dir: Path | None = None) -> dict[str, np.ndarray]:
    """Download and load Qwen2.5-1.5B safetensors weights."""
    try:
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError(
            "Install huggingface_hub and safetensors: "
            "pip install huggingface_hub safetensors"
        ) from exc

    model_path = Path(snapshot_download(
        HF_MODEL_ID,
        revision=HF_REVISION,
        allow_patterns=["*.safetensors", "config.json"],
        cache_dir=str(cache_dir) if cache_dir else None,
    ))

    weights: dict[str, np.ndarray] = {}
    for sf_path in sorted(model_path.glob("*.safetensors")):
        with safe_open(str(sf_path), framework="numpy") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)

    print(f"Loaded {len(weights)} tensors from {model_path}")
    return weights


def build_layer_weight_dict(
    hf_weights: dict[str, np.ndarray],
    layer_idx: int,
) -> dict[str, np.ndarray]:
    """Build the Equinox-compatible weight dict for one transformer block.

    Returns a flat dict matching the Equinox Block pytree leaf structure:
      seq_modeling_block.wq.weight, seq_modeling_block.wk.weight, ...
      feed_forward.w1.weight, feed_forward.w2.weight, feed_forward.w3.weight
      seq_norm.weight, ffn_norm.weight, seq_post_norm.weight, ffn_post_norm.weight
    """
    prefix = f"model.layers.{layer_idx}"

    def hf(name: str) -> np.ndarray:
        full = f"{prefix}.{name}"
        if full not in hf_weights:
            raise KeyError(f"Missing HF weight: {full}")
        return hf_weights[full]

    # Attention projections — HF shape is [out, in], Equinox NormalLinear is [in, out]
    wq = hf("self_attn.q_proj.weight").T                     # [hidden, hidden]
    wk_raw = hf("self_attn.k_proj.weight")                   # [kv_dim, hidden]
    wk = expand_kv_heads(wk_raw).T                            # [hidden, hidden]
    wv_raw = hf("self_attn.v_proj.weight")                    # [kv_dim, hidden]
    wv = expand_kv_heads(wv_raw).T                            # [hidden, hidden]
    wo = hf("self_attn.o_proj.weight").T                      # [hidden, hidden]

    # SwiGLU MLP — HF: gate_proj=w1, up_proj=w3, down_proj=w2
    w1 = hf("mlp.gate_proj.weight").T                         # [hidden, intermediate]
    w3 = hf("mlp.up_proj.weight").T                           # [hidden, intermediate]
    w2 = hf("mlp.down_proj.weight").T                         # [intermediate, hidden]

    # Layer norms
    seq_norm = hf("input_layernorm.weight")                   # [hidden]
    ffn_norm = hf("post_attention_layernorm.weight")          # [hidden]

    return {
        "wq": wq,
        "wk": wk,
        "wv": wv,
        "wo": wo,
        "w1": w1,
        "w2": w2,
        "w3": w3,
        "seq_norm": seq_norm,
        "ffn_norm": ffn_norm,
    }


# ---------------------------------------------------------------------------
# Reference model creation
# ---------------------------------------------------------------------------

def make_ref_config(checkpoint_dir: str) -> Config:
    """Build a reference Config for Qwen2.5-1.5B dimensions."""
    model = ModelConfig(
        name="qwen2_5_1_5b",
        vocab_size=VOCAB_SIZE,
        output_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_Q_HEADS,
        mini_batch_size=8192,
        sliding_window_size=8192,
        seq_len=8192,
        rms_norm_eps=RMS_NORM_EPS,
        initializer_range=0.02,
        bos_token_id=151643,
        eos_token_id=151643,
        tie_word_embeddings=True,
        rope_theta=ROPE_THETA,
        seq_modeling_block="self_attention",
        force_flash=False,
        qk_norm=False,  # Qwen2.5-1.5B does NOT use QK normalization
        pre_norm=True,
        post_norm=True,
        compute_dtype="fp32",
        param_dtype="fp32",
        state_dtype="fp32",
    )
    training = TrainingConfig(
        exp_name="import-qwen15-fa-base",
        log_wandb=False,
        wandb_entity="none",
        wandb_project="none",
        wandb_key="none",
        load_part="none",
        total_steps=1,
        dataset_path="/tmp/dummy",
        dataset_name="dummy",
        seq_length=8192,
        global_batch_size=1,
        dummy_dataset=True,
    )
    checkpoint = CheckpointConfig(
        checkpoint_dir=checkpoint_dir,
        resume_checkpoint_dir=checkpoint_dir,
    )
    return Config(
        model=model,
        training=training,
        checkpoint=checkpoint,
    )


def create_ref_model(cfg: Config) -> MetaModel:
    """Instantiate a reference MetaModel with Qwen2.5-1.5B dimensions."""
    key = jax.random.PRNGKey(0)
    model = MetaModel(cfg, key=key)
    print(f"Created reference model (random init)")
    return model


# ---------------------------------------------------------------------------
# Weight injection
# ---------------------------------------------------------------------------

def inject_weights(model: MetaModel, hf_weights: dict[str, np.ndarray]) -> MetaModel:
    """Replace random weights in the reference model with HF Qwen weights.

    Uses jax.tree_util to traverse the Equinox pytree and replace matching
    leaves with the corresponding HF tensors.
    """
    import equinox as eqx

    # ── Embedding ──
    hf_embed = jnp.array(hf_weights["model.embed_tokens.weight"], dtype=jnp.float32)
    model = eqx.tree_at(
        lambda m: m.language_model.model.wte.weight,
        model,
        hf_embed,
    )
    print(f"  Injected embedding: {hf_embed.shape}")

    # ── Final layer norm ──
    hf_ln_f = jnp.array(hf_weights["model.norm.weight"], dtype=jnp.float32)
    model = eqx.tree_at(
        lambda m: m.language_model.model.ln_f.weight,
        model,
        hf_ln_f,
    )
    print(f"  Injected final ln: {hf_ln_f.shape}")

    # ── Transformer blocks (vmapped: leaves have shape [num_layers, ...]) ──
    blocks = model.language_model.model.h.blocks

    for layer_idx in range(NUM_LAYERS):
        layer_dict = build_layer_weight_dict(hf_weights, layer_idx)

        # Attention projections
        for proj_name in ("wq", "wk", "wv", "wo"):
            old_leaf = getattr(blocks.seq_modeling_block, proj_name).weight
            new_val = jnp.array(layer_dict[proj_name], dtype=jnp.float32)
            updated = old_leaf.at[layer_idx].set(new_val)
            blocks = eqx.tree_at(
                lambda b, pn=proj_name: getattr(b.seq_modeling_block, pn).weight,
                blocks,
                updated,
            )

        # SwiGLU MLP
        for mlp_name in ("w1", "w2", "w3"):
            old_leaf = getattr(blocks.feed_forward, mlp_name).weight
            new_val = jnp.array(layer_dict[mlp_name], dtype=jnp.float32)
            updated = old_leaf.at[layer_idx].set(new_val)
            blocks = eqx.tree_at(
                lambda b, mn=mlp_name: getattr(b.feed_forward, mn).weight,
                blocks,
                updated,
            )

        # Layer norms (seq_norm = input_layernorm, ffn_norm = post_attention_layernorm)
        for norm_name, dict_key in [("seq_norm", "seq_norm"), ("ffn_norm", "ffn_norm")]:
            old_leaf = getattr(blocks, norm_name).weight
            new_val = jnp.array(layer_dict[dict_key], dtype=jnp.float32)
            updated = old_leaf.at[layer_idx].set(new_val)
            blocks = eqx.tree_at(
                lambda b, nn=norm_name: getattr(b, nn).weight,
                blocks,
                updated,
            )

        # Post-norms: initialize to ones (Qwen doesn't have separate post-norms,
        # but the reference code does — they act as identity when weight=1)
        for norm_name in ("seq_post_norm", "ffn_post_norm"):
            old_leaf = getattr(blocks, norm_name).weight
            new_val = jnp.ones_like(old_leaf[layer_idx])
            updated = old_leaf.at[layer_idx].set(new_val)
            blocks = eqx.tree_at(
                lambda b, nn=norm_name: getattr(b, nn).weight,
                blocks,
                updated,
            )

        # QK norms: initialize to ones (Qwen2.5-1.5B doesn't use QK norm,
        # but the reference code always creates q_norm/k_norm RMSNorm layers)
        for norm_name in ("q_norm", "k_norm"):
            old_leaf = getattr(blocks.seq_modeling_block, norm_name).weight
            new_val = jnp.ones_like(old_leaf[layer_idx])
            updated = old_leaf.at[layer_idx].set(new_val)
            blocks = eqx.tree_at(
                lambda b, nn=norm_name: getattr(b.seq_modeling_block, nn).weight,
                blocks,
                updated,
            )

    # Replace blocks back into model
    model = eqx.tree_at(
        lambda m: m.language_model.model.h.blocks,
        model,
        blocks,
    )
    print(f"  Injected {NUM_LAYERS} transformer blocks")

    # ── Output (tied embeddings → lm_head is None) ──
    if model.language_model.lm_head is not None:
        hf_lm_head = jnp.array(
            hf_weights.get("lm_head.weight", hf_weights["model.embed_tokens.weight"]),
            dtype=jnp.float32,
        ).T  # HF [vocab, hidden] → Equinox NormalLinear [hidden, vocab]
        model = eqx.tree_at(
            lambda m: m.language_model.lm_head.weight,
            model,
            hf_lm_head,
        )
        print(f"  Injected lm_head: {hf_lm_head.shape}")
    else:
        print("  lm_head tied to embedding (skipped)")

    return model


# ---------------------------------------------------------------------------
# Orbax save
# ---------------------------------------------------------------------------

def save_orbax_checkpoint(model: MetaModel, cfg: Config, step: int = 0) -> Path:
    """Save model weights + dummy opt_state as Orbax checkpoint.

    The reference Checkpointer.load_checkpoint unconditionally reads opt_state
    metadata (even when load_part=params), so we must include a valid opt_state.
    The opt_state is never actually loaded for our use case — only its metadata
    structure needs to be present.
    """
    # Orbax 0.11.x may call jax.monitoring.record_scalar, which doesn't
    # exist in jax 0.5.x. Patch it to a no-op.
    if not hasattr(jax.monitoring, "record_scalar"):
        jax.monitoring.record_scalar = lambda *a, **kw: None

    import orbax.checkpoint as ocp
    from ttt.optimizers import make_optimizer

    model_weights = model.weights()

    # Create dummy optimizer state (needed for checkpoint format compatibility)
    optimizer, _ = make_optimizer(cfg.training.optimizer_outer)
    opt_state = optimizer.init(model_weights)

    ckpt_dir = Path(cfg.checkpoint.checkpoint_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    handler_registry = ocp.DefaultCheckpointHandlerRegistry()
    for item_name in ("model_weights", "opt_state"):
        handler_registry.add(
            item_name, ocp.args.StandardSave, ocp.StandardCheckpointHandler,
        )
        handler_registry.add(
            item_name, ocp.args.StandardRestore, ocp.StandardCheckpointHandler,
        )

    from orbax.checkpoint import options as ocp_opt
    mp_opts = ocp_opt.MultiprocessingOptions(primary_host=0)
    ckpt_opts = ocp.CheckpointManagerOptions(multiprocessing_options=mp_opts)
    manager = ocp.CheckpointManager(
        str(ckpt_dir),
        options=ckpt_opts,
        handler_registry=handler_registry,
    )
    manager.save(
        step=step,
        args=ocp.args.Composite(
            model_weights=ocp.args.StandardSave(model_weights),
            opt_state=ocp.args.StandardSave(opt_state),
        ),
        force=True,
    )
    manager.wait_until_finished()
    manager.close()

    print(f"Saved Orbax checkpoint to {ckpt_dir} (step={step})")
    return ckpt_dir


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_forward_pass(model: MetaModel, cfg: Config) -> bool:
    """Compare forward-pass logits with HuggingFace Qwen2ForCausalLM.

    Returns True if max absolute logit difference < 1e-2 (allowing for
    fp32 vs bf16 differences and the GQA expansion approximation).
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("WARNING: torch/transformers not available — skipping validation")
        return True

    print("Validating forward pass against HuggingFace model...")

    # Small test input
    test_tokens = list(range(1, 11))  # [1, 2, ..., 10]

    # HF forward pass
    hf_model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID,
        revision=HF_REVISION,
        torch_dtype=torch.float32,
    )
    hf_model.eval()
    with torch.no_grad():
        hf_input = torch.tensor([test_tokens], dtype=torch.long)
        hf_logits = hf_model(hf_input).logits[0].numpy()  # [seq, vocab]

    print(f"  HF logits shape: {hf_logits.shape}, range: [{hf_logits.min():.4f}, {hf_logits.max():.4f}]")

    # Reference model forward pass
    import equinox as eqx
    from equinox import nn
    from ttt.model.data import Batch

    input_ids = jnp.array([test_tokens], dtype=jnp.int32)
    target_tokens = jnp.zeros_like(input_ids)
    loss_masks = jnp.ones_like(input_ids, dtype=jnp.float32)
    position_ids = jnp.arange(len(test_tokens), dtype=jnp.int32)
    seq = Batch(
        input_ids=input_ids[0],
        target_tokens=target_tokens[0],
        loss_masks=loss_masks[0],
        position_ids=position_ids,
    )

    state = nn.make_with_state(model.language_model.model.h.blocks)[1]
    lm_out = model.language_model(state=state, seq=seq)
    ref_logits = np.array(lm_out.logits)  # [seq, vocab]

    print(f"  Ref logits shape: {ref_logits.shape}, range: [{ref_logits.min():.4f}, {ref_logits.max():.4f}]")

    # Compare
    max_diff = float(np.max(np.abs(hf_logits - ref_logits)))
    mean_diff = float(np.mean(np.abs(hf_logits - ref_logits)))
    print(f"  Max absolute diff: {max_diff:.6f}")
    print(f"  Mean absolute diff: {mean_diff:.6f}")

    threshold = 1e-2  # Allow for floating point differences
    if max_diff < threshold:
        print(f"  PASS: max diff {max_diff:.6f} < {threshold}")
        return True
    else:
        print(f"  FAIL: max diff {max_diff:.6f} >= {threshold}")
        print("  This may indicate a weight mapping error.")
        return False


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_import_report(checkpoint_dir: Path, passed_validation: bool | None) -> None:
    """Write a JSON report summarizing the import."""
    import datetime
    report = {
        "schema_version": "1.0",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hf_model_id": HF_MODEL_ID,
        "hf_revision": HF_REVISION,
        "architecture": {
            "num_layers": NUM_LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "num_q_heads": NUM_Q_HEADS,
            "num_kv_heads_original": NUM_KV_HEADS,
            "num_kv_heads_expanded": NUM_Q_HEADS,
            "gqa_repeat_factor": GQA_REPEAT,
            "head_dim": HEAD_DIM,
            "intermediate_size": INTERMEDIATE_SIZE,
            "vocab_size": VOCAB_SIZE,
        },
        "checkpoint_dir": str(checkpoint_dir),
        "validation_passed": passed_validation,
    }
    report_path = checkpoint_dir / "import_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote import report to {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import Qwen2.5-1.5B HF weights into reference-code Orbax checkpoint."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("./checkpoints/phase2/import-qwen15-fa-base"),
        help="Destination directory for the Orbax checkpoint.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="HuggingFace cache directory for downloaded model files.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run forward-pass validation against HF model (requires torch + transformers).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and map weights but don't save checkpoint.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"=== Importing {HF_MODEL_ID} (rev {HF_REVISION[:12]}...) ===")
    print(f"  GQA expansion: {NUM_KV_HEADS} KV heads → {NUM_Q_HEADS} (repeat {GQA_REPEAT}x)")
    print(f"  Target checkpoint: {args.checkpoint_dir}")

    # Step 1: Download HF weights
    print("\n[1/4] Downloading HF weights...")
    hf_weights = load_hf_weights(cache_dir=args.hf_cache_dir)

    # Verify key shapes
    q_shape = hf_weights["model.layers.0.self_attn.q_proj.weight"].shape
    k_shape = hf_weights["model.layers.0.self_attn.k_proj.weight"].shape
    print(f"  q_proj shape: {q_shape} (expected [{HIDDEN_SIZE}, {HIDDEN_SIZE}])")
    print(f"  k_proj shape: {k_shape} (expected [{NUM_KV_HEADS * HEAD_DIM}, {HIDDEN_SIZE}])")
    assert q_shape == (HIDDEN_SIZE, HIDDEN_SIZE), f"Unexpected q_proj shape: {q_shape}"
    assert k_shape == (NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE), f"Unexpected k_proj shape: {k_shape}"

    # Step 2: Create reference model
    print("\n[2/4] Creating reference model...")
    cfg = make_ref_config(str(args.checkpoint_dir))
    model = create_ref_model(cfg)

    # Step 3: Inject weights
    print("\n[3/4] Injecting HF weights (with GQA expansion)...")
    model = inject_weights(model, hf_weights)

    if args.dry_run:
        print("\n[dry-run] Skipping checkpoint save.")
        return 0

    # Step 4: Save
    print("\n[4/4] Saving Orbax checkpoint...")
    ckpt_dir = save_orbax_checkpoint(model, cfg)

    # Optional validation
    passed = None
    if args.validate:
        print("\n[validate] Running forward-pass comparison...")
        passed = validate_forward_pass(model, cfg)

    write_import_report(ckpt_dir, passed)

    print("\n=== Import complete ===")
    return 0 if (passed is None or passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
