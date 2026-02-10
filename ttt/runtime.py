"""Phase-1 local runtime utilities.

This module provides:
- a deterministic simulation loop for fast orchestration checks, and
- a lightweight token-statistics trainer that runs on real token batches.

Both runtimes share the same run artifact and checkpoint conventions so we can
validate experiment lineage while full JAX internals are being re-implemented.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from ttt.config import Config
from ttt.dataloader import build_batch_iterator
from ttt.infra import Phase1Checkpointer
from ttt.model import TokenStatsModel, TokenStatsState


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    resolved_config_path: Path
    unresolved_config_path: Path
    metrics_path: Path


def prepare_run_artifacts(cfg: Config) -> RunArtifacts:
    run_dir = Path(cfg.training.exp_dir) / cfg.training.exp_folder / cfg.training.exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_config_path = run_dir / "phase1_resolved_config.yaml"
    unresolved_config_path = run_dir / "phase1_unresolved_config.yaml"
    metrics_path = run_dir / "phase1_metrics.jsonl"

    resolved_config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    unresolved_config_path.write_text(OmegaConf.to_yaml(cfg, resolve=False))

    return RunArtifacts(
        run_dir=run_dir,
        resolved_config_path=resolved_config_path,
        unresolved_config_path=unresolved_config_path,
        metrics_path=metrics_path,
    )


class Phase1Simulator:
    """Deterministic stand-in training loop for orchestration validation."""

    def __init__(self, cfg: Config, artifacts: RunArtifacts, logger: logging.Logger):
        self.cfg = cfg
        self.artifacts = artifacts
        self.logger = logger
        self.rng = random.Random(cfg.training.model_seed)
        self.checkpointer = Phase1Checkpointer(Path(cfg.checkpoint.checkpoint_dir))

    def _estimate_model_size_m(self) -> float:
        mcfg = self.cfg.model
        # Rough dense-transformer estimate used only for run metadata and trend scaling.
        per_layer = (4 * mcfg.hidden_size * mcfg.hidden_size) + (
            3 * mcfg.hidden_size * mcfg.intermediate_size
        )
        embed = mcfg.vocab_size * mcfg.hidden_size
        total = embed + (mcfg.num_hidden_layers * per_layer)
        return total / 1_000_000.0

    def _resolve_start_step(self) -> tuple[int, dict[str, Any]]:
        load_part = str(self.cfg.training.load_part)
        resume_dir = Path(self.cfg.checkpoint.resume_checkpoint_dir)
        resume_ckpt = Phase1Checkpointer(resume_dir)

        payload: dict[str, Any] = {
            "load_part": load_part,
            "resume_checkpoint_dir": str(resume_dir),
            "resume_exp_name": self.cfg.training.resume_exp_name,
        }

        if load_part == "none":
            return 0, payload

        restored = resume_ckpt.load(step=self.cfg.training.resume_step)
        if restored is None:
            payload["restore_status"] = "missing_checkpoint"
            return 0, payload

        payload["restore_status"] = "ok"
        payload["restore_step"] = restored.step

        if load_part == "all":
            return restored.step + 1, payload

        # params warm-start: start loop from step 0 but record source.
        return 0, payload

    def _learning_rate(self, step: int) -> float:
        opt = self.cfg.training.optimizer_outer

        lr = float(opt.lr)
        init_lr = float(opt.init_lr)
        end_lr = float(opt.end_lr)
        warmup_steps = max(int(opt.lr_warmup_steps), 0)
        decay_steps = max(int(opt.lr_decay_steps), 1)

        if warmup_steps > 0 and step < warmup_steps:
            progress = float(step + 1) / float(warmup_steps)
            return init_lr + progress * (lr - init_lr)

        decay_step = max(step - warmup_steps, 0)
        decay_span = max(decay_steps - warmup_steps, 1)
        t = min(max(decay_step / decay_span, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        return end_lr + cosine * (lr - end_lr)

    def _loss_baseline(self) -> float:
        mode = str(self.cfg.training.train_mode)
        seq = int(self.cfg.training.seq_length)

        if mode == "meta":
            base = 2.95
        else:
            base = 3.10

        if seq >= 32768:
            base -= 0.06
        elif seq >= 16384:
            base -= 0.03

        size_m = self._estimate_model_size_m()
        if size_m >= 2500:
            base -= 0.07
        elif size_m >= 900:
            base -= 0.04
        elif size_m >= 500:
            base -= 0.02

        return base

    def run(self) -> None:
        start_step, restore_payload = self._resolve_start_step()
        total_steps = int(self.cfg.training.total_steps)

        if start_step >= total_steps:
            self.logger.info(
                "Phase 1 simulator: start_step=%d >= total_steps=%d; nothing to run.",
                start_step,
                total_steps,
            )
            return

        save_freq = int(self.cfg.training.save_milestone_freq)
        break_step = int(self.cfg.training.break_step)
        model_size_m = self._estimate_model_size_m()
        base_loss = self._loss_baseline()

        self.logger.info(
            "Phase 1 simulator starting: start_step=%d total_steps=%d model_size_m=%.1f restore=%s",
            start_step,
            total_steps,
            model_size_m,
            restore_payload,
        )

        run_start = time.time()
        with self.artifacts.metrics_path.open("a", encoding="utf-8") as metrics_file:
            for step in range(start_step, total_steps):
                if 0 <= break_step < step:
                    self.logger.info("Break step reached at step=%d", step)
                    break

                lr = self._learning_rate(step)
                progress = (step + 1) / max(total_steps, 1)
                decay_term = 0.45 * (1.0 - progress)
                noise = (self.rng.random() - 0.5) * 0.01
                loss = max(1.2, base_loss + decay_term + noise)
                grad_norm = max(
                    0.05,
                    1.5 - 0.9 * progress + (self.rng.random() - 0.5) * 0.06,
                )

                record = {
                    "step": step,
                    "loss": round(loss, 6),
                    "gradient_norm": round(grad_norm, 6),
                    "outer_learning_rate": round(lr, 10),
                    "train_mode": str(self.cfg.training.train_mode),
                    "seq_length": int(self.cfg.training.seq_length),
                    "global_batch_size": int(self.cfg.training.global_batch_size),
                    "model_size_m": round(model_size_m, 3),
                    "restore": restore_payload,
                    "runtime_mode": "simulate",
                }
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")

                if step % 100 == 0 or step == total_steps - 1:
                    self.logger.info(
                        "step=%d/%d loss=%.4f grad_norm=%.4f lr=%.6g",
                        step,
                        total_steps,
                        loss,
                        grad_norm,
                        lr,
                    )

                should_save = (
                    (save_freq > 0 and step > 0 and step % save_freq == 0)
                    or step == total_steps - 1
                )
                if should_save:
                    ckpt_payload = {
                        "exp_name": self.cfg.training.exp_name,
                        "exp_folder": self.cfg.training.exp_folder,
                        "elapsed_seconds": round(time.time() - run_start, 3),
                        "loss": round(loss, 6),
                        "gradient_norm": round(grad_norm, 6),
                    }
                    ckpt_path = self.checkpointer.save(step=step, payload=ckpt_payload)
                    self.logger.info("Saved phase1 checkpoint: %s", ckpt_path)

        elapsed = time.time() - run_start
        self.logger.info(
            "Phase 1 simulator finished in %.2fs; metrics=%s",
            elapsed,
            self.artifacts.metrics_path,
        )


class Phase1TokenStatsTrainer:
    """Simple token-driven trainer used for phase-1 non-simulated runs."""

    def __init__(self, cfg: Config, artifacts: RunArtifacts, logger: logging.Logger):
        self.cfg = cfg
        self.artifacts = artifacts
        self.logger = logger

        self.checkpointer = Phase1Checkpointer(Path(cfg.checkpoint.checkpoint_dir))
        self.model = TokenStatsModel(vocab_size=int(cfg.model.vocab_size))

    def _flatten_targets(self, batch) -> list[int]:
        tokens: list[int] = []
        for sample in batch.samples:
            tokens.extend(sample.target_tokens)
        return tokens

    def _chunk_tokens(self, tokens: list[int], chunk_size: int) -> list[list[int]]:
        if chunk_size <= 0:
            return [tokens]

        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i : i + chunk_size]
            if chunk:
                chunks.append(chunk)
        return chunks

    def _learning_rate(self, step: int) -> float:
        opt = self.cfg.training.optimizer_outer

        lr = float(opt.lr)
        init_lr = float(opt.init_lr)
        end_lr = float(opt.end_lr)
        warmup_steps = max(int(opt.lr_warmup_steps), 0)
        decay_steps = max(int(opt.lr_decay_steps), 1)

        if warmup_steps > 0 and step < warmup_steps:
            progress = float(step + 1) / float(warmup_steps)
            return init_lr + progress * (lr - init_lr)

        decay_step = max(step - warmup_steps, 0)
        decay_span = max(decay_steps - warmup_steps, 1)
        t = min(max(decay_step / decay_span, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        return end_lr + cosine * (lr - end_lr)

    def _resolve_initial_state(self) -> tuple[int, TokenStatsState, dict[str, Any]]:
        load_part = str(self.cfg.training.load_part)
        resume_dir = Path(self.cfg.checkpoint.resume_checkpoint_dir)
        resume_ckpt = Phase1Checkpointer(resume_dir)

        restore_payload: dict[str, Any] = {
            "load_part": load_part,
            "resume_checkpoint_dir": str(resume_dir),
            "resume_exp_name": self.cfg.training.resume_exp_name,
        }

        model_state = self.model.fresh_state()
        start_step = 0

        if load_part == "none":
            return start_step, model_state, restore_payload

        restored = resume_ckpt.load(step=self.cfg.training.resume_step)
        if restored is None:
            restore_payload["restore_status"] = "missing_checkpoint"
            return start_step, model_state, restore_payload

        restore_payload["restore_status"] = "ok"
        restore_payload["restore_step"] = restored.step

        raw_state = restored.payload.get("model_state")
        if isinstance(raw_state, dict):
            model_state = TokenStatsState.from_jsonable(raw_state)
        else:
            restore_payload["state_status"] = "missing_model_state"

        if load_part == "all":
            start_step = restored.step + 1

        return start_step, model_state, restore_payload

    def _build_data_iterator(self):
        return build_batch_iterator(
            dataset_path=self.cfg.training.dataset_path,
            split=self.cfg.training.data_split,
            seq_len=int(self.cfg.training.seq_length),
            global_batch_size=int(self.cfg.training.global_batch_size),
            repeat=not bool(self.cfg.training.eval_mode),
            shuffle=not bool(self.cfg.training.eval_mode),
            seed=int(self.cfg.training.data_seed),
            dummy_dataset=bool(self.cfg.training.dummy_dataset),
            vocab_size=int(self.cfg.model.vocab_size),
        )

    def run(self) -> None:
        total_steps = int(self.cfg.training.total_steps)
        save_freq = int(self.cfg.training.save_milestone_freq)
        break_step = int(self.cfg.training.break_step)

        start_step, model_state, restore_payload = self._resolve_initial_state()

        if start_step >= total_steps:
            self.logger.info(
                "Token-stats runtime: start_step=%d >= total_steps=%d; nothing to run.",
                start_step,
                total_steps,
            )
            return

        data_iter = self._build_data_iterator()
        run_start = time.time()
        base_outer_lr = max(float(self.cfg.training.optimizer_outer.lr), 1e-8)

        self.logger.info(
            "Token-stats runtime starting: start_step=%d total_steps=%d restore=%s",
            start_step,
            total_steps,
            restore_payload,
        )

        with self.artifacts.metrics_path.open("a", encoding="utf-8") as metrics_file:
            for step in range(start_step, total_steps):
                if 0 <= break_step < step:
                    self.logger.info("Break step reached at step=%d", step)
                    break

                try:
                    batch = next(data_iter)
                except StopIteration:
                    self.logger.info("Data iterator exhausted at step=%d", step)
                    break

                targets = self._flatten_targets(batch)
                if not targets:
                    self.logger.info("Empty target batch at step=%d; stopping", step)
                    break

                outer_lr = self._learning_rate(step)

                if str(self.cfg.training.train_mode) == "meta":
                    chunk_size = max(int(self.cfg.model.mini_batch_size), 1)
                    inner_state = self.model.clone_state(model_state)
                    chunk_losses: list[float] = []

                    for chunk in self._chunk_tokens(targets, chunk_size):
                        chunk_losses.append(self.model.nll(inner_state, chunk))
                        self.model.update(inner_state, chunk, weight=1.0)

                    loss = sum(chunk_losses) / max(len(chunk_losses), 1)

                    update_weight = max(outer_lr / base_outer_lr, 0.05)
                    self.model.update(model_state, targets, weight=update_weight)
                    inner_steps = len(chunk_losses)
                else:
                    loss = self.model.nll(model_state, targets)
                    update_weight = max(outer_lr / base_outer_lr, 0.05)
                    self.model.update(model_state, targets, weight=update_weight)
                    inner_steps = 0

                grad_norm = max(0.01, min(10.0, loss * 0.2 + 0.05 * (1 + inner_steps)))

                record = {
                    "step": step,
                    "loss": round(loss, 6),
                    "gradient_norm": round(grad_norm, 6),
                    "outer_learning_rate": round(outer_lr, 10),
                    "train_mode": str(self.cfg.training.train_mode),
                    "runtime_mode": "token_stats",
                    "seq_length": int(self.cfg.training.seq_length),
                    "global_batch_size": int(self.cfg.training.global_batch_size),
                    "tokens_in_batch": len(targets),
                    "inner_steps": inner_steps,
                    "restore": restore_payload,
                }
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")

                if step % 50 == 0 or step == total_steps - 1:
                    self.logger.info(
                        "step=%d/%d loss=%.4f grad_norm=%.4f lr=%.6g tokens=%d",
                        step,
                        total_steps,
                        loss,
                        grad_norm,
                        outer_lr,
                        len(targets),
                    )

                should_save = (
                    (save_freq > 0 and step > 0 and step % save_freq == 0)
                    or step == total_steps - 1
                )
                if should_save:
                    payload = {
                        "exp_name": self.cfg.training.exp_name,
                        "exp_folder": self.cfg.training.exp_folder,
                        "elapsed_seconds": round(time.time() - run_start, 3),
                        "loss": round(loss, 6),
                        "gradient_norm": round(grad_norm, 6),
                        "model_state": model_state.to_jsonable(),
                    }
                    ckpt_path = self.checkpointer.save(step=step, payload=payload)
                    self.logger.info("Saved token-stats checkpoint: %s", ckpt_path)

        elapsed = time.time() - run_start
        self.logger.info(
            "Token-stats runtime finished in %.2fs; metrics=%s",
            elapsed,
            self.artifacts.metrics_path,
        )
