"""End-to-end capstone training script.

Trains a small GPT-style decoder with one of three parallelism strategies:

* ``--parallelism ddp``   — from-scratch DDP via ``torch.distributed``.
* ``--parallelism zero``  — DeepSpeed with a JSON config (default Stage 3 + CPU offload).

Usage examples
--------------
Single-node, 4 GPUs, DDP, 350M model:

    torchrun --standalone --nproc_per_node=4 examples/train_gpt_capstone.py \\
        --config configs/training/gpt_350m.yaml --parallelism ddp

Single-node, 8 GPUs, ZeRO-3, 1.3B model:

    deepspeed --num_gpus=8 examples/train_gpt_capstone.py \\
        --config configs/training/gpt_1b3.yaml --parallelism zero \\
        --deepspeed configs/deepspeed/zero3_offload.json
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from hpcllmforge.orchestration.checkpoint_manager import CheckpointManager
from hpcllmforge.profiling.throughput_meter import ThroughputMeter
from hpcllmforge.utils.logging import get_logger
from hpcllmforge.utils.seeding import seed_everything
from hpcllmforge.utils.topology import fingerprint

logger = get_logger(__name__)


# ----------------------------- Minimal GPT --------------------------------- #


@dataclass
class GPTConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    seq_length: int
    tied_embeddings: bool = True


class GPTBlock(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size)
        self.attn = nn.MultiheadAttention(
            cfg.hidden_size, cfg.num_heads, batch_first=True
        )
        self.ln2 = nn.LayerNorm(cfg.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, 4 * cfg.hidden_size),
            nn.GELU(),
            nn.Linear(4 * cfg.hidden_size, cfg.hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = nn.Embedding(cfg.seq_length, cfg.hidden_size)
        self.blocks = nn.ModuleList([GPTBlock(cfg) for _ in range(cfg.num_layers)])
        self.ln_f = nn.LayerNorm(cfg.hidden_size)
        self.head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tied_embeddings:
            self.head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        bsz, seq = idx.shape
        pos = torch.arange(seq, device=idx.device).unsqueeze(0).expand(bsz, -1)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ----------------------------- Data loader (toy) --------------------------- #


def synthetic_loader(cfg: GPTConfig, micro_batch_size: int, steps: int):
    """Yield synthetic (input, target) token batches for benchmarking."""
    for _ in range(steps):
        idx = torch.randint(0, cfg.vocab_size, (micro_batch_size, cfg.seq_length))
        targets = torch.roll(idx, shifts=-1, dims=1)
        yield idx, targets


# ----------------------------- Helpers ------------------------------------- #


def estimate_flops_per_step(model_cfg: GPTConfig, batch_tokens: int) -> float:
    """Standard 6N approximation for Transformer training FLOPs per token."""
    # Active parameters excluding embeddings (which contribute much less FLOPs).
    n_active = 12 * model_cfg.num_layers * model_cfg.hidden_size ** 2
    return 6 * n_active * batch_tokens


def cosine_lr(step: int, *, base_lr: float, min_lr: float, warmup: int, max_steps: int) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ----------------------------- Main ---------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parallelism", choices=["ddp", "zero"], default="zero")
    parser.add_argument("--deepspeed", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    seed_everything(cfg.get("seed", 42))
    logger.info("hardware fingerprint: %s", fingerprint())

    model_cfg = GPTConfig(
        vocab_size=cfg["model"]["vocab_size"],
        hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"],
        num_heads=cfg["model"]["num_heads"],
        seq_length=cfg["model"]["seq_length"],
        tied_embeddings=cfg["model"].get("tied_embeddings", True),
    )
    model = GPT(model_cfg)
    logger.info("model parameters: %.2fM", model.num_parameters / 1e6)

    micro_batch = cfg["training"]["micro_batch_size"]
    max_steps = cfg["training"]["max_steps"]
    loader = synthetic_loader(model_cfg, micro_batch, max_steps)

    if args.parallelism == "zero":
        import deepspeed

        engine, optim, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=str(args.deepspeed),
        )
        device = engine.device
    else:
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
        model.to(device)
        if torch.distributed.is_initialized():
            engine = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[device.index]
            )
        else:
            engine = model
        optim = torch.optim.AdamW(
            engine.parameters(),
            lr=cfg["training"]["lr"],
            betas=(cfg["training"]["beta1"], cfg["training"]["beta2"]),
            weight_decay=cfg["training"]["weight_decay"],
        )

    ckpt = CheckpointManager(
        out_dir=cfg["checkpointing"]["out_dir"],
        initial_interval_sec=300.0,
    )
    meter = ThroughputMeter(
        flops_per_step=estimate_flops_per_step(model_cfg, micro_batch * model_cfg.seq_length),
        peak_flops=989e12,  # H100 BF16 peak
    )

    loss_fn = nn.CrossEntropyLoss()
    t0 = time.time()
    for step, (idx, targets) in enumerate(loader):
        idx, targets = idx.to(device), targets.to(device)
        logits = engine(idx)
        loss = loss_fn(logits.flatten(0, 1), targets.flatten())

        if args.parallelism == "zero":
            engine.backward(loss)
            engine.step()
        else:
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(engine.parameters(), cfg["training"]["grad_clip"])
            lr = cosine_lr(
                step,
                base_lr=cfg["training"]["lr"],
                min_lr=cfg["training"]["min_lr"],
                warmup=cfg["training"]["warmup_steps"],
                max_steps=max_steps,
            )
            for pg in optim.param_groups:
                pg["lr"] = lr
            optim.step()

        meter.tick()
        if step % cfg["logging"]["log_interval"] == 0:
            mfu = meter.mfu() or 0.0
            elapsed = time.time() - t0
            logger.info(
                "step=%d loss=%.4f mfu=%.1f%% elapsed=%.1fs",
                step,
                loss.item(),
                100 * mfu,
                elapsed,
            )
        if ckpt.should_checkpoint() and step > 0:
            tag = f"step_{step}"
            if args.parallelism == "zero":
                engine.save_checkpoint(ckpt.out_dir, tag=tag)
            else:
                torch.save({"model": engine.state_dict(), "step": step}, f"{ckpt.out_dir}/{tag}.pt")

    logger.info("training complete in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
