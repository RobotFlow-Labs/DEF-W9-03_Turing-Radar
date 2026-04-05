"""Training script for pulse embedding metric learning model."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import load_config
from .embedding import PulseEmbeddingNet
from .io import iter_pulse_train_files, load_pulse_train_h5
from .preprocess import PDWPreprocessor

PROJECT = "turing-radar"
ARTIFACTS = "/mnt/artifacts-datai"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PulsePairDataset(Dataset):
    """Generates pairs of pulses: positive (same emitter) and negative (different emitter)."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        unique = torch.unique(self.labels)
        self.label_to_indices: dict[int, torch.Tensor] = {}
        for lbl in unique.tolist():
            self.label_to_indices[lbl] = torch.where(self.labels == lbl)[0]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor = self.features[idx]
        anchor_label = self.labels[idx].item()

        # Positive: same emitter
        pos_indices = self.label_to_indices[anchor_label]
        pos_idx = pos_indices[torch.randint(len(pos_indices), (1,))].item()
        positive = self.features[pos_idx]

        # Negative: different emitter
        neg_label = anchor_label
        labels_list = list(self.label_to_indices.keys())
        while neg_label == anchor_label:
            neg_label = labels_list[torch.randint(len(labels_list), (1,)).item()]
        neg_indices = self.label_to_indices[neg_label]
        neg_idx = neg_indices[torch.randint(len(neg_indices), (1,))].item()
        negative = self.features[neg_idx]

        return anchor, positive, negative


class ContrastiveTripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        d_pos = torch.sum((anchor - positive) ** 2, dim=1)
        d_neg = torch.sum((anchor - negative) ** 2, dim=1)
        loss = torch.clamp(d_pos - d_neg + self.margin, min=0.0)
        return loss.mean()


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self) -> None:
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            scale = self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base_lr * scale)

    def state_dict(self) -> dict:
        return {"current_step": self.current_step}

    def load_state_dict(self, state: dict) -> None:
        self.current_step = state["current_step"]


class CheckpointManager:
    def __init__(self, save_dir: Path, keep_top_k: int = 2, metric: str = "val_loss"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.metric = metric
        self.history: list[tuple[float, Path]] = []

    def save(self, state: dict, metric_value: float, step: int) -> Path:
        path = self.save_dir / f"checkpoint_step{step:06d}.pth"
        torch.save(state, path)
        self.history.append((metric_value, path))
        self.history.sort(key=lambda x: x[0])
        while len(self.history) > self.keep_top_k:
            _, old_path = self.history.pop()
            old_path.unlink(missing_ok=True)
        best_val, best_path = self.history[0]
        best_dest = self.save_dir / "best.pth"
        if best_path != best_dest:
            import shutil

            shutil.copy2(best_path, best_dest)
        return path


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def step(self, metric: float) -> bool:
        if metric < self.best - self.min_delta:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def load_and_preprocess_files(
    data_root: str,
    subset: str,
    preprocessor: PDWPreprocessor,
    max_files: int,
    file_pattern: str = "*.h5",
) -> tuple[np.ndarray, np.ndarray]:
    """Load pulse train files, preprocess, and concatenate."""
    files = list(iter_pulse_train_files(data_root, subset=subset, file_pattern=file_pattern))
    files = files[:max_files]
    if not files:
        raise RuntimeError(f"No files found in {data_root}/{subset}")

    all_features = []
    all_labels = []
    label_offset = 0

    for i, fpath in enumerate(files):
        record = load_pulse_train_h5(fpath)
        if record.labels is None:
            continue
        features = preprocessor.fit_transform(record.data)
        labels = record.labels.copy()
        # Make labels globally unique across files
        labels = labels + label_offset
        label_offset = labels.max() + 1
        all_features.append(features)
        all_labels.append(labels)
        if (i + 1) % 100 == 0:
            print(f"  Loaded {i + 1}/{len(files)} files")

    print(f"  Loaded {len(all_features)} labeled files from {subset}")
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return features, labels


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train pulse embedding model")
    p.add_argument("--config", default="configs/train.toml", help="Config file path")
    p.add_argument("--resume", default=None, help="Resume from checkpoint")
    p.add_argument("--max-steps", type=int, default=None, help="Max training steps (debug)")
    p.add_argument("--device", default="auto", help="Device: auto, cuda, cpu")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)

    # Read training config from TOML
    tcfg = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore
        with cfg_path.open("rb") as f:
            tcfg = tomllib.load(f)

    train_cfg = tcfg.get("training", {})
    ckpt_cfg = tcfg.get("checkpoint", {})

    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[DEVICE] {device}")

    # Hyperparameters
    batch_size = train_cfg.get("batch_size", 4096)
    lr = train_cfg.get("learning_rate", 0.001)
    epochs = train_cfg.get("epochs", 50)
    hidden_dim = train_cfg.get("hidden_dim", 128)
    embed_dim = train_cfg.get("embed_dim", 32)
    margin = train_cfg.get("margin", 1.0)
    num_workers = train_cfg.get("num_workers", 4)
    max_files = train_cfg.get("max_files", 2500)
    warmup_ratio = train_cfg.get("warmup_ratio", 0.05)
    min_lr = train_cfg.get("min_lr", 1e-6)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    gradient_clip = train_cfg.get("gradient_clip", 1.0)
    patience = train_cfg.get("patience", 10)
    val_every = train_cfg.get("val_every_n_epochs", 1)

    output_dir = Path(ckpt_cfg.get("output_dir", f"{ARTIFACTS}/checkpoints/{PROJECT}"))
    keep_top_k = ckpt_cfg.get("keep_top_k", 2)
    save_every = ckpt_cfg.get("save_every_n_steps", 500)

    log_dir = Path(f"{ARTIFACTS}/logs/{PROJECT}")
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CONFIG] {args.config}")
    print(f"[BATCH] batch_size={batch_size}")
    print(f"[MODEL] PulseEmbeddingNet(5 -> {hidden_dim} -> {embed_dim})")
    print(f"[TRAIN] {epochs} epochs, lr={lr}, optimizer=AdamW")
    print(f"[CKPT] save every {save_every} steps, keep best {keep_top_k}")

    # Load data
    print("[DATA] Loading training data...")
    preprocessor = PDWPreprocessor.from_config(cfg.preprocess)

    # Determine split directories
    data_root = cfg.data.data_root
    subset = cfg.data.subset
    # Try standard split directories: train/val/test under subset
    train_dir = f"{subset}/train"
    val_dir = f"{subset}/val"

    # Check if split dirs exist, fallback to single subset
    train_root = Path(data_root) / train_dir
    val_root = Path(data_root) / val_dir
    if not train_root.exists():
        # Fallback: use subset directly, split by file index
        print(f"  No train/val split dirs found, splitting {subset} by index")
        train_dir = subset
        val_dir = subset
        use_index_split = True
    else:
        use_index_split = False

    train_features, train_labels = load_and_preprocess_files(
        data_root, train_dir, preprocessor, max_files, cfg.data.file_pattern
    )

    if use_index_split:
        # Split 90/10
        n = len(train_features)
        perm = np.random.permutation(n)
        split = int(0.9 * n)
        val_features = train_features[perm[split:]]
        val_labels = train_labels[perm[split:]]
        train_features = train_features[perm[:split]]
        train_labels = train_labels[perm[:split]]
    else:
        val_max = max(max_files // 10, 5)
        val_features, val_labels = load_and_preprocess_files(
            data_root, val_dir, preprocessor, val_max, cfg.data.file_pattern
        )

    print(f"[DATA] train={len(train_features)} pulses, val={len(val_features)} pulses")

    # Datasets and loaders
    train_ds = PulsePairDataset(train_features, train_labels)
    val_ds = PulsePairDataset(val_features, val_labels)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    # Model
    model = PulseEmbeddingNet(in_dim=5, hidden_dim=hidden_dim, out_dim=embed_dim).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] {param_count:,} parameters")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps, min_lr)

    # Loss
    criterion = ContrastiveTripletLoss(margin=margin)

    # Checkpoint manager + early stopping
    ckpt_mgr = CheckpointManager(output_dir, keep_top_k=keep_top_k)
    early_stop = EarlyStopping(patience=patience)

    # Resume
    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("step", 0)
        print(f"[RESUME] from epoch {start_epoch}, step {global_step}")

    # Mixed precision
    use_amp = train_cfg.get("precision", "bf16") in ("bf16", "fp16")
    dtype = torch.bfloat16 if train_cfg.get("precision") == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(enabled=(use_amp and dtype == torch.float16))

    # Training history
    history: list[dict] = []

    print(f"[TRAIN] Starting training for {epochs} epochs ({total_steps} steps)")
    t0 = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for anchor, positive, negative in train_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
                z_a = model(anchor)
                z_p = model(positive)
                z_n = model(negative)
                loss = criterion(z_a, z_p, z_n)

            if torch.isnan(loss):
                print("[FATAL] Loss is NaN -- stopping training")
                print("[FIX] Reduce lr by 10x, check data for corrupt samples")
                return 1

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % save_every == 0:
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                    "config": train_cfg,
                }
                ckpt_mgr.save(state, epoch_loss / n_batches, global_step)

            if args.max_steps and global_step >= args.max_steps:
                print(f"[DEBUG] Reached max_steps={args.max_steps}, stopping")
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                    "config": train_cfg,
                }
                ckpt_mgr.save(state, epoch_loss / max(n_batches, 1), global_step)
                return 0

        avg_train_loss = epoch_loss / max(n_batches, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        # Validation
        val_loss = 0.0
        if (epoch + 1) % val_every == 0:
            model.eval()
            val_n = 0
            with torch.no_grad():
                for anchor, positive, negative in val_loader:
                    anchor = anchor.to(device)
                    positive = positive.to(device)
                    negative = negative.to(device)
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
                        z_a = model(anchor)
                        z_p = model(positive)
                        z_n = model(negative)
                        loss = criterion(z_a, z_p, z_n)
                    val_loss += loss.item()
                    val_n += 1
            val_loss = val_loss / max(val_n, 1)

            # Checkpoint
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "step": global_step,
                "config": train_cfg,
            }
            ckpt_mgr.save(state, val_loss, global_step)

            # Early stopping
            if early_stop.step(val_loss):
                elapsed = time.time() - t0
                print(f"[EARLY STOP] No improvement for {patience} epochs at epoch {epoch + 1}")
                print(f"[DONE] Best val_loss={early_stop.best:.6f}, time={elapsed:.0f}s")
                break

        elapsed = time.time() - t0
        record = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
            "step": global_step,
            "time_s": elapsed,
        }
        history.append(record)
        print(
            f"[Epoch {epoch + 1}/{epochs}] "
            f"train_loss={avg_train_loss:.6f} val_loss={val_loss:.6f} "
            f"lr={current_lr:.2e} step={global_step} time={elapsed:.0f}s"
        )

    # Save history
    history_path = log_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[DONE] History saved to {history_path}")
    print(f"[DONE] Best checkpoint at {output_dir / 'best.pth'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
