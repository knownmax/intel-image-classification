"""
train.py — Two-phase training for EfficientNetV2-S.

Phase 1: backbone frozen, train head only  (epochs_phase1, lr_phase1)
Phase 2: full fine-tuning                  (epochs_phase2, lr_phase2)

Usage:
    python src/train.py --config configs/train_config.yaml
"""

import argparse
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import get_dataloaders
from model import get_model


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Logger helpers
# ---------------------------------------------------------------------------

def init_logger(cfg):
    """Initialise W&B or fall back to TensorBoard."""
    if cfg.logger == "wandb":
        try:
            import wandb
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.get("wandb_entity", None),
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            return "wandb"
        except Exception as e:
            print(f"[logger] W&B init failed ({e}), falling back to TensorBoard.")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(Path(cfg.checkpoint_dir) / "tb_logs"))
    return writer


def log_metrics(logger, metrics: dict, step: int) -> None:
    """Log a dict of metrics to whichever backend is active."""
    if logger == "wandb":
        import wandb
        wandb.log(metrics, step=step)
    else:
        for k, v in metrics.items():
            logger.add_scalar(k, v, step)


# ---------------------------------------------------------------------------
# One epoch helpers
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, scaler, device, training: bool):
    """
    Run one epoch (train or eval).

    Returns:
        (avg_loss, accuracy) tuple.
    """
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, leave=False, desc="train" if training else "val"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=(scaler is not None)):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Training phase
# ---------------------------------------------------------------------------

def train_phase(
    model, train_loader, val_loader, cfg, phase: int, logger, global_step: int
) -> Tuple[float, int]:
    """
    Train for one phase.

    Args:
        model:        The classifier.
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader.
        cfg:          OmegaConf config.
        phase:        1 or 2.
        logger:       W&B string or TensorBoard SummaryWriter.
        global_step:  Current global step (epoch counter) before this phase.

    Returns:
        (best_val_acc, global_step_after_phase)
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    lr = cfg.lr_phase1 if phase == 1 else cfg.lr_phase2
    epochs = cfg.epochs_phase1 if phase == 1 else cfg.epochs_phase2

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler() if device.type == "cuda" else None
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    ckpt_path = Path(cfg.checkpoint_dir) / cfg.checkpoint_name
    best_val_acc = 0.0

    print(f"\n{'='*60}")
    print(f"  Phase {phase}  |  epochs={epochs}  lr={lr}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, training=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, None, device, training=False
        )
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        metrics = {
            f"phase{phase}/train_loss": train_loss,
            f"phase{phase}/train_acc":  train_acc,
            f"phase{phase}/val_loss":   val_loss,
            f"phase{phase}/val_acc":    val_acc,
            f"phase{phase}/lr":         current_lr,
        }
        log_metrics(logger, metrics, step=global_step + epoch)

        print(
            f"  epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f} | "
            f"lr={current_lr:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": global_step + epoch,
                    "phase": phase,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                },
                ckpt_path,
            )
            print(f"    ✓ checkpoint saved  (val_acc={val_acc:.4f})")

    return best_val_acc, global_step + epochs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train EfficientNetV2-S classifier")
    parser.add_argument("--config", default="configs/train_config.yaml",
                        help="Path to OmegaConf YAML config")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(cfg.get("seed", 42))

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[train] device = {device}")

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders(
        cfg.data_dir, cfg.batch_size, cfg.num_workers
    )

    model = get_model(num_classes=cfg.num_classes, freeze_backbone=True).to(device)
    logger = init_logger(cfg)

    # --- Phase 1: head only ---
    best1, step = train_phase(model, train_loader, val_loader, cfg, phase=1,
                               logger=logger, global_step=0)

    # --- Phase 2: full fine-tuning ---
    model.unfreeze_backbone()
    best2, _ = train_phase(model, train_loader, val_loader, cfg, phase=2,
                            logger=logger, global_step=step)

    print(f"\n[train] Done. Best val_acc: phase1={best1:.4f}  phase2={best2:.4f}")

    if logger != "wandb":
        logger.close()
    else:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
