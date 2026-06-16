"""
train.py
========
Training loop for the ResNet++ Audio Forgery Detection framework.

Features:
  - Mixed-precision training (torch.cuda.amp / torch.amp)
  - Gradient clipping
  - CosineAnnealingLR scheduler
  - Early stopping
  - TensorBoard logging
  - Best-model checkpoint saving
  - Per-epoch metric reporting (accuracy, F1, AUC)
  - Class-weighted CrossEntropy for imbalanced data
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from utils import (
    AverageMeter,
    EarlyStopping,
    format_metrics,
    load_config,
    save_checkpoint,
    save_metrics_json,
    set_seed,
    get_device,
    count_parameters,
)
from dataset import build_datasets, build_dataloaders
from model import build_model

logger = logging.getLogger("resnet_forgery.train")


# ---------------------------------------------------------------------------
# One-epoch training step
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler:    GradScaler,
    device:    torch.device,
    cfg:       dict,
    epoch:     int,
    writer:    Optional[SummaryWriter] = None,
) -> Dict[str, float]:
    """
    Run one full training epoch with mixed-precision support.

    Args:
        model:     The ResNetPlusPlus model (in train mode).
        loader:    Training DataLoader.
        optimizer: AdamW optimizer.
        criterion: CrossEntropyLoss (possibly with class weights).
        scaler:    GradScaler for AMP.
        device:    Target device.
        cfg:       Config dict.
        epoch:     Current epoch index (for logging).
        writer:    TensorBoard SummaryWriter (optional).

    Returns:
        Dict with keys: 'loss', 'accuracy', 'f1'.
    """
    model.train()

    loss_meter = AverageMeter("train_loss")
    all_preds, all_labels = [], []

    grad_clip = cfg["training"]["grad_clip_norm"]
    log_every = cfg["logging"]["log_every_n_steps"]
    use_amp   = cfg["training"]["mixed_precision"] and device.type in ("cuda",)

    for step, (specs, labels) in enumerate(loader):
        specs  = specs.to(device, non_blocking=True)      # [B, 3, 224, 224]
        labels = labels.to(device, non_blocking=True)     # [B]

        optimizer.zero_grad(set_to_none=True)

        # ── Forward pass with optional AMP
        with autocast(enabled=use_amp):
            logits = model(specs)                          # [B, 2]
            loss   = criterion(logits, labels)

        # ── Backward + gradient clipping + optimizer step
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # ── Accumulate metrics
        loss_meter.update(loss.item(), n=specs.size(0))
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

        # ── Step-level TensorBoard logging
        global_step = epoch * len(loader) + step
        if writer and (step % log_every == 0):
            writer.add_scalar("train/step_loss", loss.item(), global_step)

        if step % max(1, len(loader) // 5) == 0:
            logger.info(
                f"Epoch [{epoch}] Step [{step}/{len(loader)}] "
                f"Loss: {loss_meter.avg:.4f}"
            )

    # ── Epoch-level metrics
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="binary", zero_division=0)

    metrics = {"loss": loss_meter.avg, "accuracy": acc, "f1": f1}

    if writer:
        for k, v in metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)

    return metrics


# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    cfg:       dict,
    epoch:     int,
    writer:    Optional[SummaryWriter] = None,
) -> Dict[str, float]:
    """
    Evaluate the model on a validation (or test) split.

    Args:
        model:     The ResNetPlusPlus model (in eval mode).
        loader:    Val / Test DataLoader.
        criterion: CrossEntropyLoss.
        device:    Target device.
        cfg:       Config dict.
        epoch:     Current epoch (for TensorBoard).
        writer:    TensorBoard SummaryWriter (optional).

    Returns:
        Dict with loss, accuracy, precision, recall, f1, auc.
    """
    model.eval()

    loss_meter = AverageMeter("val_loss")
    all_preds, all_probs, all_labels = [], [], []

    use_amp = cfg["training"]["mixed_precision"] and device.type in ("cuda",)

    for specs, labels in loader:
        specs  = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(specs)
            loss   = criterion(logits, labels)

        probs  = torch.softmax(logits, dim=1)[:, 1]    # P(fake)
        preds  = logits.argmax(dim=1)

        loss_meter.update(loss.item(), n=specs.size(0))
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    # ── Compute metrics
    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    rec  = recall_score(all_labels, all_preds, average="binary", zero_division=0)
    f1   = f1_score(all_labels, all_preds, average="binary", zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5   # Only one class present

    metrics = {
        "loss":      loss_meter.avg,
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "auc":       auc,
    }

    if writer:
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)

    return metrics


# ---------------------------------------------------------------------------
# Main training orchestrator
# ---------------------------------------------------------------------------

def train(config_path: str = "configs/config.yaml") -> None:
    """
    Full training pipeline:
      1. Load config & set seed
      2. Build datasets + dataloaders
      3. Build model, optimizer, scheduler, criterion
      4. Train for N epochs with early stopping
      5. Save best checkpoint + final metrics JSON
    """
    from utils import setup_logger

    cfg = load_config(config_path)
    tr  = cfg["training"]
    log_dir  = cfg["paths"]["log_dir"]
    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    out_dir  = cfg["paths"]["output_dir"]

    # ── Logger + TensorBoard
    setup_logger("resnet_forgery", log_dir=log_dir)
    writer = (
        SummaryWriter(log_dir=log_dir)
        if cfg["logging"]["tensorboard"]
        else None
    )

    # ── Reproducibility
    set_seed(tr["seed"])
    logger.info(f"Random seed: {tr['seed']}")

    # ── Device
    device = get_device()
    logger.info(f"Device: {device}")

    # ── Data
    logger.info("Building datasets …")
    datasets   = build_datasets(cfg)
    dataloaders = build_dataloaders(datasets, cfg, use_weighted_sampler=True)

    train_loader = dataloaders["train"]
    val_loader   = dataloaders["val"]

    # ── Model
    logger.info("Building model …")
    model = build_model(cfg, device)
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    if writer:
        writer.add_text("model/summary", str(model))

    # ── Loss: class-weighted CrossEntropy
    class_weights = datasets["train"].get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    logger.info(f"Class weights: {class_weights.cpu().numpy()}")

    # ── Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tr["learning_rate"],
        weight_decay=tr["weight_decay"],
    )

    # ── Scheduler: CosineAnnealingLR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=tr["scheduler_T_max"],
        eta_min=tr["scheduler_eta_min"],
    )

    # ── Mixed precision scaler (CUDA only)
    scaler = GradScaler(enabled=(tr["mixed_precision"] and device.type == "cuda"))

    # ── Early stopping
    early_stopper = EarlyStopping(
        patience=tr["early_stopping_patience"],
        mode="max",    # We monitor val_f1; higher is better
        verbose=True,
    )

    # ── Training state
    best_metrics: Dict[str, float] = {}
    best_val_f1  = -1.0
    start_epoch  = 0

    # ── Resume from checkpoint (if exists)
    best_ckpt = Path(ckpt_dir) / "best_model.pth"
    if best_ckpt.exists():
        logger.info(f"Resuming from checkpoint: {best_ckpt}")
        from utils import load_checkpoint
        model, start_epoch, best_metrics = load_checkpoint(
            str(best_ckpt), model, optimizer, scheduler, device
        )
        best_val_f1 = best_metrics.get("f1", -1.0)

    # ================================================================
    # Training loop
    # ================================================================
    logger.info(f"Starting training for {tr['epochs']} epochs …")

    for epoch in range(start_epoch, tr["epochs"]):
        epoch_start = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{tr['epochs']}  |  LR: {scheduler.get_last_lr()}")
        logger.info(f"{'='*60}")

        # ── Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, cfg, epoch, writer,
        )

        # ── Validate
        val_metrics = validate(
            model, val_loader, criterion, device, cfg, epoch, writer
        )

        # ── Scheduler step
        scheduler.step()

        # ── Log LR to TensorBoard
        if writer:
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        # ── Print epoch summary
        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch + 1} complete in {epoch_time:.1f}s\n"
            f"  Train → {format_metrics(train_metrics)}\n"
            f"  Val   → {format_metrics(val_metrics)}"
        )

        # ── Save checkpoint
        val_f1 = val_metrics["f1"]
        is_best = val_f1 > best_val_f1

        if is_best:
            best_val_f1 = val_f1
            best_metrics = val_metrics

        save_checkpoint(
            state={
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "metrics":              val_metrics,
                "cfg":                  cfg,
            },
            checkpoint_dir=ckpt_dir,
            filename=f"checkpoint_epoch_{epoch:03d}.pth",
            is_best=is_best,
        )

        # ── Early stopping check
        if early_stopper(val_f1):
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    # ================================================================
    # Training complete
    # ================================================================
    logger.info("\nTraining finished!")
    logger.info(f"Best validation metrics:\n{format_metrics(best_metrics)}")

    # Save best metrics to JSON
    json_path = save_metrics_json(best_metrics, out_dir, "best_val_metrics.json")
    logger.info(f"Best metrics saved to: {json_path}")

    if writer:
        writer.flush()
        writer.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ResNet++ Audio Forgery Detector")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    train(config_path=args.config)
