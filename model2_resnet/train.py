"""
train.py  (M2-Optimized)
=========================
Training loop for the ResNet++ Audio Forgery Detection framework.
Optimized for MacBook Air M2 (8 GB RAM, no CUDA GPU).

CHANGES vs. original train.py:
  1. **Gradient Accumulation**: accumulates gradients over
     `cfg.training.gradient_accumulation_steps` mini-batches before stepping
     the optimizer.  This simulates a larger effective batch size without
     holding multiple batches of activations in RAM simultaneously.

  2. **MPS/CPU-aware Mixed Precision**:
     - CUDA: torch.autocast("cuda") + GradScaler (full AMP)
     - MPS:  torch.autocast("cpu", dtype=torch.bfloat16) — MPS does not
             support float16 AMP; bfloat16 on CPU for softmax/matmul paths
     - CPU:  torch.autocast("cpu", dtype=torch.bfloat16)
     GradScaler is disabled for non-CUDA devices (it only works with CUDA).

  3. **Explicit tensor cleanup**: `del specs, labels, logits, loss` + gc after
     each batch to release activation tensors immediately.

  4. **Best-only checkpointing**: saves only `best_model.pth` (not a new file
     every epoch) when `save_best_only=True` in config.  This avoids filling
     the 256 GB SSD and eliminates a per-epoch ~300 MB write.

  5. **Manifest-based dataset**: uses `dataset_manifest.py` instead of the
     HuggingFace `datasets` library (saves ~1–2 GB of Arrow/HF overhead).

  6. **TensorBoard optional**: disabled by default (saves ~50 MB + disk I/O).

  7. **Memory monitoring**: prints approximate process RSS at the start of each
     epoch when `psutil` is available.

  8. **Validation batch size**: automatically set to batch_size//2 (fewer peak
     activations, no gradients needed during val).
"""

import gc
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
from dataset_manifest import build_manifest_datasets, build_manifest_dataloaders
from model import build_model

logger = logging.getLogger("resnet_forgery.train")

# Optional memory monitoring
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ---------------------------------------------------------------------------
# Helper: autocast context (device-aware)
# ---------------------------------------------------------------------------

def _get_autocast_ctx(device: torch.device, enabled: bool):
    """
    Return the correct torch.autocast context for the given device.

    - CUDA:  float16 (standard AMP)
    - MPS / CPU: bfloat16 (MPS does not support fp16 autocast)
    - disabled: nullcontext (no-op)
    """
    if not enabled:
        import contextlib
        return contextlib.nullcontext()

    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        # bfloat16 is supported on CPU and MPS for basic ops
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)


def _log_memory(prefix: str = "") -> None:
    """Log approximate process memory usage if psutil is available."""
    if _HAS_PSUTIL:
        proc = psutil.Process(os.getpid())
        rss_mb = proc.memory_info().rss / 1024 / 1024
        logger.info(f"{prefix}RAM usage: {rss_mb:.0f} MB")


# ---------------------------------------------------------------------------
# One-epoch training step  (with gradient accumulation)
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    cfg:       dict,
    epoch:     int,
    writer=None,
) -> Dict[str, float]:
    """
    Run one full training epoch with:
      - Gradient accumulation (N micro-steps before optimizer.step())
      - MPS/CPU-aware mixed precision (bfloat16 autocast)
      - Explicit tensor cleanup after every batch

    Args:
        model:     ResNetPlusPlus in train mode.
        loader:    Training DataLoader.
        optimizer: AdamW optimizer.
        criterion: CrossEntropyLoss (with class weights).
        device:    Target device (mps / cpu / cuda).
        cfg:       Config dict.
        epoch:     Current epoch index.
        writer:    TensorBoard SummaryWriter (optional, may be None).

    Returns:
        Dict with keys: 'loss', 'accuracy', 'f1'.
    """
    model.train()

    tr_cfg    = cfg["training"]
    accum_steps = tr_cfg.get("gradient_accumulation_steps", 1)
    grad_clip   = tr_cfg.get("grad_clip_norm", 1.0)
    log_every   = cfg["logging"].get("log_every_n_steps", 10)

    # Mixed precision: GradScaler only useful for CUDA fp16
    use_amp = tr_cfg.get("mixed_precision", False)
    use_scaler = (use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    loss_meter = AverageMeter("train_loss")
    all_preds, all_labels = [], []

    optimizer.zero_grad(set_to_none=True)   # Clear at epoch start

    for step, (specs, labels) in enumerate(loader):
        specs  = specs.to(device, non_blocking=False)      # [B, 3, H, W]
        labels = labels.to(device, non_blocking=False)     # [B]

        # ── Forward pass with MPS-aware autocast
        with _get_autocast_ctx(device, use_amp):
            logits = model(specs)                          # [B, 2]
            # Divide loss by accum_steps so gradients average correctly
            loss   = criterion(logits, labels) / accum_steps

        # ── Backward
        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # ── Accumulate metrics (scale loss back up for display)
        loss_meter.update(loss.item() * accum_steps, n=specs.size(0))
        with torch.no_grad():
            preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

        # ── Optimizer step every accum_steps micro-batches
        if (step + 1) % accum_steps == 0:
            if use_scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # set_to_none frees grad tensors

        # ── Explicit tensor cleanup to release activations immediately
        del specs, labels, logits, loss
        if step % 10 == 0:
            gc.collect()

        # ── TensorBoard step logging
        global_step = epoch * len(loader) + step
        if writer and (step % log_every == 0):
            writer.add_scalar("train/step_loss", loss_meter.val, global_step)

        if step % max(1, len(loader) // 5) == 0:
            logger.info(
                f"Epoch [{epoch+1}] Step [{step}/{len(loader)}] "
                f"Loss: {loss_meter.avg:.4f}"
            )

    # Handle remaining gradient accumulation steps at end of epoch
    if len(loader) % accum_steps != 0:
        if use_scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

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
    writer=None,
) -> Dict[str, float]:
    """
    Evaluate the model on a validation (or test) split.

    No gradients, no scaler, minimal peak RAM.
    """
    model.eval()

    loss_meter = AverageMeter("val_loss")
    all_preds, all_probs, all_labels = [], [], []

    for specs, labels in loader:
        specs  = specs.to(device, non_blocking=False)
        labels = labels.to(device, non_blocking=False)

        logits = model(specs)
        loss   = criterion(logits, labels)

        probs = torch.softmax(logits.float(), dim=1)[:, 1]   # cast for softmax stability
        preds = logits.argmax(dim=1)

        loss_meter.update(loss.item(), n=specs.size(0))
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

        del specs, labels, logits, loss, probs, preds

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
        auc = 0.5   # Only one class present in this batch

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
      2. Build datasets (manifest-based) + dataloaders
      3. Build model, optimizer, scheduler, criterion
      4. Train for N epochs with early stopping & gradient accumulation
      5. Save best checkpoint + final metrics JSON
    """
    from utils import setup_logger

    cfg = load_config(config_path)
    tr  = cfg["training"]
    log_dir  = cfg["paths"]["log_dir"]
    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    out_dir  = cfg["paths"]["output_dir"]

    # ── Logger
    setup_logger("resnet_forgery", log_dir=log_dir)

    # ── TensorBoard (optional)
    writer = None
    if cfg["logging"].get("tensorboard", False):
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            logger.warning("TensorBoard not installed; skipping.")

    # ── Reproducibility
    set_seed(tr["seed"])
    logger.info(f"Seed: {tr['seed']}")

    # ── Device
    device = get_device()
    logger.info(f"Device: {device}")

    # Enable MPS fallback for unsupported ops (prevents crashes on M2)
    if device.type == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        logger.info("MPS fallback enabled for unsupported ops")

    _log_memory("Before dataset build — ")

    # ── Data (manifest-based: no HuggingFace overhead)
    logger.info("Building manifest datasets …")
    datasets    = build_manifest_datasets(cfg)
    dataloaders = build_manifest_dataloaders(datasets, cfg, use_weighted_sampler=True)

    train_loader = dataloaders["train"]
    val_loader   = dataloaders["val"]

    logger.info(
        f"DataLoader sizes — "
        f"train: {len(train_loader)} batches | "
        f"val: {len(val_loader)} batches"
    )

    _log_memory("After dataset build — ")

    # ── Model
    logger.info("Building model …")
    model = build_model(cfg, device)
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    _log_memory("After model build — ")

    # ── Loss: class-weighted CrossEntropy
    class_weights = datasets["train"].get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    logger.info(f"Class weights: {class_weights.cpu().numpy()}")

    # ── Optimizer (only optimizes trainable params → smaller state)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=tr["learning_rate"],
        weight_decay=tr["weight_decay"],
    )

    # ── Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=tr.get("scheduler_T_max", tr["epochs"]),
        eta_min=tr.get("scheduler_eta_min", 1e-6),
    )

    # ── Early stopping
    early_stopper = EarlyStopping(
        patience=tr.get("early_stopping_patience", 5),
        mode="max",
        verbose=True,
    )

    # ── Training state
    best_metrics: Dict[str, float] = {}
    best_val_f1  = -1.0
    start_epoch  = 0
    save_best_only = cfg["training"].get("save_best_only", True)

    # ── Resume from checkpoint
    best_ckpt = Path(ckpt_dir) / "best_model.pth"
    if tr.get("resume", False) and best_ckpt.exists():
        logger.info(f"Resuming from checkpoint: {best_ckpt}")
        from utils import load_checkpoint
        model, start_epoch, best_metrics = load_checkpoint(
            str(best_ckpt), model, optimizer, scheduler, device
        )
        best_val_f1 = best_metrics.get("f1", -1.0)

    # ================================================================
    # Training loop
    # ================================================================
    logger.info(f"Starting training — {tr['epochs']} epoch(s), "
                f"batch_size={tr['batch_size']}, "
                f"accum_steps={tr.get('gradient_accumulation_steps', 1)}, "
                f"effective_batch={tr['batch_size'] * tr.get('gradient_accumulation_steps', 1)}")

    for epoch in range(start_epoch, tr["epochs"]):
        epoch_start = time.time()
        logger.info(f"\n{'='*60}")
        lr_now = scheduler.get_last_lr()[0] if epoch > 0 else tr["learning_rate"]
        logger.info(f"Epoch {epoch + 1}/{tr['epochs']}  |  LR: {lr_now:.2e}")
        logger.info(f"{'='*60}")

        _log_memory(f"Epoch {epoch+1} start — ")

        # ── Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, cfg, epoch, writer,
        )

        # ── Validate
        val_metrics = validate(
            model, val_loader, criterion, device, cfg, epoch, writer
        )

        # ── Scheduler step
        scheduler.step()

        if writer:
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        # ── Print epoch summary
        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch + 1} done in {epoch_time:.1f}s\n"
            f"  Train → {format_metrics(train_metrics)}\n"
            f"  Val   → {format_metrics(val_metrics)}"
        )

        # ── Checkpoint: save only best model to avoid disk / RAM overhead
        val_f1  = val_metrics["f1"]
        is_best = val_f1 > best_val_f1

        if is_best:
            best_val_f1  = val_f1
            best_metrics = val_metrics

        if is_best or not save_best_only:
            state = {
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "metrics":              val_metrics,
                "cfg":                  cfg,
            }
            fname = "best_model.pth" if save_best_only else f"checkpoint_epoch_{epoch:03d}.pth"
            save_checkpoint(
                state=state,
                checkpoint_dir=ckpt_dir,
                filename=fname,
                is_best=is_best,
            )

        # ── Explicit GC between epochs
        gc.collect()
        _log_memory(f"Epoch {epoch+1} end — ")

        # ── Early stopping check
        if early_stopper(val_f1):
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # ================================================================
    # Training complete
    # ================================================================
    logger.info("\nTraining finished!")
    logger.info(f"Best validation metrics:\n{format_metrics(best_metrics)}")

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

    parser = argparse.ArgumentParser(description="Train ResNet++ Audio Forgery Detector (M2-Optimized)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    train(config_path=args.config)
