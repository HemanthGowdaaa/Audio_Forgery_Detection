"""ResNet++ training and evaluation on the local dataset."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.amp import GradScaler, autocast

from audio_forgery.data import AudioSample, build_resnet_dataloaders
from audio_forgery.evaluation import compute_metrics, curve_data, save_json

from models.model import build_model
from models.utils import EarlyStopping, count_parameters, get_device, set_seed

LOGGER = logging.getLogger(__name__)


def _amp_device(device: torch.device) -> str:
    return "cuda" if device.type == "cuda" else "cpu"


def _use_amp(cfg: dict, device: torch.device) -> bool:
    return bool(cfg["training"].get("mixed_precision", True)) and device.type == "cuda"


def _criterion(train_samples: list[AudioSample], cfg: dict, device: torch.device) -> nn.Module:
    labels = np.array([sample.label for sample in train_samples])
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    imbalance = abs(counts[0] - counts[1]) / max(float(counts.sum()), 1.0)
    if imbalance >= float(cfg["training"].get("imbalance_threshold", 0.2)):
        weights = len(labels) / (2.0 * np.maximum(counts, 1.0))
        LOGGER.info("Class imbalance detected; using weighted CE: %s", weights)
        return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    return nn.CrossEntropyLoss()


def _epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    cfg: dict,
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    preds: list[int] = []
    labels_all: list[int] = []
    use_amp = _use_amp(cfg, device)
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(_amp_device(device), enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip_norm"])
            optimizer.step()
        losses.append(float(loss.item()))
        preds.extend(torch.argmax(logits.detach(), dim=1).cpu().numpy().tolist())
        labels_all.extend(y.cpu().numpy().tolist())
    metrics = compute_metrics(np.array(labels_all), np.array(preds), np.array(preds, dtype=float))
    return {"loss": float(np.mean(losses)), "accuracy": metrics["accuracy"], "f1_score": metrics["f1_score"]}


@torch.no_grad()
def evaluate_resnet_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: dict,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a ResNet model and return metrics plus arrays."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    use_amp = _use_amp(cfg, device)
    for x, y in loader:
        x = x.to(device)
        with autocast(_amp_device(device), enabled=use_amp):
            logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(preds.tolist())
        y_score.extend(probs.tolist())
    true = np.array(y_true)
    pred = np.array(y_pred)
    score = np.array(y_score)
    metrics = compute_metrics(true, pred, score)
    metrics["curves"] = curve_data(true, score)
    return metrics, true, pred, score


def train_resnet(splits: dict[str, list[AudioSample]], cfg: dict) -> dict[str, Any]:
    """Train ResNet++ with mixed precision, clipping, early stopping, and resume."""
    started = time.perf_counter()
    set_seed(int(cfg["training"]["seed"]))
    device = get_device()
    loaders = build_resnet_dataloaders(splits, cfg)
    model = build_model(cfg, device)
    total, trainable = count_parameters(model)
    LOGGER.info("ResNet++ parameters: total=%s trainable=%s", total, trainable)
    criterion = _criterion(splits["train"], cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg["training"]["scheduler_T_max"]),
        eta_min=float(cfg["training"]["scheduler_eta_min"]),
    )
    scaler = GradScaler("cuda", enabled=_use_amp(cfg, device))
    stopper = EarlyStopping(patience=int(cfg["training"]["early_stopping_patience"]), mode="max")
    checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best_resnet.pth"
    start_epoch = 0
    best_f1 = -1.0
    history: list[dict[str, Any]] = []

    if cfg["training"].get("resume", True) and best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_f1 = float(checkpoint.get("metrics", {}).get("f1_score", -1.0))
        LOGGER.info("Resumed ResNet++ from %s at epoch %s", best_path, start_epoch)

    for epoch in range(start_epoch, int(cfg["training"]["epochs"])):
        train_metrics = _epoch(model, loaders["train"], optimizer, criterion, scaler, device, cfg)
        val_metrics, _, _, _ = evaluate_resnet_model(model, loaders["val"], device, cfg)
        scheduler.step()
        row = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
        history.append(row)
        LOGGER.info("Epoch %s train=%s val=%s", epoch + 1, train_metrics, val_metrics)
        if val_metrics["f1_score"] > best_f1:
            best_f1 = val_metrics["f1_score"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "metrics": val_metrics,
                "cfg": cfg,
            }, best_path)
        if stopper(val_metrics["f1_score"]):
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, _, _, _ = evaluate_resnet_model(model, loaders["test"], device, cfg)
    test_metrics["training_time_sec"] = round(time.perf_counter() - started, 3)
    test_metrics["model_type"] = "resnet"
    test_metrics["history"] = history
    test_metrics["parameters_total"] = total
    test_metrics["parameters_trainable"] = trainable
    save_json(test_metrics, Path(cfg["paths"]["output_dir"]) / "resnet_metrics.json")
    return {"model": model, "model_path": str(best_path), "metrics": test_metrics}
