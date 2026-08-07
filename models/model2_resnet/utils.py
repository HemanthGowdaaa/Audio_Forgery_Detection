"""
utils.py
========
Shared utility functions for the ResNet++ Audio Forgery Detection framework.

Includes:
  - Reproducibility seeding
  - Device selection (CUDA / MPS / CPU)
  - YAML config loading
  - Logging setup
  - AverageMeter for tracking running metrics
  - Equal Error Rate (EER) computation
  - Checkpoint save / load helpers
  - Pretty-print metric dicts
"""

import os
import random
import logging
import math
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behaviour (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Device Selection
# ---------------------------------------------------------------------------

def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Return the best available device in priority order:
      1. CUDA GPU
      2. Apple Silicon MPS
      3. CPU
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            logging.info(f"Using CUDA device: {gpu_name}")
            return device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("Using Apple Silicon MPS device")
            return device
    device = torch.device("cpu")
    logging.info("Using CPU device")
    return device


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load a YAML configuration file and return it as a nested dict."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def flatten_config(cfg: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested config dict for easy TensorBoard logging."""
    flat = {}
    for k, v in cfg.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_config(v, prefix=key))
        else:
            flat[key] = v
    return flat


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def setup_logger(
    name: str = "resnet_forgery",
    log_dir: str = "./logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure a logger that writes to both stdout and a rotating file.

    Args:
        name:    Logger name (used in log records).
        log_dir: Directory where the log file will be written.
        level:   Logging level (default INFO).

    Returns:
        Configured Logger instance.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# AverageMeter – running statistics
# ---------------------------------------------------------------------------

class AverageMeter:
    """Computes and stores the average and current value of a metric."""

    def __init__(self, name: str = "metric"):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


# ---------------------------------------------------------------------------
# Equal Error Rate (EER)
# ---------------------------------------------------------------------------

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute the Equal Error Rate (EER).

    EER is the point on the ROC curve where FAR (False Accept Rate)
    equals FRR (False Reject Rate). Lower EER ⟹ better model.

    Args:
        y_true:   Ground-truth binary labels (0 = genuine, 1 = fake).
        y_scores: Predicted probability of the positive class (fake).

    Returns:
        EER value in [0, 1].
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr  # False Negative Rate

    # Find the EER via interpolation
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    except ValueError:
        # Fallback: point where |FPR - FNR| is minimal
        eer = float(np.min(np.abs(fpr - fnr)))

    return float(eer)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "checkpoint.pth",
    is_best: bool = False,
) -> str:
    """
    Save a training checkpoint to disk.

    Args:
        state:          Dict containing model/optimizer state dicts + metadata.
        checkpoint_dir: Directory to write checkpoints to.
        filename:       Checkpoint filename.
        is_best:        If True, also saves a 'best_model.pth' copy.

    Returns:
        Path of the saved checkpoint.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(checkpoint_dir) / filename
    torch.save(state, save_path)

    if is_best:
        best_path = Path(checkpoint_dir) / "best_model.pth"
        torch.save(state, best_path)
        logging.info(f"New best model saved → {best_path}")

    return str(save_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, int, Dict[str, float]]:
    """
    Load a checkpoint and restore model (and optionally optimizer/scheduler) state.

    Args:
        checkpoint_path: Path to the .pth file.
        model:           Model to load weights into.
        optimizer:       Optional optimizer to restore state.
        scheduler:       Optional LR scheduler to restore state.
        device:          Target device (defaults to current model device).

    Returns:
        Tuple of (model, start_epoch, best_metrics).
    """
    if device is None:
        device = next(model.parameters()).device

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    best_metrics = checkpoint.get("metrics", {})

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logging.info(
        f"Checkpoint loaded from '{checkpoint_path}' "
        f"(epoch={checkpoint.get('epoch', '?')}, "
        f"metrics={best_metrics})"
    )
    return model, start_epoch, best_metrics


# ---------------------------------------------------------------------------
# Metric pretty-printer
# ---------------------------------------------------------------------------

def format_metrics(metrics: Dict[str, float], width: int = 60) -> str:
    """Return a nicely formatted multi-line string of metric name → value."""
    sep = "─" * width
    lines = [sep]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k:<30s} {v:.4f}")
        else:
            lines.append(f"  {k:<30s} {v}")
    lines.append(sep)
    return "\n".join(lines)


def save_metrics_json(metrics: Dict[str, Any], output_dir: str, filename: str) -> str:
    """Serialize a metrics dict to JSON on disk."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / filename
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return str(path)


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Monitors a validation metric and triggers early stopping when the metric
    stops improving.

    Args:
        patience:   Number of epochs to wait for improvement before stopping.
        mode:       'max' if higher metric is better (e.g. F1), else 'min'.
        min_delta:  Minimum change to qualify as an improvement.
        verbose:    Print info when patience counter increases.
    """

    def __init__(
        self,
        patience: int = 10,
        mode: str = "max",
        min_delta: float = 1e-4,
        verbose: bool = True,
    ):
        assert mode in ("max", "min"), "mode must be 'max' or 'min'"
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_value: Optional[float] = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """
        Call after each epoch.

        Returns:
            True if training should be stopped, False otherwise.
        """
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logging.info(
                    f"EarlyStopping counter: {self.counter}/{self.patience}"
                )
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    logging.info(
                        f"Early stopping triggered after {self.patience} epochs "
                        f"without improvement."
                    )

        return self.should_stop


# ---------------------------------------------------------------------------
# Utility: count model parameters
# ---------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters of a model.

    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
