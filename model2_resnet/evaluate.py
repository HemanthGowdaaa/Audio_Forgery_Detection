"""
evaluate.py
===========
Comprehensive evaluation of the ResNet++ Audio Forgery Detection model.

Computes and reports:
  - Accuracy
  - Precision, Recall, F1 Score (binary)
  - ROC-AUC
  - Equal Error Rate (EER)
  - False Positive Rate (FPR)
  - True Negative Rate (TNR / Specificity)
  - Confusion Matrix (with visualization)
  - ROC Curve plot
  - Precision-Recall Curve plot

All plots and metrics are saved to the outputs/ directory.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.cuda.amp import autocast

from dataset import build_datasets, build_dataloaders
from model import build_model
from utils import (
    compute_eer,
    format_metrics,
    get_device,
    load_checkpoint,
    load_config,
    save_metrics_json,
    set_seed,
    setup_logger,
)

logger = logging.getLogger("resnet_forgery.evaluate")


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_evaluation(
    model:  nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg:    dict,
    split:  str = "test",
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference over all batches and collect predictions.

    Args:
        model:  Trained model in eval mode.
        loader: DataLoader for the evaluation split.
        device: Target device.
        cfg:    Config dict.
        split:  Name of the split (for logging).

    Returns:
        Tuple of:
          - metrics dict
          - all_labels  (N,)  int array
          - all_preds   (N,)  int array
          - all_probs   (N,)  float array  P(fake)
    """
    model.eval()

    all_labels, all_preds, all_probs = [], [], []

    use_amp = cfg["training"]["mixed_precision"] and device.type == "cuda"

    logger.info(f"Evaluating on '{split}' split ({len(loader.dataset)} samples) …")

    for batch_idx, (specs, labels) in enumerate(loader):
        specs  = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(specs)

        probs = torch.softmax(logits, dim=1)[:, 1]    # P(fake)
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    all_labels = np.array(all_labels, dtype=np.int32)
    all_preds  = np.array(all_preds,  dtype=np.int32)
    all_probs  = np.array(all_probs,  dtype=np.float32)

    # ── Compute metrics
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    rec  = recall_score(all_labels,  all_preds, average="binary", zero_division=0)
    f1   = f1_score(all_labels,  all_preds, average="binary", zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5

    eer = compute_eer(all_labels, all_probs)

    # Confusion matrix entries
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, cm[0, 0])

    fpr_val = fp / (fp + tn + 1e-9)   # False Positive Rate
    tnr_val = tn / (tn + fp + 1e-9)   # True Negative Rate (Specificity)

    metrics = {
        "accuracy":           float(acc),
        "precision":          float(prec),
        "recall":             float(rec),
        "f1_score":           float(f1),
        "roc_auc":            float(auc),
        "eer":                float(eer),
        "false_positive_rate": float(fpr_val),
        "true_negative_rate":  float(tnr_val),
        "true_positives":      int(tp),
        "true_negatives":      int(tn),
        "false_positives":     int(fp),
        "false_negatives":     int(fn),
    }

    return metrics, all_labels, all_preds, all_probs


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    labels: np.ndarray,
    preds:  np.ndarray,
    output_path: str,
    class_names = ("Genuine", "Fake"),
) -> None:
    """
    Save a styled confusion matrix heatmap to disk.
    """
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label",      fontsize=13)
    ax.set_title("Confusion Matrix – ResNet++ Audio Forgery Detection", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved → {output_path}")


def plot_roc_curve(
    labels:      np.ndarray,
    probs:       np.ndarray,
    auc_score:   float,
    eer_value:   float,
    output_path: str,
) -> None:
    """
    Save an ROC curve plot with AUC annotation and EER marker.
    """
    fpr, tpr, thresholds = roc_curve(labels, probs, pos_label=1)
    fnr = 1.0 - tpr

    fig, ax = plt.subplots(figsize=(8, 6))

    # ROC curve
    ax.plot(fpr, tpr, color="#4C72B0", lw=2.5,
            label=f"ROC Curve (AUC = {auc_score:.4f})")

    # EER marker — find approximate EER point on the curve
    eer_idx = np.argmin(np.abs(fpr - fnr))
    ax.scatter(
        fpr[eer_idx], tpr[eer_idx],
        s=120, color="red", zorder=5,
        label=f"EER = {eer_value:.4f}",
    )

    # Diagonal chance line
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate",  fontsize=13)
    ax.set_title("ROC Curve – ResNet++ Audio Forgery Detection", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC curve saved → {output_path}")


def plot_precision_recall_curve(
    labels:      np.ndarray,
    probs:       np.ndarray,
    f1_score_val: float,
    output_path: str,
) -> None:
    """
    Save a Precision-Recall curve plot.
    """
    precision, recall, _ = precision_recall_curve(labels, probs, pos_label=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#DD8452", lw=2.5,
            label=f"PR Curve (F1 = {f1_score_val:.4f})")
    ax.set_xlabel("Recall",    fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision-Recall Curve – ResNet++ Audio Forgery Detection", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"PR curve saved → {output_path}")


def print_classification_report(
    labels: np.ndarray,
    preds:  np.ndarray,
    class_names = ("Genuine", "Fake"),
) -> None:
    """Print sklearn classification report."""
    report = classification_report(
        labels, preds,
        target_names=list(class_names),
        digits=4,
    )
    logger.info(f"\nClassification Report:\n{report}")


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate(
    config_path: str = "configs/config.yaml",
    checkpoint_path: Optional[str] = None,
    split: str = "test",
) -> Dict[str, float]:
    """
    Complete evaluation pipeline.

    Args:
        config_path:     Path to YAML config.
        checkpoint_path: Path to model checkpoint (.pth).
                         Defaults to checkpoints/best_model.pth.
        split:           Which data split to evaluate ('val' or 'test').

    Returns:
        Final metrics dict.
    """
    cfg = load_config(config_path)
    setup_logger("resnet_forgery", log_dir=cfg["paths"]["log_dir"])
    set_seed(cfg["training"]["seed"])

    out_dir  = cfg["paths"]["output_dir"]
    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    device = get_device()

    # ── Load datasets (we only need the requested split)
    logger.info("Building datasets …")
    datasets    = build_datasets(cfg)
    dataloaders = build_dataloaders(datasets, cfg, use_weighted_sampler=False)
    loader      = dataloaders[split]

    # ── Build model
    model = build_model(cfg, device)

    # ── Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = str(Path(ckpt_dir) / "best_model.pth")

    model, _, _ = load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    # ── Run evaluation
    metrics, labels, preds, probs = run_evaluation(model, loader, device, cfg, split)

    # ── Report
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation Results [{split} split]:")
    logger.info(format_metrics(metrics))

    print_classification_report(labels, preds)

    # ── Save metrics JSON
    json_path = save_metrics_json(metrics, out_dir, f"{split}_metrics.json")
    logger.info(f"Metrics saved → {json_path}")

    # ── Save plots
    plot_confusion_matrix(
        labels, preds,
        output_path=str(Path(out_dir) / f"{split}_confusion_matrix.png"),
    )

    plot_roc_curve(
        labels, probs,
        auc_score=metrics["roc_auc"],
        eer_value=metrics["eer"],
        output_path=str(Path(out_dir) / f"{split}_roc_curve.png"),
    )

    plot_precision_recall_curve(
        labels, probs,
        f1_score_val=metrics["f1_score"],
        output_path=str(Path(out_dir) / f"{split}_pr_curve.png"),
    )

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the ResNet++ Audio Forgery Detection model"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (.pth). Defaults to checkpoints/best_model.pth",
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["val", "test"],
        help="Which split to evaluate on",
    )
    args = parser.parse_args()

    evaluate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        split=args.split,
    )
