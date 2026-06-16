"""Evaluation metrics and curve serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    """Compute binary classification metrics for fake-class probability scores."""
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["REAL", "FAKE"],
            output_dict=True,
            zero_division=0,
        ),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        metrics["roc_auc"] = 0.5
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    except ValueError:
        metrics["pr_auc"] = 0.0
    return metrics


def curve_data(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, list[float]]:
    """Return ROC and precision-recall curve points."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
    except ValueError:
        fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
    except ValueError:
        precision, recall = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    return {
        "roc_fpr": fpr.astype(float).tolist(),
        "roc_tpr": tpr.astype(float).tolist(),
        "pr_precision": precision.astype(float).tolist(),
        "pr_recall": recall.astype(float).tolist(),
    }


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Save JSON with stable indentation."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
