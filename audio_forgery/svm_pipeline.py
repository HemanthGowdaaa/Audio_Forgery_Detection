"""Optimized SVM baseline training and evaluation."""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from joblib import parallel_backend
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from audio_forgery.data import AudioSample
from audio_forgery.evaluation import compute_metrics, curve_data, save_json
from audio_forgery.features import cached_svm_feature

LOGGER = logging.getLogger(__name__)


def _matrix(samples: list[AudioSample], cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    features = [cached_svm_feature(sample.path, cfg) for sample in samples]
    labels = [sample.label for sample in samples]
    return np.vstack(features), np.array(labels, dtype=np.int64)


def _stratified_cap(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic stratified subset when full SVC fitting is too large."""
    if not max_samples or len(y) <= max_samples:
        return X, y
    _, X_sub, _, y_sub = train_test_split(
        X,
        y,
        test_size=max_samples,
        random_state=seed,
        stratify=y,
    )
    LOGGER.info("Using stratified SVM subset: %s/%s samples", len(y_sub), len(y))
    return X_sub, y_sub


def train_svm(splits: dict[str, list[AudioSample]], cfg: dict) -> dict[str, Any]:
    """Train Linear/RBF SVM with GridSearchCV and evaluate on the shared test split."""
    started = time.perf_counter()
    train_samples = splits["train"] + splits["val"]
    X_train, y_train = _matrix(train_samples, cfg)
    X_test, y_test = _matrix(splits["test"], cfg)
    X_grid, y_grid = _stratified_cap(
        X_train,
        y_train,
        int(cfg["svm"].get("max_grid_samples", 6000)),
        int(cfg["dataset"].get("seed", 42)),
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(probability=True, class_weight="balanced", random_state=42)),
    ])
    grid = {
        "svm__kernel": cfg["svm"]["kernel"],
        "svm__C": cfg["svm"]["C"],
        "svm__gamma": cfg["svm"]["gamma"],
    }
    search = GridSearchCV(
        pipe,
        grid,
        cv=int(cfg["svm"].get("cv", 3)),
        scoring="f1",
        n_jobs=int(cfg["svm"].get("n_jobs", -1)),
        refit=True,
        verbose=1,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.svm")
        with parallel_backend("threading"):
            search.fit(X_grid, y_grid)
    best_params = {
        key.replace("svm__", ""): value
        for key, value in search.best_params_.items()
    }
    X_final, y_final = _stratified_cap(
        X_train,
        y_train,
        int(cfg["svm"].get("max_final_samples", 12000)),
        int(cfg["dataset"].get("seed", 42)),
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            probability=True,
            class_weight="balanced",
            random_state=42,
            **best_params,
        )),
    ])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.svm")
        model.fit(X_final, y_final)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_score)
    metrics["best_params"] = search.best_params_
    metrics["grid_train_samples"] = int(len(y_grid))
    metrics["final_train_samples"] = int(len(y_final))
    metrics["training_time_sec"] = round(time.perf_counter() - started, 3)
    metrics["model_type"] = "svm"
    metrics["curves"] = curve_data(y_test, y_score)

    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "svm_model.joblib"
    joblib.dump(model, model_path)
    save_json(metrics, output_dir / "svm_metrics.json")
    LOGGER.info("Saved SVM model to %s", model_path)
    return {"model": model, "model_path": str(model_path), "metrics": metrics}
