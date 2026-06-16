"""Configuration helpers for the local audio forgery pipeline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "dataset": {
        "root": "dataset/release_in_the_wild",
        "sample_rate": 16000,
        "duration": 5.0,
        "extensions": [".wav", ".mp3", ".flac"],
        "seed": 42,
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "num_workers": 2,
    },
    "preprocessing": {
        "n_mels": 128,
        "n_mfcc": 20,
        "n_fft": 2048,
        "hop_length": 512,
        "target_size": [224, 224],
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
        "spec_augment": {
            "time_mask_width": 20,
            "freq_mask_width": 10,
            "num_time_masks": 2,
            "num_freq_masks": 2,
        },
    },
    "model": {
        "backbone": "resnet50",
        "pretrained": False,
        "cbam_reduction_ratio": 16,
        "cbam_kernel_size": 7,
        "se_reduction_ratio": 16,
        "transformer_heads": 8,
        "transformer_ff_dim": 4096,
        "transformer_dropout": 0.1,
        "transformer_layers": 1,
        "num_classes": 2,
        "fc_hidden_dim": 512,
        "dropout_rate": 0.5,
    },
    "training": {
        "seed": 42,
        "epochs": 20,
        "batch_size": 16,
        "learning_rate": 1.0e-4,
        "weight_decay": 1.0e-4,
        "grad_clip_norm": 1.0,
        "scheduler_T_max": 20,
        "scheduler_eta_min": 1.0e-6,
        "mixed_precision": True,
        "early_stopping_patience": 5,
        "imbalance_threshold": 0.20,
        "resume": True,
    },
    "svm": {
        "cv": 3,
        "n_jobs": -1,
        "max_grid_samples": 6000,
        "max_final_samples": 12000,
        "C": [0.01, 0.1, 1, 10, 100],
        "gamma": ["scale", "auto"],
        "kernel": ["linear", "rbf"],
    },
    "paths": {
        "cache_dir": "cache",
        "output_dir": "outputs",
        "checkpoint_dir": "outputs/resnet_checkpoints",
        "best_model_dir": "outputs/best_model",
        "log_dir": "outputs/logs",
    },
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config if present and merge it with production defaults."""
    cfg = deepcopy(DEFAULT_CONFIG)
    if config_path and Path(config_path).exists():
        with Path(config_path).open("r", encoding="utf-8") as handle:
            cfg = _merge(cfg, yaml.safe_load(handle) or {})
    for directory in cfg["paths"].values():
        Path(directory).mkdir(parents=True, exist_ok=True)
    return cfg
