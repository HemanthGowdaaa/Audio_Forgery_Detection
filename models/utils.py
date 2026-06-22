import os
import random
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

LOGGER = logging.getLogger("resnet_forgery.utils")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            LOGGER.info(f"Using CUDA device: {gpu_name}")
            return device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            LOGGER.info("Using Apple Silicon MPS device")
            return device
    device = torch.device("cpu")
    LOGGER.info("Using CPU device")
    return device


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 1e-4, verbose: bool = True):
        assert mode in ("max", "min"), "mode must be 'max' or 'min'"
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_value: Optional[float] = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
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
                LOGGER.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    LOGGER.info(f"Early stopping triggered after {self.patience} epochs.")

        return self.should_stop


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
