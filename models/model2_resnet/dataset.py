"""
dataset.py
==========
PyTorch Dataset for the mueller91/In-The-Wild audio deepfake dataset.

Key features:
  - Loads the HuggingFace dataset (mueller91/In-The-Wild)
  - Creates train / val / test splits deterministically
  - Preprocesses each sample through the full pipeline (preprocess.py)
  - Caches preprocessed spectrograms to disk to accelerate subsequent epochs
  - Supports on-the-fly SpecAugment for training splits
  - Exposes class weights for imbalanced-class handling
"""

import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from datasets import load_dataset, Audio as HFAudio

from preprocess import preprocess_audio

logger = logging.getLogger("resnet_forgery.dataset")


# ---------------------------------------------------------------------------
# Label mapping helpers
# ---------------------------------------------------------------------------

def _resolve_label(sample: dict) -> int:
    """
    Extract binary label from a HuggingFace In-The-Wild sample.

    The dataset uses a 'label' column where:
      'genuine' / 'bona-fide' / 0  →  0  (real speech)
      'spoof'   / 'fake'      / 1  →  1  (synthesized / tampered)

    Falls back to checking the 'speaker' column if 'label' is absent.
    """
    # Prefer an explicit 'label' column
    for col in ("label", "is_genuine", "bona_fide", "is_spoof"):
        if col in sample:
            raw = sample[col]
            # Boolean or integer
            if isinstance(raw, (bool, int, np.integer)):
                # 'is_genuine' True → 0, 'is_spoof' True → 1
                if col in ("is_genuine", "bona_fide"):
                    return 0 if bool(raw) else 1
                return int(bool(raw))
            # String labels
            if isinstance(raw, str):
                raw_lower = raw.lower()
                if raw_lower in ("genuine", "bona-fide", "bonafide", "real", "0"):
                    return 0
                if raw_lower in ("spoof", "fake", "synthetic", "1"):
                    return 1

    # Fallback: try ClassLabel integer encoding (0 = genuine, 1 = spoof)
    if "label" in sample and hasattr(sample["label"], "__int__"):
        return int(sample["label"])

    raise KeyError(
        f"Cannot determine binary label from sample keys: {list(sample.keys())}"
    )


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def _cache_key(index: int, split_name: str, cfg: dict) -> str:
    """
    Generate a deterministic filename for a cached spectrogram tensor.
    Incorporates preprocessing parameters so the cache is invalidated
    automatically when the config changes.
    """
    pre = cfg["preprocessing"]
    key_str = (
        f"{split_name}_{index}_"
        f"sr{cfg['dataset']['sample_rate']}_"
        f"dur{cfg['dataset']['max_audio_length']}_"
        f"nm{pre['n_mels']}_nfft{pre['n_fft']}_hl{pre['hop_length']}_"
        f"h{pre['target_size'][0]}_w{pre['target_size'][1]}"
    )
    return hashlib.md5(key_str.encode()).hexdigest() + ".pt"


# ---------------------------------------------------------------------------
# Core Dataset class
# ---------------------------------------------------------------------------

class InTheWildDataset(Dataset):
    """
    PyTorch Dataset wrapping the mueller91/In-The-Wild HuggingFace dataset.

    Args:
        hf_split:    HuggingFace dataset split object (already sliced).
        split_name:  Human-readable name ('train', 'val', 'test').
        cfg:         Full config dict loaded from configs/config.yaml.
        augment:     If True, apply SpecAugment during __getitem__.
        cache_dir:   Directory for caching preprocessed spectrograms.
                     Set to None to disable caching.
    """

    def __init__(
        self,
        hf_split,
        split_name: str,
        cfg: dict,
        augment: bool = False,
        cache_dir: Optional[str] = None,
    ):
        self.hf_split   = hf_split
        self.split_name = split_name
        self.cfg        = cfg
        self.augment    = augment
        self.cache_dir  = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Resolve all labels up-front (fast, labels are tiny)
        logger.info(f"[{split_name}] Resolving labels for {len(hf_split)} samples …")
        self.labels: List[int] = []
        self._valid_indices: List[int] = []

        for idx in range(len(hf_split)):
            try:
                label = _resolve_label(hf_split[idx])
                self.labels.append(label)
                self._valid_indices.append(idx)
            except (KeyError, Exception) as e:
                logger.warning(f"Skipping sample {idx}: {e}")

        genuine_count = self.labels.count(0)
        fake_count    = self.labels.count(1)
        logger.info(
            f"[{split_name}] {len(self.labels)} valid samples | "
            f"genuine={genuine_count}, fake={fake_count}"
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._valid_indices)

    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            (spectrogram_tensor, label)
            spectrogram_tensor: float32 Tensor [3, 224, 224]
            label: int  0=genuine, 1=fake
        """
        real_idx = self._valid_indices[idx]
        label    = self.labels[idx]

        # ── Try loading from cache (only non-augmented base tensor is cached)
        cache_path = None
        if self.cache_dir:
            cache_path = self.cache_dir / _cache_key(real_idx, self.split_name, self.cfg)
            if cache_path.exists():
                spec = torch.load(cache_path, weights_only=True)
                # Apply augmentation on the fly from the cached tensor
                if self.augment:
                    from preprocess import SpecAugment
                    aug_cfg = self.cfg["preprocessing"]["spec_augment"]
                    spec = SpecAugment(
                        time_mask_width=aug_cfg["time_mask_width"],
                        freq_mask_width=aug_cfg["freq_mask_width"],
                        num_time_masks=aug_cfg["num_time_masks"],
                        num_freq_masks=aug_cfg["num_freq_masks"],
                    )(spec)
                return spec, label

        # ── Load raw audio from HuggingFace
        sample = self.hf_split[real_idx]

        # HF Audio feature returns a dict: {'array': np.ndarray, 'sampling_rate': int}
        audio_data = sample.get("audio", None)
        if audio_data is None:
            # Some dataset variants expose audio differently
            raise KeyError(f"Sample {real_idx} has no 'audio' column. Keys: {list(sample.keys())}")

        waveform_np = audio_data["array"].astype(np.float32)
        sr_orig     = audio_data["sampling_rate"]

        # Inject the actual sample rate into a local config copy so
        # preprocess_audio resamples correctly when sr_orig != target_sr
        # (load_audio already handles resampling, we just need the numpy array)
        spec = preprocess_audio(
            audio_input=waveform_np,
            cfg=self.cfg,
            augment=False,          # save base tensor; augment afterwards
        )

        # ── Save base tensor to cache
        if cache_path is not None:
            torch.save(spec, cache_path)

        # ── Apply augmentation on the just-computed tensor
        if self.augment:
            from preprocess import SpecAugment
            aug_cfg = self.cfg["preprocessing"]["spec_augment"]
            spec = SpecAugment(
                time_mask_width=aug_cfg["time_mask_width"],
                freq_mask_width=aug_cfg["freq_mask_width"],
                num_time_masks=aug_cfg["num_time_masks"],
                num_freq_masks=aug_cfg["num_freq_masks"],
            )(spec)

        return spec, label

    # ------------------------------------------------------------------

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute per-class weights for use with WeightedRandomSampler or
        CrossEntropyLoss(weight=…) to handle class imbalance.

        Returns:
            Tensor of shape [num_classes] where weight_c = N / (num_classes * n_c).
        """
        num_classes = 2
        counts = np.bincount(self.labels, minlength=num_classes).astype(float)
        weights = len(self.labels) / (num_classes * (counts + 1e-6))
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        """
        Per-sample weights for WeightedRandomSampler (inverse class frequency).
        """
        class_weights = self.get_class_weights()
        return torch.tensor(
            [class_weights[label].item() for label in self.labels],
            dtype=torch.float32,
        )


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def build_datasets(cfg: dict) -> Dict[str, InTheWildDataset]:
    """
    Load the HuggingFace dataset and build train / val / test Dataset objects.

    If the HF dataset already has 'train' / 'validation' / 'test' splits, those
    are used directly. Otherwise, the full dataset is merged and split
    according to cfg['training']['train_ratio'] etc.

    Args:
        cfg: Full config dict.

    Returns:
        Dict with keys 'train', 'val', 'test' mapping to Dataset instances.
    """
    ds_cfg  = cfg["dataset"]
    tr_cfg  = cfg["training"]
    cache   = cfg["paths"]["cache_dir"]

    hf_name = ds_cfg["name"]
    logger.info(f"Loading HuggingFace dataset: {hf_name}")

    # Load the dataset; cast 'audio' column to HF Audio feature for auto-decoding
    raw = load_dataset(hf_name, trust_remote_code=True)
    # Cast audio column for consistent decoding
    for split_key in raw:
        if "audio" in raw[split_key].column_names:
            raw[split_key] = raw[split_key].cast_column(
                "audio", HFAudio(sampling_rate=ds_cfg["sample_rate"])
            )

    available_splits = list(raw.keys())
    logger.info(f"Available HF splits: {available_splits}")

    # ── Case 1: dataset already has train/validation/test
    hf_train = hf_val = hf_test = None

    if "train" in available_splits and "test" in available_splits:
        hf_train = raw["train"]
        hf_test  = raw["test"]
        if "validation" in available_splits:
            hf_val = raw["validation"]
        else:
            # Carve out validation from training set
            split = hf_train.train_test_split(test_size=tr_cfg["val_ratio"], seed=42)
            hf_train = split["train"]
            hf_val   = split["test"]

    else:
        # ── Case 2: single-split dataset → merge and resplit
        all_data = raw[available_splits[0]]
        for sk in available_splits[1:]:
            from datasets import concatenate_datasets
            all_data = concatenate_datasets([all_data, raw[sk]])

        n = len(all_data)
        indices = list(range(n))
        rng = np.random.default_rng(42)
        rng.shuffle(indices)

        n_train = int(n * tr_cfg["train_ratio"])
        n_val   = int(n * tr_cfg["val_ratio"])

        train_idx = indices[:n_train]
        val_idx   = indices[n_train : n_train + n_val]
        test_idx  = indices[n_train + n_val :]

        hf_train = all_data.select(train_idx)
        hf_val   = all_data.select(val_idx)
        hf_test  = all_data.select(test_idx)

    logger.info(
        f"Split sizes — train: {len(hf_train)}, "
        f"val: {len(hf_val)}, "
        f"test: {len(hf_test)}"
    )

    train_ds = InTheWildDataset(hf_train, "train", cfg, augment=True,  cache_dir=cache)
    val_ds   = InTheWildDataset(hf_val,   "val",   cfg, augment=False, cache_dir=cache)
    test_ds  = InTheWildDataset(hf_test,  "test",  cfg, augment=False, cache_dir=cache)

    return {"train": train_ds, "val": val_ds, "test": test_ds}


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    datasets: Dict[str, InTheWildDataset],
    cfg: dict,
    use_weighted_sampler: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create DataLoader objects for each split.

    Args:
        datasets:             Output of build_datasets().
        cfg:                  Full config dict.
        use_weighted_sampler: If True, use WeightedRandomSampler for training
                              to address class imbalance.

    Returns:
        Dict with 'train', 'val', 'test' DataLoaders.
    """
    bs          = cfg["training"]["batch_size"]
    num_workers = cfg["dataset"]["num_workers"]

    loaders = {}

    for split, ds in datasets.items():
        is_train = split == "train"

        if is_train and use_weighted_sampler:
            sample_weights = ds.get_sample_weights()
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            loaders[split] = DataLoader(
                ds,
                batch_size=bs,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=(num_workers > 0),
            )
        else:
            loaders[split] = DataLoader(
                ds,
                batch_size=bs,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=(num_workers > 0),
            )

    return loaders
