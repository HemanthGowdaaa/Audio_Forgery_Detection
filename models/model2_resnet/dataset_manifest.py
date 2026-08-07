"""
dataset_manifest.py  (M2-Optimized)
=====================================
Memory-efficient PyTorch Dataset for the In-The-Wild audio deepfake dataset.

WHY THIS REPLACES dataset.py / HuggingFace loading:
  The HuggingFace `datasets` library loads all metadata (and sometimes audio
  arrays) into an Arrow table in RAM.  For 31,000+ samples, this alone can
  consume 1–2 GB before a single training step begins.

  This module replaces that approach with a plain CSV manifest — the WAV files
  are already on disk (`outputs/train_manifest.csv` etc.).  Each row contains:
    path, label, label_name, speaker

OPTIMIZATIONS:
  1. **Lazy loading**: audio files are read from disk ONLY in `__getitem__`.
     No audio is pre-loaded into RAM at dataset init time.
  2. **Module-level `MelSpectrogramExtractor`**: the torchaudio transform
     objects (MelSpectrogram, AmplitudeToDB) are created ONCE at init and
     reused, avoiding per-sample object creation overhead.
  3. **Disk cache**: processed tensors are saved to `./cache/spectrograms/`
     as `.pt` files.  The cache key encodes all preprocessing params so it
     auto-invalidates when you change the config.
  4. **`pin_memory=False`**: only beneficial for CUDA; wastes RAM on MPS/CPU.
  5. **WeightedRandomSampler**: still supported for class-imbalance handling.
  6. **`torch.load(mmap=True)`** (PyTorch ≥ 2.1): zero-copy cache reads.
"""

import csv
import gc
import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import torchvision.transforms as VT
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ── Prefer soundfile over TorchCodec (torchaudio 2.11 default) for WAV files
# soundfile is faster, more compatible, and doesn't require torchcodec.
try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ImportError:
    _HAS_SOUNDFILE = False

logger = logging.getLogger("resnet_forgery.dataset")


# ---------------------------------------------------------------------------
# Audio loading helper (soundfile-first for Mac compatibility)
# ---------------------------------------------------------------------------

def _load_audio_file(path: str) -> Tuple[torch.Tensor, int]:
    """
    Load a WAV/MP3 file and return (waveform [C, T], sample_rate).

    Prefers soundfile (fast, no TorchCodec dependency).
    Falls back to torchaudio if soundfile fails or is not installed.

    Returns:
        waveform: float32 Tensor [1, T] (mono, or multi-channel → mono on return)
        sr:       Sample rate (int)
    """
    if _HAS_SOUNDFILE and path.endswith(('.wav', '.flac', '.ogg')):
        try:
            data, sr = sf.read(path, dtype='float32', always_2d=True)  # [T, C]
            waveform = torch.from_numpy(data.T)  # [C, T]
            return waveform, sr
        except Exception:
            pass  # fall through to torchaudio
    # torchaudio fallback
    return torchaudio.load(path)


# ---------------------------------------------------------------------------
# Memory-efficient Mel Spectrogram Extractor
# ---------------------------------------------------------------------------

class FastMelExtractor:
    """
    Converts a waveform numpy array to a [3, H, W] ImageNet-normalized tensor.

    Unlike the original MelSpectrogramExtractor this class:
      - Is instantiated ONCE per dataset (not per sample)
      - Uses F.interpolate (all-tensor, no PIL round-trip)
      - Operates entirely on CPU tensors (no PIL/numpy conversion)

    Args:
        sample_rate: Target sample rate (Hz).
        n_mels:      Number of mel filterbanks.
        n_fft:       FFT window size.
        hop_length:  FFT hop length.
        target_h:    Output spectrogram height (pixels).
        target_w:    Output spectrogram width (pixels).
        mean:        ImageNet normalization mean (tuple of 3).
        std:         ImageNet normalization std (tuple of 3).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        target_h: int = 128,
        target_w: int = 128,
        mean: Tuple = (0.485, 0.456, 0.406),
        std:  Tuple = (0.229, 0.224, 0.225),
    ):
        self.target_h = target_h
        self.target_w = target_w

        # Build torchaudio transforms (created ONCE — not per sample)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
            normalized=False,
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80.0)

        # Pre-compute normalization tensors for fast application
        _mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        _std  = torch.tensor(std,  dtype=torch.float32).view(3, 1, 1)
        self.register_mean = _mean
        self.register_std  = _std

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [1, T] float32 mono waveform tensor.

        Returns:
            [3, target_h, target_w] ImageNet-normalized float32 tensor.
        """
        # Mel spectrogram + dB scale
        mel = self.mel_transform(waveform)        # [1, n_mels, T_frames]
        mel = self.amplitude_to_db(mel)           # still [1, n_mels, T_frames]

        # Min-max normalize to [0, 1]
        mel_min = mel.min()
        mel_max = mel.max()
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-9)

        # Resize using F.interpolate (all-tensor: no PIL, no numpy conversion)
        # [1, 1, n_mels, T_frames] → [1, 1, H, W] → [1, H, W]
        mel = F.interpolate(
            mel.unsqueeze(0),                     # add batch dim: [1, 1, n_mels, T]
            size=(self.target_h, self.target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)                              # [1, H, W]

        # Replicate to 3 channels (ResNet expects RGB)
        mel = mel.expand(3, -1, -1).contiguous() # [3, H, W]  (expand = zero-copy view)

        # ImageNet normalization
        mel = (mel - self.register_mean) / self.register_std

        return mel.float()


# ---------------------------------------------------------------------------
# Pre-emphasis filter (vectorized)
# ---------------------------------------------------------------------------

def _pre_emphasis(waveform: torch.Tensor, coeff: float = 0.97) -> torch.Tensor:
    """First-order high-pass pre-emphasis: y[t] = x[t] - coeff * x[t-1]."""
    return torch.cat(
        [waveform[:, :1], waveform[:, 1:] - coeff * waveform[:, :-1]], dim=-1
    )


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _make_cache_key(row_path: str, cfg: dict) -> str:
    """
    Generate a deterministic cache filename.  The key encodes the audio file
    path AND all preprocessing hyper-parameters so changing the config
    automatically invalidates old cached tensors.
    """
    pre = cfg["preprocessing"]
    ds  = cfg["dataset"]
    key_str = (
        f"{row_path}_"
        f"sr{ds['sample_rate']}_dur{ds['max_audio_length']}_"
        f"nm{pre['n_mels']}_nfft{pre['n_fft']}_hl{pre['hop_length']}_"
        f"h{pre['target_size'][0]}_w{pre['target_size'][1]}"
    )
    return hashlib.md5(key_str.encode()).hexdigest() + ".pt"


# ---------------------------------------------------------------------------
# Core Dataset
# ---------------------------------------------------------------------------

class ManifestDataset(Dataset):
    """
    Memory-efficient Dataset that reads audio files listed in a CSV manifest.

    Expected CSV format (as in outputs/train_manifest.csv):
        path, label, label_name, speaker

    Features:
      - Lazy audio loading (no audio in RAM at init time)
      - Disk-based spectrogram cache (tensor cache survives restarts)
      - Single shared FastMelExtractor (not re-created per sample)
      - SpecAugment applied on-the-fly from cached tensors (no double storage)
      - Explicit garbage collection after processing heavy samples

    Args:
        manifest_path:  Path to the CSV manifest file.
        cfg:            Full config dict.
        split:          'train', 'val', or 'test' (used in logging only).
        augment:        If True, apply SpecAugment during __getitem__.
        cache_dir:      Directory for caching preprocessed tensors.
                        Set to None to disable caching.
        root_dir:       Root directory to resolve relative paths in the CSV.
                        If None, paths are resolved from the current working dir.
        subset:         If set, randomly sample this many items from the manifest.
    """

    def __init__(
        self,
        manifest_path: str,
        cfg: dict,
        split: str = "train",
        augment: bool = False,
        cache_dir: Optional[str] = None,
        root_dir: Optional[str] = None,
        subset: Optional[int] = None,
    ):
        self.cfg        = cfg
        self.split      = split
        self.augment    = augment
        self.cache_dir  = Path(cache_dir) if cache_dir else None
        self.root_dir   = Path(root_dir) if root_dir else Path.cwd()

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Parse manifest CSV
        # ------------------------------------------------------------------
        self.records: List[dict] = []
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Resolve relative path
                audio_path = Path(row["path"])
                if not audio_path.is_absolute():
                    audio_path = self.root_dir / audio_path
                if not audio_path.exists():
                    logger.warning(f"Audio file not found, skipping: {audio_path}")
                    continue
                self.records.append({
                    "path":  str(audio_path),
                    "label": int(row["label"]),
                })

        logger.info(f"[{split}] Loaded {len(self.records)} valid samples from {manifest_path}")

        # Optionally subsample
        if subset is not None and subset < len(self.records):
            rng = np.random.default_rng(cfg["dataset"].get("seed", 42))
            idxs = rng.choice(len(self.records), size=subset, replace=False)
            self.records = [self.records[i] for i in idxs]
            logger.info(f"[{split}] Subsampled to {len(self.records)} samples (subset={subset})")

        # Label summary
        labels = [r["label"] for r in self.records]
        n_real = labels.count(0)
        n_fake = labels.count(1)
        logger.info(f"[{split}] real={n_real}, fake={n_fake}")

        # ------------------------------------------------------------------
        # Build shared MelExtractor (ONCE — not per __getitem__ call)
        # ------------------------------------------------------------------
        pre = cfg["preprocessing"]
        ds  = cfg["dataset"]
        self.extractor = FastMelExtractor(
            sample_rate=ds["sample_rate"],
            n_mels=pre["n_mels"],
            n_fft=pre["n_fft"],
            hop_length=pre["hop_length"],
            target_h=pre["target_size"][0],
            target_w=pre["target_size"][1],
            mean=tuple(pre["imagenet_mean"]),
            std=tuple(pre["imagenet_std"]),
        )

        # SpecAugment config (used in __getitem__)
        aug = pre.get("spec_augment", {})
        self._aug_time_w = aug.get("time_mask_width", 15)
        self._aug_freq_w = aug.get("freq_mask_width", 8)
        self._aug_n_time = aug.get("num_time_masks", 1)
        self._aug_n_freq = aug.get("num_freq_masks", 1)

        # Audio config
        self._sample_rate = ds["sample_rate"]
        self._max_samples = int(ds["max_audio_length"] * ds["sample_rate"])

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            (spectrogram_tensor, label)
            spectrogram_tensor: float32 Tensor [3, H, W]
            label: int  0=real/genuine, 1=fake/spoof
        """
        record = self.records[idx]
        label  = record["label"]
        path   = record["path"]

        # ── Try cache first (base tensor, no augmentation)
        cache_path = None
        if self.cache_dir:
            cache_path = self.cache_dir / _make_cache_key(path, self.cfg)
            if cache_path.exists():
                try:
                    # mmap=True (PyTorch ≥ 2.1): read without copying into RAM
                    spec = torch.load(cache_path, weights_only=True, mmap=True)
                    spec = spec.clone()   # detach from mmap for mutation
                except Exception:
                    # Fallback for older PyTorch or corrupted cache
                    spec = torch.load(cache_path, weights_only=True)
                if self.augment:
                    spec = self._apply_spec_augment(spec)
                return spec, label

        # ── Load raw audio from disk (lazy)
        try:
            waveform, sr = _load_audio_file(path)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e} — returning zeros")
            h = self.cfg["preprocessing"]["target_size"][0]
            w = self.cfg["preprocessing"]["target_size"][1]
            return torch.zeros(3, h, w, dtype=torch.float32), label

        # ── Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)   # [1, T]

        # ── Resample if needed
        if sr != self._sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self._sample_rate)
            waveform  = resampler(waveform)

        # ── Pad / trim to fixed length
        T_curr = waveform.shape[-1]
        if T_curr >= self._max_samples:
            waveform = waveform[:, :self._max_samples]
        else:
            waveform = F.pad(waveform, (0, self._max_samples - T_curr))

        # ── Zero-mean / unit-variance normalization
        mean = waveform.mean()
        std  = waveform.std()
        waveform = (waveform - mean) / (std + 1e-9)

        # ── Pre-emphasis
        waveform = _pre_emphasis(waveform)

        # ── Mel spectrogram + dB + resize + normalize
        spec = self.extractor(waveform)   # [3, H, W]

        # Free waveform immediately (helps on tight RAM)
        del waveform
        gc.collect()

        # ── Cache the base tensor (no augment)
        if cache_path is not None:
            try:
                torch.save(spec, cache_path)
            except Exception as e:
                logger.warning(f"Cache write failed for {path}: {e}")

        # ── Apply SpecAugment on-the-fly
        if self.augment:
            spec = self._apply_spec_augment(spec)

        return spec, label

    # ------------------------------------------------------------------

    def _apply_spec_augment(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply frequency and time masking (SpecAugment) to a tensor."""
        spec = spec.clone()
        _, H, W = spec.shape
        fill   = spec.mean().item()

        # Frequency masks (horizontal bands)
        for _ in range(self._aug_n_freq):
            f  = int(np.random.uniform(0, max(1, self._aug_freq_w)))
            f0 = int(np.random.uniform(0, max(1, H - f)))
            spec[:, f0:f0 + f, :] = fill

        # Time masks (vertical bands)
        for _ in range(self._aug_n_time):
            t  = int(np.random.uniform(0, max(1, self._aug_time_w)))
            t0 = int(np.random.uniform(0, max(1, W - t)))
            spec[:, :, t0:t0 + t] = fill

        return spec

    # ------------------------------------------------------------------

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for CrossEntropyLoss(weight=…).

        Returns:
            Tensor [num_classes] where weight_c = N / (num_classes * n_c)
        """
        labels  = [r["label"] for r in self.records]
        counts  = np.bincount(labels, minlength=2).astype(float)
        weights = len(labels) / (2.0 * (counts + 1e-6))
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler."""
        class_w = self.get_class_weights()
        return torch.tensor(
            [class_w[r["label"]].item() for r in self.records],
            dtype=torch.float32,
        )


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def build_manifest_datasets(cfg: dict) -> Dict[str, ManifestDataset]:
    """
    Build train / val / test ManifestDataset objects from CSV manifests.

    The manifest paths are read from cfg['dataset']:
        train_manifest, val_manifest, test_manifest

    Args:
        cfg: Full config dict (from configs/config.yaml).

    Returns:
        Dict with keys 'train', 'val', 'test'.
    """
    ds_cfg = cfg["dataset"]
    cache  = cfg["paths"]["cache_dir"]

    # Resolve manifest paths relative to this file's directory
    here = Path(__file__).parent

    def _resolve(key: str) -> str:
        p = Path(ds_cfg[key])
        if not p.is_absolute():
            p = (here / p).resolve()
        return str(p)

    # Root dir for resolving audio file paths inside the CSV
    root_dir = here.parent   # project root (DL_project/)

    train_ds = ManifestDataset(
        manifest_path=_resolve("train_manifest"),
        cfg=cfg,
        split="train",
        augment=True,
        cache_dir=cache,
        root_dir=str(root_dir),
        subset=ds_cfg.get("train_subset"),
    )
    val_ds = ManifestDataset(
        manifest_path=_resolve("val_manifest"),
        cfg=cfg,
        split="val",
        augment=False,
        cache_dir=cache,
        root_dir=str(root_dir),
        subset=ds_cfg.get("validation_subset"),
    )
    test_ds = ManifestDataset(
        manifest_path=_resolve("test_manifest"),
        cfg=cfg,
        split="test",
        augment=False,
        cache_dir=cache,
        root_dir=str(root_dir),
        subset=ds_cfg.get("test_subset"),
    )

    return {"train": train_ds, "val": val_ds, "test": test_ds}


# ---------------------------------------------------------------------------
# DataLoader factory (memory-aware)
# ---------------------------------------------------------------------------

def build_manifest_dataloaders(
    datasets: Dict[str, ManifestDataset],
    cfg: dict,
    use_weighted_sampler: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create memory-efficient DataLoaders for each split.

    Key memory decisions:
      - `pin_memory=False`:         Only helps CUDA; wastes RAM on MPS/CPU
      - `num_workers=0`:            Avoid subprocess memory copies on Mac
      - `persistent_workers=False`: Only valid when num_workers > 0
      - `drop_last=True` (train):   Avoids tiny last batch causing BN issues
      - val/test use larger batch if possible (no gradients needed)

    Args:
        datasets:             Output of build_manifest_datasets().
        cfg:                  Full config dict.
        use_weighted_sampler: Use WeightedRandomSampler for training split.

    Returns:
        Dict with 'train', 'val', 'test' DataLoaders.
    """
    bs          = cfg["training"]["batch_size"]
    num_workers = cfg["dataset"].get("num_workers", 0)

    # Validation can use a smaller batch to keep peak RAM low
    val_bs = max(1, bs // 2)

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
                pin_memory=False,          # no CUDA → no benefit
                drop_last=True,
                persistent_workers=False,  # only valid for num_workers > 0
            )
        else:
            loaders[split] = DataLoader(
                ds,
                batch_size=val_bs,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False,
                drop_last=False,
                persistent_workers=False,
            )

    return loaders
