"""Audio loading and feature extraction for ResNet++ and SVM models."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torchvision.transforms import Normalize

LOGGER = logging.getLogger(__name__)


def load_audio(path: str | Path, sample_rate: int = 16000, duration: float = 5.0) -> np.ndarray:
    """Load mono audio, resample to target rate, center-crop or zero-pad."""
    y, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    target_len = int(sample_rate * duration)
    if len(y) > target_len:
        start = max(0, (len(y) - target_len) // 2)
        y = y[start:start + target_len]
    elif len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    y = y.astype(np.float32)
    std = float(np.std(y))
    if std > 1e-8:
        y = (y - float(np.mean(y))) / (std + 1e-8)
    return y


def _cache_path(path: str | Path, feature_type: str, cfg: dict) -> Path:
    p = Path(path)
    ds = cfg["dataset"]
    pp = cfg["preprocessing"]
    # We combine path, mtime, sample rate, duration, and feature_type into the cache key
    mtime = p.stat().st_mtime_ns if p.exists() else 0
    key = f"{p.resolve()}|{mtime}|{ds['sample_rate']}|{ds['duration']}|{pp['n_mels']}|{pp['n_mfcc']}|{feature_type}"
    hasher = hashlib.sha1(key.encode())
    cache_dir = Path(cfg["paths"].get("cache_dir", "cache")) / "features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{hasher.hexdigest()}_{feature_type}.npy"


def get_mel_spectrogram(path: str | Path, cfg: dict) -> np.ndarray:
    """Get or extract cached Mel Spectrogram as .npy."""
    cache_p = _cache_path(path, "mel", cfg)
    if cache_p.exists():
        return np.load(cache_p)
    
    ds = cfg["dataset"]
    pp = cfg["preprocessing"]
    sr = int(ds["sample_rate"])
    y = load_audio(path, sr, float(ds["duration"]))
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=int(pp["n_fft"]),
        hop_length=int(pp["hop_length"]),
        n_mels=int(pp["n_mels"]),
        power=2.0,
    )
    np.save(cache_p, mel)
    return mel


def get_mfcc(path: str | Path, cfg: dict) -> np.ndarray:
    """Get or extract cached MFCC as .npy."""
    cache_p = _cache_path(path, "mfcc", cfg)
    if cache_p.exists():
        return np.load(cache_p)
    
    ds = cfg["dataset"]
    pp = cfg["preprocessing"]
    sr = int(ds["sample_rate"])
    y = load_audio(path, sr, float(ds["duration"]))
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=int(pp.get("n_mfcc", 20)),
        n_fft=int(pp["n_fft"]),
        hop_length=int(pp["hop_length"]),
    )
    np.save(cache_p, mfcc)
    return mfcc


def _normalize_matrix(feature: np.ndarray) -> np.ndarray:
    mean = np.mean(feature)
    std = np.std(feature)
    return ((feature - mean) / (std + 1e-8)).astype(np.float32)


def extract_feature_bundle(path: str | Path, cfg: dict) -> dict[str, np.ndarray]:
    """Generate normalized Mel, log-Mel, MFCC, chroma, contrast, ZCR, and RMS."""
    ds = cfg["dataset"]
    pp = cfg["preprocessing"]
    sr = int(ds["sample_rate"])
    y = load_audio(path, sr, float(ds["duration"]))
    
    # Use cached Mel and MFCC where possible
    mel = get_mel_spectrogram(path, cfg)
    mfcc = get_mfcc(path, cfg)
    
    log_mel = librosa.power_to_db(mel, ref=np.max)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=int(pp["n_fft"]), hop_length=int(pp["hop_length"]))
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=int(pp["n_fft"]), hop_length=int(pp["hop_length"]))
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=int(pp["hop_length"]))
    rms = librosa.feature.rms(y=y, frame_length=int(pp["n_fft"]), hop_length=int(pp["hop_length"]))
    return {
        "mel": _normalize_matrix(mel),
        "log_mel": _normalize_matrix(log_mel),
        "mfcc": _normalize_matrix(mfcc),
        "chroma": _normalize_matrix(chroma),
        "spectral_contrast": _normalize_matrix(contrast),
        "zcr": _normalize_matrix(zcr),
        "rms": _normalize_matrix(rms),
    }


def build_resnet_tensor(path: str | Path, cfg: dict) -> torch.Tensor:
    """Create a 3-channel log-Mel tensor for ResNet50 input using cached Mel Spectrogram."""
    mel = get_mel_spectrogram(path, cfg)
    mel_tensor = torch.tensor(mel).unsqueeze(0)
    
    # Compute log-Mel
    log_mel = T.AmplitudeToDB(stype="power", top_db=80.0)(mel_tensor)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    
    pp = cfg["preprocessing"]
    image = F.interpolate(
        log_mel.unsqueeze(0),
        size=tuple(pp["target_size"]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    image = image.repeat(3, 1, 1)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return Normalize(pp["imagenet_mean"], pp["imagenet_std"])(image.float())


def aggregate_svm_features(bundle: dict[str, np.ndarray]) -> np.ndarray:
    """Aggregate SVM features using mean, std, min, and max per coefficient."""
    vectors: list[np.ndarray] = []
    for name in ("mfcc", "chroma", "spectral_contrast", "rms", "zcr"):
        matrix = bundle[name]
        vectors.extend([
            np.mean(matrix, axis=1),
            np.std(matrix, axis=1),
            np.min(matrix, axis=1),
            np.max(matrix, axis=1),
        ])
    return np.concatenate(vectors).astype(np.float32)


def cached_svm_feature(path: str | Path, cfg: dict) -> np.ndarray:
    """Extract or load cached aggregated SVM features."""
    cache_p = _cache_path(path, "svm", cfg)
    if cache_p.exists():
        return np.load(cache_p)
    features = aggregate_svm_features(extract_feature_bundle(path, cfg))
    np.save(cache_p, features)
    return features

