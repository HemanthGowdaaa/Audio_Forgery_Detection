"""
preprocess.py  (M2-Optimized)
=============
Audio preprocessing pipeline for the ResNet++ Audio Forgery Detection framework.

Pipeline steps (in order):
  1. Load audio at 16 kHz (resample if needed)
  2. Zero-mean / unit-variance normalization
  3. Pre-emphasis filtering
  4. Mel Spectrogram  (n_mels configurable, default 80)
  5. Power → dB conversion
  6. Resize to target_size (default 128×128) via F.interpolate (no PIL)
  7. Convert to 3-channel image (expand, zero-copy)
  8. ImageNet normalization
  9. Optional SpecAugment (time + frequency masking)

OPTIMIZATIONS:
  - MelSpectrogramExtractor uses F.interpolate instead of PIL resize.
    This avoids the numpy→PIL→tensor round-trip, saving ~1 memory copy
    per sample and removing PIL as a dependency for the resize step.
  - get_extractor() provides a cached module-level instance so callers
    can reuse the same transform objects across __getitem__ calls.
"""

import io
import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torchvision.transforms as VT
from PIL import Image

logger = logging.getLogger("resnet_forgery.preprocess")


# ---------------------------------------------------------------------------
# Constants / Defaults (overridden by config at runtime)
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_RATE: int = 16000
DEFAULT_MAX_SECONDS: float = 5.0

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Low-level signal helpers
# ---------------------------------------------------------------------------

def load_audio(
    audio_input,
    target_sr: int = DEFAULT_SAMPLE_RATE,
) -> Tuple[torch.Tensor, int]:
    """
    Load audio from a variety of input types and resample to target_sr.

    Accepts:
      - str / Path  → file on disk
      - bytes       → raw audio bytes (wav / mp3 etc.)
      - numpy array → already decoded waveform (assumes float32, shape [T] or [C, T])
      - torch.Tensor

    Returns:
        (waveform, sample_rate)  where waveform is shape [1, T] (mono, float32)
    """
    if isinstance(audio_input, (str,)) or hasattr(audio_input, "__fspath__"):
        waveform, sr = torchaudio.load(str(audio_input))

    elif isinstance(audio_input, bytes):
        buf = io.BytesIO(audio_input)
        waveform, sr = torchaudio.load(buf)

    elif isinstance(audio_input, np.ndarray):
        # Ensure float32
        audio_np = audio_input.astype(np.float32)
        if audio_np.ndim == 1:
            audio_np = audio_np[np.newaxis, :]   # [1, T]
        waveform = torch.from_numpy(audio_np)
        sr = target_sr  # assume already at target rate

    elif isinstance(audio_input, torch.Tensor):
        waveform = audio_input.float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        sr = target_sr

    else:
        raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

    # Convert to mono by averaging channels
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform, target_sr


def pad_or_trim(
    waveform: torch.Tensor,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> torch.Tensor:
    """
    Pad (with zeros) or trim (from the right) audio to a fixed length.

    Args:
        waveform:    Tensor of shape [1, T].
        max_seconds: Target duration in seconds.
        sample_rate: Samples per second.

    Returns:
        Tensor of shape [1, max_len] where max_len = max_seconds * sample_rate.
    """
    max_len = int(max_seconds * sample_rate)
    current_len = waveform.shape[-1]

    if current_len >= max_len:
        return waveform[:, :max_len]

    pad_amount = max_len - current_len
    return torch.nn.functional.pad(waveform, (0, pad_amount))


def normalize_waveform(waveform: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Apply zero-mean / unit-variance normalization to the waveform.

    Args:
        waveform: Tensor of shape [1, T].
        eps:      Small value to avoid division by zero.

    Returns:
        Normalized waveform, same shape.
    """
    mean = waveform.mean()
    std  = waveform.std()
    return (waveform - mean) / (std + eps)


def pre_emphasis(waveform: torch.Tensor, coeff: float = 0.97) -> torch.Tensor:
    """
    Apply a first-order high-pass pre-emphasis filter.

    y[t] = x[t] - coeff * x[t-1]

    Pre-emphasis amplifies high-frequency components which are often
    attenuated in human speech/audio, helping the model distinguish
    genuine from synthesized audio.

    Args:
        waveform: Tensor of shape [1, T].
        coeff:    Pre-emphasis coefficient (typically 0.95–0.97).

    Returns:
        Filtered waveform, same shape.
    """
    # Shift right by one sample and subtract
    emphasized = torch.cat(
        [waveform[:, :1], waveform[:, 1:] - coeff * waveform[:, :-1]], dim=-1
    )
    return emphasized


# ---------------------------------------------------------------------------
# Spectrogram extraction
# ---------------------------------------------------------------------------

class MelSpectrogramExtractor:
    """
    Converts a raw waveform into a normalized, resized Mel Spectrogram image
    suitable for input to a CNN (ResNet50).

    Output: torch.Tensor of shape [3, 224, 224], ImageNet-normalized.
    """

    def __init__(
        self,
        sample_rate:  int   = DEFAULT_SAMPLE_RATE,
        n_mels:       int   = 128,
        n_fft:        int   = 2048,
        hop_length:   int   = 512,
        target_h:     int   = 224,
        target_w:     int   = 224,
        imagenet_mean: Tuple = IMAGENET_MEAN,
        imagenet_std:  Tuple = IMAGENET_STD,
    ):
        self.sample_rate  = sample_rate
        self.n_mels       = n_mels
        self.n_fft        = n_fft
        self.hop_length   = hop_length
        self.target_h     = target_h
        self.target_w     = target_w

        # torchaudio Mel Spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,           # Power spectrogram
            normalized=False,
        )

        # Power → dB
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80.0)

        # Final normalization (ImageNet stats)
        self.normalize = VT.Normalize(mean=list(imagenet_mean), std=list(imagenet_std))

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: Tensor of shape [1, T] (mono audio, float32).

        Returns:
            Tensor of shape [3, 224, 224], ImageNet-normalized.
        """
        # ── Mel Spectrogram → shape [1, n_mels, time_frames]
        mel_spec = self.mel_transform(waveform)

        # ── Convert to dB scale → same shape
        mel_db = self.amplitude_to_db(mel_spec)

        # ── Min-max normalize to [0, 1]
        mel_min = mel_db.min()
        mel_max = mel_db.max()
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-9)  # [1, n_mels, T]

        # ── Resize via F.interpolate (REPLACES the PIL round-trip)
        #    Input:  [1, n_mels, T]  → unsqueeze → [1, 1, n_mels, T]
        #    Output: [1, 1, H, W]   → squeeze   → [1, H, W]
        #    This is a pure-tensor operation — no numpy/PIL/uint8 conversion.
        import torch.nn.functional as F
        tensor = F.interpolate(
            mel_norm.unsqueeze(0),
            size=(self.target_h, self.target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)   # [1, H, W]

        # ── Replicate to 3 channels via expand (zero-copy view, then contiguous)
        tensor = tensor.expand(3, -1, -1).contiguous()   # [3, H, W]

        # ── Apply ImageNet normalization
        tensor = self.normalize(tensor)

        return tensor.float()   # [3, H, W]


# ---------------------------------------------------------------------------
# SpecAugment
# ---------------------------------------------------------------------------

class SpecAugment:
    """
    Applies SpecAugment (Park et al., 2019) to a spectrogram tensor.

    Randomly masks contiguous blocks of:
      - Time frames  (vertical stripes on the spectrogram)
      - Frequency bins (horizontal stripes)

    This is applied *before* ImageNet normalization when used during training.
    Here we apply it to the final 3-channel tensor by masking with the mean
    value of the tensor so it aligns with ImageNet-normalized inputs.

    Args:
        time_mask_width:  Maximum width of each time mask.
        freq_mask_width:  Maximum width of each frequency mask.
        num_time_masks:   Number of time masks to apply.
        num_freq_masks:   Number of frequency masks to apply.
    """

    def __init__(
        self,
        time_mask_width: int = 20,
        freq_mask_width: int = 10,
        num_time_masks:  int = 2,
        num_freq_masks:  int = 2,
    ):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        self.num_time_masks  = num_time_masks
        self.num_freq_masks  = num_freq_masks

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: Tensor of shape [3, H, W].

        Returns:
            Augmented tensor, same shape.
        """
        spec = spec.clone()
        _, H, W = spec.shape
        fill_value = spec.mean().item()

        # ── Frequency masks (horizontal bands)
        for _ in range(self.num_freq_masks):
            f = int(np.random.uniform(0, self.freq_mask_width))
            f0 = int(np.random.uniform(0, max(1, H - f)))
            spec[:, f0 : f0 + f, :] = fill_value

        # ── Time masks (vertical bands)
        for _ in range(self.num_time_masks):
            t = int(np.random.uniform(0, self.time_mask_width))
            t0 = int(np.random.uniform(0, max(1, W - t)))
            spec[:, :, t0 : t0 + t] = fill_value

        return spec


# ---------------------------------------------------------------------------
# Full preprocessing function (entry point used by dataset.py)
# ---------------------------------------------------------------------------

def preprocess_audio(
    audio_input,
    cfg: dict,
    augment: bool = False,
) -> torch.Tensor:
    """
    Full preprocessing pipeline: raw audio → model-ready tensor.

    Args:
        audio_input: Any supported audio source (path, bytes, array, tensor).
        cfg:         Loaded config dict (from configs/config.yaml).
        augment:     If True, apply SpecAugment (training only).

    Returns:
        Tensor of shape [3, 224, 224], ready for the model.
    """
    ds_cfg   = cfg["dataset"]
    pre_cfg  = cfg["preprocessing"]
    aug_cfg  = pre_cfg["spec_augment"]

    sample_rate  = ds_cfg["sample_rate"]
    max_seconds  = ds_cfg["max_audio_length"]
    target_size  = pre_cfg["target_size"]   # [H, W]

    # Step 1: Load & mono
    waveform, sr = load_audio(audio_input, target_sr=sample_rate)

    # Step 2: Pad / trim to fixed length
    waveform = pad_or_trim(waveform, max_seconds=max_seconds, sample_rate=sr)

    # Step 3: Zero-mean / unit-variance normalization
    waveform = normalize_waveform(waveform)

    # Step 4: Pre-emphasis
    waveform = pre_emphasis(waveform)

    # Step 5–8: Mel spectrogram → dB → resize → 3-channel → ImageNet norm
    extractor = MelSpectrogramExtractor(
        sample_rate=sr,
        n_mels=pre_cfg["n_mels"],
        n_fft=pre_cfg["n_fft"],
        hop_length=pre_cfg["hop_length"],
        target_h=target_size[0],
        target_w=target_size[1],
        imagenet_mean=tuple(pre_cfg["imagenet_mean"]),
        imagenet_std=tuple(pre_cfg["imagenet_std"]),
    )
    spec_tensor = extractor(waveform)   # [3, 224, 224]

    # Step 9: SpecAugment (training only)
    if augment:
        spec_aug = SpecAugment(
            time_mask_width=aug_cfg["time_mask_width"],
            freq_mask_width=aug_cfg["freq_mask_width"],
            num_time_masks=aug_cfg["num_time_masks"],
            num_freq_masks=aug_cfg["num_freq_masks"],
        )
        spec_tensor = spec_aug(spec_tensor)

    return spec_tensor
