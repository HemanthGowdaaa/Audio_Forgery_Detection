"""Local dataset discovery, validation, and deterministic splitting."""

from __future__ import annotations

import csv
import hashlib
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from audio_forgery.features import build_resnet_tensor

LOGGER = logging.getLogger(__name__)
LABEL_MAP = {
    "bona-fide": 0,
    "bonafide": 0,
    "genuine": 0,
    "real": 0,
    "0": 0,
    "spoof": 1,
    "fake": 1,
    "synthetic": 1,
    "forged": 1,
    "1": 1,
}


@dataclass(frozen=True)
class AudioSample:
    """One validated local audio sample."""

    path: str
    label: int
    label_name: str
    speaker: str = ""


def _read_metadata(root: Path) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    for name in ("meta.csv", "metadata.csv", "labels.csv"):
        csv_path = root / name
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                file_key = row.get("file") or row.get("filename") or row.get("path")
                if file_key:
                    metadata[Path(file_key).name] = row
        LOGGER.info("Loaded %s metadata rows from %s", len(metadata), csv_path)
        return metadata
    return metadata


def _infer_label(path: Path, row: dict[str, str] | None) -> tuple[int, str]:
    if row:
        for column in ("label", "class", "target", "is_fake", "is_spoof"):
            raw = row.get(column)
            if raw is not None and str(raw).strip().lower() in LABEL_MAP:
                label = LABEL_MAP[str(raw).strip().lower()]
                return label, "FAKE" if label == 1 else "REAL"
    parts = [part.lower() for part in path.parts]
    if any(part in {"fake", "spoof", "synthetic", "forged"} for part in parts):
        return 1, "FAKE"
    if any(part in {"real", "genuine", "bona-fide", "bonafide"} for part in parts):
        return 0, "REAL"
    raise ValueError(f"No label found for {path}")


def _is_valid_audio(path: Path) -> bool:
    try:
        info = sf.info(str(path))
        return info.frames > 0 and info.samplerate > 0
    except Exception as exc:  # pragma: no cover - depends on local codecs
        LOGGER.warning("Skipping unreadable audio %s: %s", path, exc)
        return False


def discover_samples(root: str | Path, extensions: Iterable[str]) -> list[AudioSample]:
    """Recursively discover valid audio files and attach labels from metadata."""
    root = Path(root)
    metadata = _read_metadata(root)
    exts = {ext.lower() for ext in extensions}
    skipped_path = Path("outputs") / "skipped_files.log"
    skipped_path.parent.mkdir(parents=True, exist_ok=True)
    samples: list[AudioSample] = []
    skipped: list[str] = []

    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in exts):
        try:
            label, label_name = _infer_label(path, metadata.get(path.name))
            if not _is_valid_audio(path):
                skipped.append(f"{path}\tunreadable")
                continue
            speaker = (metadata.get(path.name) or {}).get("speaker", "")
            samples.append(AudioSample(str(path), label, label_name, speaker))
        except Exception as exc:
            LOGGER.warning("Skipping %s: %s", path, exc)
            skipped.append(f"{path}\t{exc}")

    skipped_path.write_text("\n".join(skipped), encoding="utf-8")
    if not samples:
        raise RuntimeError(f"No valid labelled audio files found under {root}")
    LOGGER.info("Discovered %s valid samples; skipped %s", len(samples), len(skipped))
    return samples


def deterministic_split(
    samples: list[AudioSample],
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, list[AudioSample]]:
    """Create 70/15/15 deterministic stratified splits."""
    labels = np.array([sample.label for sample in samples])
    stratify = labels if len(np.unique(labels)) == 2 else None
    train, temp = train_test_split(
        samples,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    temp_labels = np.array([sample.label for sample in temp])
    val_fraction = val_ratio / (1.0 - train_ratio)
    stratify_temp = temp_labels if len(np.unique(temp_labels)) == 2 else None
    val, test = train_test_split(
        temp,
        train_size=val_fraction,
        random_state=seed,
        shuffle=True,
        stratify=stratify_temp,
    )
    return {"train": train, "val": val, "test": test}


def save_split_manifest(splits: dict[str, list[AudioSample]], output_dir: str | Path) -> None:
    """Persist exact split membership for reproducible SVM/ResNet evaluation."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, samples in splits.items():
        with (out / f"{split}_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "label", "label_name", "speaker"])
            writer.writeheader()
            writer.writerows(asdict(sample) for sample in samples)


def load_or_create_splits(cfg: dict) -> dict[str, list[AudioSample]]:
    """Discover local data and create deterministic splits."""
    ds = cfg["dataset"]
    samples = discover_samples(ds["root"], ds["extensions"])
    splits = deterministic_split(
        samples,
        seed=ds.get("seed", 42),
        train_ratio=ds.get("train_ratio", 0.70),
        val_ratio=ds.get("val_ratio", 0.15),
    )
    save_split_manifest(splits, cfg["paths"]["output_dir"])
    return splits


class ResNetAudioDataset(Dataset[tuple[torch.Tensor, int]]):
    """PyTorch dataset backed by local files with feature caching."""

    def __init__(self, samples: list[AudioSample], cfg: dict, split: str, augment: bool = False):
        self.samples = samples
        self.cfg = cfg
        self.split = split
        self.augment = augment
        self.cache_dir = Path(cfg["paths"]["cache_dir"]) / "resnet"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.samples)

    def _cache_path(self, sample: AudioSample) -> Path:
        pp = self.cfg["preprocessing"]
        ds = self.cfg["dataset"]
        key = f"{sample.path}|{Path(sample.path).stat().st_mtime_ns}|{ds['sample_rate']}|{ds['duration']}|{pp['n_mels']}"
        return self.cache_dir / f"{hashlib.sha1(key.encode()).hexdigest()}.pt"

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        cache_path = self._cache_path(sample)
        if cache_path.exists():
            tensor = torch.load(cache_path, map_location="cpu", weights_only=True)
        else:
            tensor = build_resnet_tensor(sample.path, self.cfg)
            torch.save(tensor, cache_path)
        return tensor, sample.label

    def class_weights(self) -> torch.Tensor:
        labels = np.array([sample.label for sample in self.samples])
        counts = np.bincount(labels, minlength=2).astype(np.float32)
        weights = len(labels) / (2.0 * np.maximum(counts, 1.0))
        return torch.tensor(weights, dtype=torch.float32)

    def get_class_weights(self) -> torch.Tensor:
        """Compatibility alias for the original ResNet training script."""
        return self.class_weights()

    def sample_weights(self) -> torch.Tensor:
        weights = self.class_weights()
        return torch.tensor([weights[sample.label].item() for sample in self.samples])

    def get_sample_weights(self) -> torch.Tensor:
        """Compatibility alias for the original ResNet training script."""
        return self.sample_weights()


def build_resnet_dataloaders(splits: dict[str, list[AudioSample]], cfg: dict) -> dict[str, DataLoader]:
    """Build efficient DataLoaders for Apple MPS or CPU."""
    batch_size = cfg["training"]["batch_size"]
    workers = int(cfg["dataset"].get("num_workers", 0))
    loaders: dict[str, DataLoader] = {}
    for split, samples in splits.items():
        dataset = ResNetAudioDataset(samples, cfg, split, augment=(split == "train"))
        sampler = None
        shuffle = split == "train"
        if split == "train":
            sampler = WeightedRandomSampler(dataset.sample_weights(), len(dataset), replacement=True)
            shuffle = False
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
            drop_last=split == "train" and len(dataset) >= batch_size,
        )
    return loaders
