"""Predict REAL or FAKE for one audio file or a folder using the deployed best model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import torch

from audio_forgery.config import load_config
from audio_forgery.features import build_resnet_tensor, cached_svm_feature

ROOT = Path(__file__).resolve().parent
RESNET_DIR = ROOT / "model2_resnet"
if str(RESNET_DIR) not in sys.path:
    sys.path.insert(0, str(RESNET_DIR))

from model import build_model  # noqa: E402
from utils import get_device  # noqa: E402

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}


def _load_metadata(best_dir: Path) -> dict[str, Any]:
    metadata_path = best_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("No deployed model found. Run: python train_pipeline.py")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


class Predictor:
    """Inference wrapper for either deployed ResNet++ or SVM."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.best_dir = Path(cfg["paths"]["best_model_dir"])
        self.metadata = _load_metadata(self.best_dir)
        self.best_model = self.metadata["best_model"]
        self.metrics = self.metadata.get("metrics", {})
        self.device = get_device()
        if self.best_model == "resnet":
            checkpoint = torch.load(self.best_dir / "best_resnet.pth", map_location=self.device)
            model_cfg = checkpoint.get("cfg", cfg)
            self.model = build_model(model_cfg, self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
        else:
            self.model = joblib.load(self.best_dir / "best_svm.joblib")

    @torch.no_grad()
    def predict_file(self, path: str | Path) -> dict[str, Any]:
        """Predict one audio file."""
        p = Path(path)
        if self.best_model == "resnet":
            tensor = build_resnet_tensor(p, self.cfg).unsqueeze(0).to(self.device)
            logits = self.model(tensor)
            prob_fake = float(torch.softmax(logits, dim=1)[0, 1].item())
        else:
            features = cached_svm_feature(p, self.cfg).reshape(1, -1)
            prob_fake = float(self.model.predict_proba(features)[0, 1])
        label = 1 if prob_fake >= 0.5 else 0
        confidence = prob_fake if label == 1 else 1.0 - prob_fake
        return {
            "filename": str(p),
            "prediction": "FAKE" if label == 1 else "REAL",
            "confidence": round(confidence * 100.0, 2),
        }


def _audio_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)


def _print_metrics(metrics: dict[str, Any]) -> None:
    print("------------------")
    for key in ("accuracy", "precision", "recall", "f1_score"):
        print(f"{key.replace('_', ' ').title()}: {float(metrics.get(key, 0.0)):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audio deepfake inference")
    parser.add_argument("path", help="Audio file or folder")
    parser.add_argument("--config", default="model2_resnet/configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    predictor = Predictor(cfg)
    target = Path(args.path)
    files = _audio_files(target)
    if not files:
        raise FileNotFoundError(f"No supported audio files found: {target}")
    results = [predictor.predict_file(path) for path in files]

    if target.is_dir():
        out_path = Path(cfg["paths"]["output_dir"]) / "predictions.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["filename", "prediction", "confidence"])
            writer.writeheader()
            writer.writerows(results)
        for result in results:
            print(f"{result['filename']}: {result['prediction']} ({result['confidence']:.2f}%)")
        print(f"Saved: {out_path}")
    else:
        result = results[0]
        print("---")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}%")
    _print_metrics(predictor.metrics)


if __name__ == "__main__":
    main()
