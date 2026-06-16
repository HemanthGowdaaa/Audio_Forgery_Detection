"""
inference.py
============
Single-file inference script for the ResNet++ Audio Forgery Detection model.

Usage examples:
  # Predict on a single file:
  python inference.py --audio path/to/audio.wav

  # Predict on a directory of audio files:
  python inference.py --audio path/to/folder/ --output_dir outputs/predictions/

  # Use a specific checkpoint:
  python inference.py --audio clip.wav --checkpoint checkpoints/best_model.pth

Output (per file):
  {
    "file": "clip.wav",
    "prediction": "Fake",
    "label_id": 1,
    "confidence": 0.9823,
    "prob_genuine": 0.0177,
    "prob_fake": 0.9823
  }
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
import numpy as np

from preprocess import preprocess_audio
from model import build_model
from utils import (
    get_device,
    load_config,
    load_checkpoint,
    set_seed,
    setup_logger,
    format_metrics,
)

logger = logging.getLogger("resnet_forgery.inference")

# Human-readable class names
CLASS_NAMES = {0: "Genuine", 1: "Fake"}

# Supported audio extensions
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}


# ---------------------------------------------------------------------------
# Core predictor class
# ---------------------------------------------------------------------------

class AudioForgeryPredictor:
    """
    High-level inference wrapper for the ResNet++ Audio Forgery Detection model.

    Encapsulates:
      - Model loading from checkpoint
      - Preprocessing pipeline
      - Batch or single-file prediction
      - Confidence thresholding

    Args:
        config_path:     Path to configs/config.yaml.
        checkpoint_path: Path to model checkpoint (.pth file).
        device:          torch.device. If None, auto-selects best device.
        threshold:       Decision threshold on P(fake). Default 0.5.
    """

    def __init__(
        self,
        config_path:     str = "configs/config.yaml",
        checkpoint_path: Optional[str] = None,
        device:          Optional[torch.device] = None,
        threshold:       float = 0.5,
    ):
        # ── Load config
        self.cfg       = load_config(config_path)
        self.threshold = threshold

        # ── Device
        self.device = device if device is not None else get_device()
        logger.info(f"Inference device: {self.device}")

        # ── Resolve checkpoint path
        if checkpoint_path is None:
            checkpoint_path = str(
                Path(self.cfg["paths"]["checkpoint_dir"]) / "best_model.pth"
            )
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                "Please train the model first with: python train.py"
            )

        # ── Build + load model
        logger.info(f"Loading model from: {checkpoint_path}")
        self.model = build_model(self.cfg, self.device)
        self.model, _, _ = load_checkpoint(
            checkpoint_path, self.model, device=self.device
        )
        self.model.eval()
        logger.info("Model loaded and ready for inference.")

    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_file(self, audio_path: Union[str, Path]) -> Dict:
        """
        Predict whether a single audio file is Genuine or Fake.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Dict with keys:
              file, prediction, label_id, confidence, prob_genuine, prob_fake
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        t_start = time.perf_counter()

        # ── Preprocess: audio → spectrogram tensor [3, 224, 224]
        spec = preprocess_audio(
            audio_input=str(audio_path),
            cfg=self.cfg,
            augment=False,
        )
        spec = spec.unsqueeze(0).to(self.device)   # [1, 3, 224, 224]

        # ── Forward pass
        logits = self.model(spec)                  # [1, 2]
        probs  = F.softmax(logits, dim=1)          # [1, 2]

        prob_genuine = probs[0, 0].item()
        prob_fake    = probs[0, 1].item()
        label_id     = 1 if prob_fake >= self.threshold else 0
        prediction   = CLASS_NAMES[label_id]
        confidence   = prob_fake if label_id == 1 else prob_genuine

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        result = {
            "file":         str(audio_path),
            "prediction":   prediction,
            "label_id":     label_id,
            "confidence":   round(confidence, 6),
            "prob_genuine": round(prob_genuine, 6),
            "prob_fake":    round(prob_fake, 6),
            "inference_ms": round(elapsed_ms, 2),
        }

        return result

    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_batch(
        self,
        audio_paths: List[Union[str, Path]],
        batch_size: int = 8,
    ) -> List[Dict]:
        """
        Run batch inference on a list of audio files.

        Spectrograms are computed in parallel (via list) and collated
        into mini-batches for efficient GPU utilization.

        Args:
            audio_paths: List of paths to audio files.
            batch_size:  Number of files to process per forward pass.

        Returns:
            List of result dicts (same structure as predict_file).
        """
        results = []
        n = len(audio_paths)

        for batch_start in range(0, n, batch_size):
            batch_paths = audio_paths[batch_start : batch_start + batch_size]
            specs_list  = []
            valid_paths = []

            for path in batch_paths:
                try:
                    spec = preprocess_audio(
                        audio_input=str(path),
                        cfg=self.cfg,
                        augment=False,
                    )
                    specs_list.append(spec)
                    valid_paths.append(path)
                except Exception as e:
                    logger.error(f"Failed to preprocess {path}: {e}")
                    results.append({
                        "file":       str(path),
                        "prediction": "ERROR",
                        "error":      str(e),
                    })

            if not specs_list:
                continue

            batch_tensor = torch.stack(specs_list).to(self.device)  # [B, 3, 224, 224]
            logits = self.model(batch_tensor)                        # [B, 2]
            probs  = F.softmax(logits, dim=1)                       # [B, 2]

            for i, path in enumerate(valid_paths):
                prob_genuine = probs[i, 0].item()
                prob_fake    = probs[i, 1].item()
                label_id     = 1 if prob_fake >= self.threshold else 0
                prediction   = CLASS_NAMES[label_id]
                confidence   = prob_fake if label_id == 1 else prob_genuine

                results.append({
                    "file":         str(path),
                    "prediction":   prediction,
                    "label_id":     label_id,
                    "confidence":   round(confidence, 6),
                    "prob_genuine": round(prob_genuine, 6),
                    "prob_fake":    round(prob_fake, 6),
                })

            logger.info(
                f"Processed {min(batch_start + batch_size, n)}/{n} files …"
            )

        return results

    # ------------------------------------------------------------------

    def predict_directory(
        self,
        directory:  Union[str, Path],
        batch_size: int = 8,
        recursive:  bool = True,
    ) -> List[Dict]:
        """
        Predict on all audio files found in a directory.

        Args:
            directory:  Root directory to scan.
            batch_size: Batch size for inference.
            recursive:  If True, scan subdirectories recursively.

        Returns:
            List of result dicts.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        glob_fn   = directory.rglob if recursive else directory.glob
        all_files = sorted([
            f for f in glob_fn("*")
            if f.suffix.lower() in AUDIO_EXTENSIONS
        ])

        logger.info(f"Found {len(all_files)} audio file(s) in '{directory}'")
        if not all_files:
            logger.warning("No audio files found.")
            return []

        return self.predict_batch(all_files, batch_size=batch_size)

    # ------------------------------------------------------------------

    def predict_waveform(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> Dict:
        """
        Predict directly from a raw numpy waveform array.

        Useful for integration into streaming or real-time pipelines.

        Args:
            waveform:    Float32 numpy array of shape [T] or [C, T].
            sample_rate: Sampling rate of the waveform.

        Returns:
            Result dict (same structure as predict_file, file='<waveform>').
        """
        import io
        # Inject actual sample rate via a shallow config copy
        import copy
        cfg_copy = copy.deepcopy(self.cfg)
        cfg_copy["dataset"]["sample_rate"] = sample_rate

        spec = preprocess_audio(
            audio_input=waveform,
            cfg=cfg_copy,
            augment=False,
        )
        spec = spec.unsqueeze(0).to(self.device)   # [1, 3, 224, 224]

        with torch.no_grad():
            logits = self.model(spec)
            probs  = F.softmax(logits, dim=1)

        prob_genuine = probs[0, 0].item()
        prob_fake    = probs[0, 1].item()
        label_id     = 1 if prob_fake >= self.threshold else 0
        prediction   = CLASS_NAMES[label_id]
        confidence   = prob_fake if label_id == 1 else prob_genuine

        return {
            "file":         "<waveform>",
            "prediction":   prediction,
            "label_id":     label_id,
            "confidence":   round(confidence, 6),
            "prob_genuine": round(prob_genuine, 6),
            "prob_fake":    round(prob_fake, 6),
        }


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _print_result(result: Dict) -> None:
    """Pretty-print a single inference result."""
    icon = "🔴 FAKE" if result.get("label_id") == 1 else "🟢 GENUINE"
    print(f"\n  File       : {result['file']}")
    print(f"  Prediction : {icon}")
    print(f"  Confidence : {result.get('confidence', 'N/A'):.4f}")
    print(f"  P(Genuine) : {result.get('prob_genuine', 'N/A'):.4f}")
    print(f"  P(Fake)    : {result.get('prob_fake', 'N/A'):.4f}")
    if "inference_ms" in result:
        print(f"  Latency    : {result['inference_ms']:.1f} ms")


def _print_batch_summary(results: List[Dict]) -> None:
    """Print an aggregate summary for batch predictions."""
    total   = len(results)
    genuine = sum(1 for r in results if r.get("label_id") == 0)
    fake    = sum(1 for r in results if r.get("label_id") == 1)
    errors  = sum(1 for r in results if r.get("prediction") == "ERROR")

    print(f"\n{'─'*50}")
    print(f"  Total files : {total}")
    print(f"  Genuine     : {genuine}")
    print(f"  Fake        : {fake}")
    print(f"  Errors      : {errors}")
    print(f"{'─'*50}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ResNet++ Audio Forgery Detection — Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file:
  python inference.py --audio speech.wav

  # Directory (all audio files):
  python inference.py --audio ./recordings/ --output results.json

  # Custom checkpoint + threshold:
  python inference.py --audio clip.flac \\
      --checkpoint checkpoints/checkpoint_epoch_019.pth \\
      --threshold 0.6
""",
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to an audio file OR a directory containing audio files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint. Defaults to checkpoints/best_model.pth",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold on P(fake). Default: 0.5",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for directory inference. Default: 8",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save results as JSON (e.g., outputs/results.json)",
    )
    parser.add_argument(
        "--no_recursive",
        action="store_true",
        help="Disable recursive directory scanning",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # ── Setup
    setup_logger("resnet_forgery", log_dir="logs")
    set_seed(args.seed)

    # ── Build predictor
    predictor = AudioForgeryPredictor(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
    )

    # ── Run inference
    audio_path = Path(args.audio)

    if audio_path.is_dir():
        results = predictor.predict_directory(
            directory=audio_path,
            batch_size=args.batch_size,
            recursive=not args.no_recursive,
        )
        for r in results:
            _print_result(r)
        _print_batch_summary(results)

    elif audio_path.is_file():
        result = predictor.predict_file(audio_path)
        _print_result(result)
        results = [result]

    else:
        raise FileNotFoundError(
            f"'{args.audio}' is neither an existing file nor directory."
        )

    # ── Optionally save to JSON
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_path}")
        print(f"\nResults saved → {output_path}")


if __name__ == "__main__":
    main()
