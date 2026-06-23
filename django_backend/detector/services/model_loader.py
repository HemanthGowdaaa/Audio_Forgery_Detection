"""
model_loader.py — Singleton model cache with HuggingFace auto-download fallback.

If model files are not found locally (e.g. build.sh did not run on Render),
this module downloads them directly from HuggingFace Hub at startup.
Files are cached to disk so subsequent restarts skip the download.
"""

import json
import logging
import os
from pathlib import Path

import joblib
import torch
from django.conf import settings

from audio_forgery.config import load_config
from models.model import build_model
from models.utils import get_device

logger = logging.getLogger("backend_server")

# ── HuggingFace download config ───────────────────────────────────────────────
HF_REPO = os.getenv("HF_REPO_ID", "Hkm2003/audio-forgery-models")

# Files to download: (hf_filename, local_path_resolver)
# local_path_resolver takes (best_dir, alt_dir, outputs_dir) and returns a Path
_HF_FILES = [
    ("best_model/metadata.json",    lambda b, a, o: a / "metadata.json"),
    ("best_model/best_svm.joblib",  lambda b, a, o: a / "best_svm.joblib"),
    ("best_model/best_resnet.pth",  lambda b, a, o: a / "best_resnet.pth"),
    ("svm_metrics.json",            lambda b, a, o: o / "svm_metrics.json"),
    ("resnet_metrics.json",         lambda b, a, o: o / "resnet_metrics.json"),
    ("model_comparison.json",       lambda b, a, o: o / "model_comparison.json"),
]


def _download_from_hf(hf_filename: str, local_dest: Path) -> bool:
    """Download a single file from HuggingFace Hub. Returns True on success."""
    try:
        from huggingface_hub import hf_hub_download
        logger.info(f"⬇️  Downloading from HF: {hf_filename} → {local_dest}")
        local_dest.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=HF_REPO,
            filename=hf_filename,
            local_dir=str(local_dest.parent),
            local_dir_use_symlinks=False,
        )
        # hf_hub_download saves to local_dir/basename(hf_filename)
        landing = local_dest.parent / Path(hf_filename).name
        if landing.exists() and landing != local_dest:
            landing.rename(local_dest)
        if local_dest.exists():
            size_mb = local_dest.stat().st_size / 1_048_576
            logger.info(f"✅ Downloaded: {local_dest.name} ({size_mb:.1f} MB)")
            return True
        else:
            logger.error(f"❌ Download finished but file missing: {local_dest}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to download {hf_filename}: {e}")
        return False


def _ensure_model_files(best_dir: Path, alt_dir: Path, outputs_dir: Path):
    """
    Download any missing model files from HuggingFace Hub.
    Skips files that already exist locally.
    """
    alt_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for hf_filename, resolver in _HF_FILES:
        local_path = resolver(best_dir, alt_dir, outputs_dir)
        if not local_path.exists():
            missing.append((hf_filename, local_path))

    if not missing:
        logger.info("✅ All model files found locally — skipping HF download.")
        return

    logger.info(f"📥 {len(missing)} model file(s) not found locally. Downloading from HF Hub: {HF_REPO}")
    for hf_filename, local_path in missing:
        _download_from_hf(hf_filename, local_path)


class ModelCache:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.cfg = None
        self.device = None
        self.svm_model = None
        self.resnet_model = None
        self.metadata = {}
        self._initialized = True

    def _find_file(self, *candidates) -> Path | None:
        for p in candidates:
            p = Path(p)
            if p.exists():
                return p
        return None

    def load_models(self):
        # ── Load config ───────────────────────────────────────────────────────
        config_path = Path(settings.ML_CONFIG_PATH)
        if not config_path.is_absolute():
            config_path = settings.PROJECT_ROOT / config_path

        try:
            self.cfg = load_config(str(config_path))
            logger.info(f"Loaded config from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise

        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        # ── Resolve local model directories ───────────────────────────────────
        best_model_dir_rel  = self.cfg["paths"]["best_model_dir"].lstrip("./")
        output_dir_rel      = self.cfg["paths"]["output_dir"].lstrip("./")
        checkpoint_dir_rel  = self.cfg["paths"]["checkpoint_dir"].lstrip("./")

        best_dir    = settings.PROJECT_ROOT / best_model_dir_rel
        alt_dir     = settings.PROJECT_ROOT / "outputs/best_model"
        outputs_dir = settings.PROJECT_ROOT / output_dir_rel

        # ── Auto-download missing files from HuggingFace ──────────────────────
        _ensure_model_files(best_dir, alt_dir, outputs_dir)

        # ── Load metadata ─────────────────────────────────────────────────────
        metadata_path = self._find_file(
            alt_dir / "metadata.json",
            best_dir / "metadata.json",
        )
        if metadata_path:
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata from {metadata_path}")
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
        else:
            logger.warning("Metadata file not found even after HF download attempt.")

        # ── Load SVM model ────────────────────────────────────────────────────
        svm_path = self._find_file(
            alt_dir  / "best_svm.joblib",
            best_dir / "best_svm.joblib",
            outputs_dir / "svm_model.joblib",
        )
        if svm_path:
            try:
                self.svm_model = joblib.load(svm_path)
                logger.info(f"✅ Loaded SVM model from {svm_path}")
            except Exception as e:
                logger.error(f"Failed to load SVM model: {e}")
        else:
            logger.warning("SVM model file not found even after HF download attempt.")

        # ── Load ResNet++ model ───────────────────────────────────────────────
        resnet_path = self._find_file(
            alt_dir  / "best_resnet.pth",
            best_dir / "best_resnet.pth",
            settings.PROJECT_ROOT / checkpoint_dir_rel / "best_resnet.pth",
        )
        if resnet_path:
            try:
                checkpoint = torch.load(resnet_path, map_location=self.device, weights_only=False)
                model_cfg = checkpoint.get("cfg", self.cfg)
                self.resnet_model = build_model(model_cfg, self.device)
                self.resnet_model.load_state_dict(checkpoint["model_state_dict"])
                self.resnet_model.eval()
                logger.info(f"✅ Loaded ResNet++ model from {resnet_path}")
            except Exception as e:
                logger.error(f"Failed to load ResNet++ model: {e}")
        else:
            logger.warning("ResNet++ checkpoint file not found even after HF download attempt.")


def get_model_cache() -> ModelCache:
    return ModelCache()
