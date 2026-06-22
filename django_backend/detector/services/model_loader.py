import json
import logging
from pathlib import Path
import joblib
import torch

from django.conf import settings

# Import our root package modules
from audio_forgery.config import load_config
from models.model import build_model
from models.utils import get_device

logger = logging.getLogger("backend_server")

class ModelCache:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelCache, cls).__new__(cls, *args, **kwargs)
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
            p_path = Path(p)
            if p_path.exists():
                return p_path
        return None

    def load_models(self):
        config_path = Path(settings.ML_CONFIG_PATH)
        if not config_path.is_absolute():
            config_path = settings.PROJECT_ROOT / config_path
            
        try:
            self.cfg = load_config(str(config_path))
            logger.info(f"Loaded config from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise e

        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        # Resolve paths relative to PROJECT_ROOT
        best_model_dir_rel = self.cfg["paths"]["best_model_dir"].lstrip("./")
        output_dir_rel = self.cfg["paths"]["output_dir"].lstrip("./")
        checkpoint_dir_rel = self.cfg["paths"]["checkpoint_dir"].lstrip("./")

        best_dir = settings.PROJECT_ROOT / best_model_dir_rel
        alt_dir = settings.PROJECT_ROOT / "outputs/best_model"

        # Load metadata
        metadata_path = self._find_file(best_dir / "metadata.json", alt_dir / "metadata.json")
        if metadata_path:
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata from {metadata_path}")
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
        else:
            logger.warning("Metadata file not found.")

        # Load SVM
        svm_path = self._find_file(
            best_dir / "best_svm.joblib",
            alt_dir / "best_svm.joblib",
            settings.PROJECT_ROOT / output_dir_rel / "svm_model.joblib",
        )
        if svm_path:
            try:
                self.svm_model = joblib.load(svm_path)
                logger.info(f"✅ Loaded SVM model from {svm_path}")
            except Exception as e:
                logger.error(f"Failed to load SVM model: {e}")
        else:
            logger.warning("SVM model file not found.")

        # Load ResNet++
        resnet_path = self._find_file(
            best_dir / "best_resnet.pth",
            alt_dir / "best_resnet.pth",
            settings.PROJECT_ROOT / checkpoint_dir_rel / "best_resnet.pth",
        )
        if resnet_path:
            try:
                # Use map_location for MPS/CPU safety
                checkpoint = torch.load(resnet_path, map_location=self.device)
                model_cfg = checkpoint.get("cfg", self.cfg)
                self.resnet_model = build_model(model_cfg, self.device)
                self.resnet_model.load_state_dict(checkpoint["model_state_dict"])
                self.resnet_model.eval()
                logger.info(f"✅ Loaded ResNet++ model from {resnet_path}")
            except Exception as e:
                logger.error(f"Failed to load ResNet++ model: {e}")
        else:
            logger.warning("ResNet++ checkpoint file not found.")

def get_model_cache():
    return ModelCache()
