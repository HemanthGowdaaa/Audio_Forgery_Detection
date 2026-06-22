#!/usr/bin/env bash
# =============================================================================
# build.sh — Render Build Script for Audio Forgery Detection Django Backend
# =============================================================================
# This script runs during the Render build phase (before the server starts).
# It handles:
#   1. Installing all Python dependencies
#   2. Downloading trained ML model files from Hugging Face Hub
#   3. Running Django management commands (collectstatic, migrate)
# =============================================================================

set -o errexit   # Exit immediately on any error

echo "═══════════════════════════════════════════════════"
echo " 🔧 Audio Forgery Detection — Render Build Script"
echo "═══════════════════════════════════════════════════"

# ── 1. Install Python dependencies ──────────────────────────────────────────
echo ""
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# ── 2. Download ML model files from Hugging Face Hub ────────────────────────
echo ""
echo "🤖 Downloading model files from Hugging Face Hub..."

# The HF_REPO_ID env var must be set in Render dashboard
# e.g. HemanthGowdaaa/audio-forgery-models
HF_REPO="${HF_REPO_ID:-Hkm2003/audio-forgery-models}"

# Resolve the project root (one level up from django_backend/)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/outputs/best_model"
METRICS_DIR="${PROJECT_ROOT}/outputs"

echo "   HF Repo  : ${HF_REPO}"
echo "   Model Dir: ${MODEL_DIR}"

mkdir -p "${MODEL_DIR}"
mkdir -p "${METRICS_DIR}"

# Download each file using huggingface_hub Python API
python3 - <<'PYEOF'
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

HF_REPO = os.environ.get("HF_REPO_ID", "Hkm2003/audio-forgery-models")

# Resolve directories relative to the project root (parent of django_backend)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_DIR = PROJECT_ROOT / "outputs" / "best_model"
METRICS_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Files to download from HuggingFace repo
downloads = [
    # (filename_in_hf_repo, local_destination)
    ("best_model/best_resnet.pth",     MODEL_DIR / "best_resnet.pth"),
    ("best_model/best_svm.joblib",     MODEL_DIR / "best_svm.joblib"),
    ("best_model/metadata.json",       MODEL_DIR / "metadata.json"),
    ("svm_metrics.json",               METRICS_DIR / "svm_metrics.json"),
    ("resnet_metrics.json",            METRICS_DIR / "resnet_metrics.json"),
    ("model_comparison.json",          METRICS_DIR / "model_comparison.json"),
]

for hf_filename, local_path in downloads:
    if local_path.exists():
        print(f"   ✅ Already exists: {local_path.name}")
        continue
    try:
        print(f"   ⬇️  Downloading: {hf_filename} ...", flush=True)
        cached = hf_hub_download(
            repo_id=HF_REPO,
            filename=hf_filename,
            local_dir=str(local_path.parent),
            local_dir_use_symlinks=False,
        )
        # hf_hub_download downloads to local_dir with the filename preserved
        downloaded = Path(local_path.parent) / Path(hf_filename).name
        if downloaded != local_path and downloaded.exists():
            downloaded.rename(local_path)
        print(f"   ✅ Saved: {local_path}", flush=True)
    except Exception as e:
        print(f"   ⚠️  WARNING: Could not download {hf_filename}: {e}", file=sys.stderr)
        print(f"   ℹ️  Backend will start with fallback/default metrics if model files are missing.", file=sys.stderr)

print("   ✅ Model download phase complete.", flush=True)
PYEOF

# ── 3. Django management commands ───────────────────────────────────────────
echo ""
echo "🗄️  Running Django management commands..."
python manage.py collectstatic --no-input
python manage.py migrate --no-input

echo ""
echo "═══════════════════════════════════════════════════"
echo " ✅ Build complete! Server is ready to start."
echo "═══════════════════════════════════════════════════"
