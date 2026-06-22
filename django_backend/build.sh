#!/usr/bin/env bash
# =============================================================================
# build.sh — Render Build Script for Audio Forgery Detection Django Backend
# =============================================================================
# Runs during the Render build phase (before the server starts).
#   1. Install Python dependencies
#   2. Download trained ML models from Hugging Face Hub (Hkm2003/audio-forgery-models-bucket)
#   3. Run Django collectstatic + migrate
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

# The HF_REPO_ID env var is set in render.yaml / Render dashboard
HF_REPO="${HF_REPO_ID:-Hkm2003/audio-forgery-models-bucket}"

# Resolve the project root (one level up from django_backend/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/outputs/best_model"
METRICS_DIR="${PROJECT_ROOT}/outputs"

echo "   HF Repo  : ${HF_REPO}"
echo "   Model Dir: ${MODEL_DIR}"

mkdir -p "${MODEL_DIR}"
mkdir -p "${METRICS_DIR}"

# Download each file using huggingface_hub Python API
HF_REPO="${HF_REPO}" PROJECT_ROOT="${PROJECT_ROOT}" python3 - <<'PYEOF'
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

HF_REPO    = os.environ["HF_REPO"]
MODEL_DIR  = Path(os.environ["PROJECT_ROOT"]) / "outputs" / "best_model"
METRICS_DIR = Path(os.environ["PROJECT_ROOT"]) / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Files to download: (filename_in_hf_repo, local_destination)
downloads = [
    ("best_model/best_resnet.pth",   MODEL_DIR  / "best_resnet.pth"),
    ("best_model/best_svm.joblib",   MODEL_DIR  / "best_svm.joblib"),
    ("best_model/metadata.json",     MODEL_DIR  / "metadata.json"),
    ("svm_metrics.json",             METRICS_DIR / "svm_metrics.json"),
    ("resnet_metrics.json",          METRICS_DIR / "resnet_metrics.json"),
    ("model_comparison.json",        METRICS_DIR / "model_comparison.json"),
]

all_ok = True
for hf_filename, local_path in downloads:
    if local_path.exists():
        print(f"   ✅ Already exists: {local_path.name}", flush=True)
        continue
    try:
        print(f"   ⬇️  Downloading: {hf_filename} ...", flush=True)
        hf_hub_download(
            repo_id=HF_REPO,
            filename=hf_filename,
            local_dir=str(local_path.parent),
            local_dir_use_symlinks=False,
        )
        # The file lands at local_path.parent / basename(hf_filename)
        downloaded = local_path.parent / Path(hf_filename).name
        if downloaded.exists() and downloaded != local_path:
            downloaded.rename(local_path)
        if local_path.exists():
            size_mb = local_path.stat().st_size / 1_048_576
            print(f"   ✅ Saved: {local_path.name} ({size_mb:.1f} MB)", flush=True)
        else:
            print(f"   ⚠️  File not found after download: {local_path}", file=sys.stderr)
            all_ok = False
    except Exception as e:
        print(f"   ⚠️  WARNING: Could not download {hf_filename}: {e}", file=sys.stderr)
        all_ok = False

if all_ok:
    print("   ✅ All model files downloaded successfully.", flush=True)
else:
    print("   ⚠️  Some model files are missing — server will start with fallback metrics.", file=sys.stderr)
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
