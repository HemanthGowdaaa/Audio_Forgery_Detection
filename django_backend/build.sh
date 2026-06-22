#!/usr/bin/env bash
# =============================================================================
# build.sh — Render Build Script for Audio Forgery Detection Django Backend
# =============================================================================
# rootDir in render.yaml is "django_backend", so this script runs FROM
# inside the django_backend/ directory. PROJECT_ROOT is one level up.
# =============================================================================

set -o errexit   # Exit immediately on any error

echo "═══════════════════════════════════════════════════"
echo " 🔧 Audio Forgery Detection — Render Build Script"
echo "═══════════════════════════════════════════════════"
echo "   PWD         : $(pwd)"
echo "   Python      : $(python3 --version)"
echo "   Pip         : $(pip --version | cut -d' ' -f1-2)"
echo ""

# ── 1. Install Python dependencies ──────────────────────────────────────────
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# ── 2. Resolve paths ─────────────────────────────────────────────────────────
# When render.yaml sets rootDir: django_backend, Render sets CWD to
# /opt/render/project/src/django_backend  (build.sh runs from here)
DJANGO_DIR="$(pwd)"                          # /opt/render/project/src/django_backend
PROJECT_ROOT="$(cd "${DJANGO_DIR}/.." && pwd)" # /opt/render/project/src

MODEL_DIR="${PROJECT_ROOT}/outputs/best_model"
METRICS_DIR="${PROJECT_ROOT}/outputs"

# ── 3. Download ML model files from Hugging Face Hub ─────────────────────────
echo ""
echo "🤖 Downloading model files from Hugging Face Hub..."
HF_REPO="${HF_REPO_ID:-Hkm2003/audio-forgery-models}"
echo "   HF Repo     : ${HF_REPO}"
echo "   Project Root: ${PROJECT_ROOT}"
echo "   Model Dir   : ${MODEL_DIR}"

mkdir -p "${MODEL_DIR}"
mkdir -p "${METRICS_DIR}"

# Pass paths via env to the heredoc python script
export HF_REPO MODEL_DIR METRICS_DIR
python3 << 'PYEOF'
import os, sys
from pathlib import Path
from huggingface_hub import hf_hub_download

HF_REPO     = os.environ["HF_REPO"]
MODEL_DIR   = Path(os.environ["MODEL_DIR"])
METRICS_DIR = Path(os.environ["METRICS_DIR"])

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# (hf_filename, local_destination_path)
downloads = [
    ("best_model/best_resnet.pth",  MODEL_DIR  / "best_resnet.pth"),
    ("best_model/best_svm.joblib",  MODEL_DIR  / "best_svm.joblib"),
    ("best_model/metadata.json",    MODEL_DIR  / "metadata.json"),
    ("svm_metrics.json",            METRICS_DIR / "svm_metrics.json"),
    ("resnet_metrics.json",         METRICS_DIR / "resnet_metrics.json"),
    ("model_comparison.json",       METRICS_DIR / "model_comparison.json"),
]

errors = []
for hf_filename, dest in downloads:
    if dest.exists():
        size_mb = dest.stat().st_size / 1_048_576
        print(f"   ✅ Cached: {dest.name} ({size_mb:.1f} MB)", flush=True)
        continue
    try:
        print(f"   ⬇️  Downloading {hf_filename} ...", flush=True)
        hf_hub_download(
            repo_id=HF_REPO,
            filename=hf_filename,
            local_dir=str(dest.parent),
            local_dir_use_symlinks=False,
        )
        # hf_hub_download saves to dest.parent/basename(hf_filename)
        landing = dest.parent / Path(hf_filename).name
        if landing.exists() and landing != dest:
            landing.rename(dest)
        if dest.exists():
            size_mb = dest.stat().st_size / 1_048_576
            print(f"   ✅ Saved: {dest.name} ({size_mb:.1f} MB)", flush=True)
        else:
            raise FileNotFoundError(f"File not found after download: {dest}")
    except Exception as e:
        print(f"   ⚠️  WARN: {hf_filename} — {e}", file=sys.stderr, flush=True)
        errors.append(hf_filename)

if errors:
    print(f"\n   ⚠️  Missing files: {errors}", file=sys.stderr)
    print("   ℹ️  Server will use fallback/default metrics.", file=sys.stderr)
else:
    print("   ✅ All model files ready.", flush=True)
PYEOF

# ── 4. Django management commands ────────────────────────────────────────────
echo ""
echo "🗄️  Running Django management commands..."

# Ensure staticfiles directory exists before collectstatic
mkdir -p staticfiles

# Run from django_backend (CWD is already django_backend on Render)
python3 manage.py collectstatic --no-input
python3 manage.py migrate --no-input

echo ""
echo "═══════════════════════════════════════════════════"
echo " ✅ Build complete! Starting gunicorn via Procfile..."
echo "═══════════════════════════════════════════════════"
