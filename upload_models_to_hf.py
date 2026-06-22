#!/usr/bin/env python3
"""
upload_models_to_hf.py
======================
One-time script to upload model files from your local machine to
Hugging Face Hub repo: Hkm2003/audio-forgery-models

Usage:
    pip install huggingface_hub
    huggingface-cli login          # enter your HF token
    python upload_models_to_hf.py
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO_ID   = "Hkm2003/audio-forgery-models-bucket"
REPO_TYPE = "model"

# Files to upload: (local_path, path_in_repo)
PROJECT_ROOT = Path(__file__).resolve().parent
UPLOADS = [
    (PROJECT_ROOT / "outputs/best_model/best_resnet.pth",   "best_model/best_resnet.pth"),
    (PROJECT_ROOT / "outputs/best_model/best_svm.joblib",   "best_model/best_svm.joblib"),
    (PROJECT_ROOT / "outputs/best_model/metadata.json",     "best_model/metadata.json"),
    (PROJECT_ROOT / "outputs/svm_metrics.json",             "svm_metrics.json"),
    (PROJECT_ROOT / "outputs/resnet_metrics.json",          "resnet_metrics.json"),
    (PROJECT_ROOT / "outputs/model_comparison.json",        "model_comparison.json"),
]

api = HfApi()

# Create the repo if it doesn't exist (set private=False for free public hosting)
print(f"Creating/verifying repo: {REPO_ID}")
create_repo(REPO_ID, repo_type=REPO_TYPE, exist_ok=True, private=False)

for local_path, repo_path in UPLOADS:
    if not local_path.exists():
        print(f"  ⚠️  SKIPPING (not found): {local_path}")
        continue
    size_mb = local_path.stat().st_size / (1024 * 1024)
    print(f"  ⬆️  Uploading: {local_path.name} ({size_mb:.1f} MB) → {repo_path}")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=repo_path,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )
    print(f"  ✅ Done: {repo_path}")

print("\n🎉 All model files uploaded to HuggingFace Hub!")
print(f"   View at: https://huggingface.co/{REPO_ID}")
