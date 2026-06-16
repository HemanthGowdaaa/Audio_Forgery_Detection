# ResNet++ Audio Forgery Detection

A production-ready implementation of the **ResNet++ Audio Forgery Detection** framework with CBAM attention, SE blocks, Transformer branches, and multi-scale feature fusion — trained on the `mueller91/In-The-Wild` dataset.

---

## Project Structure

```
model2_resnet/
│
├── configs/
│   └── config.yaml        # All hyperparameters and paths
│
├── checkpoints/           # Saved model checkpoints (.pth)
├── logs/                  # TensorBoard logs + training log files
├── outputs/               # Evaluation plots + metrics JSON
├── cache/                 # Preprocessed spectrogram cache
│
├── dataset.py             # HuggingFace dataset loader + DataLoader factory
├── preprocess.py          # Full audio preprocessing pipeline
├── model.py               # ResNet++ model (CBAM + SE + Transformer + MSF)
├── train.py               # Training loop with AMP, early stopping, checkpointing
├── evaluate.py            # Full evaluation with all 9 metrics + plots
├── inference.py           # Inference script (file / directory / waveform)
├── utils.py               # Shared utilities (EER, logging, metrics, etc.)
└── requirements.txt
```

---

## Architecture

```
Input [B, 3, 224, 224]
        │
   ResNet50 Backbone (pretrained, FC removed)
        │  [B, 2048, 7, 7]
        │
   CBAM (Channel Attention + Spatial Attention)
        │  [B, 2048, 7, 7]
        │
   ┌────┴────┐
   │         │
SE Block   Transformer Branch
   │    (flatten→49 tokens→MHSA→reshape)
   └────┬────┘
        │  element-wise sum → [B, 2048, 7, 7]
        │
   Multi-Scale Fusion (1×1, 3×3, 5×5, 7×7 parallel convs)
        │  [B, 2048, 7, 7]
        │
   Classification Head
   GAP → Linear(2048→512) → BN → ReLU → Dropout(0.5) → Linear(512→2)
        │
   Logits [B, 2]
```

---

## Quick Start

### 1. Install dependencies

```bash
# For Apple Silicon (MPS)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# For CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Train

```bash
python train.py --config configs/config.yaml
```

Training will:
- Auto-download the `mueller91/In-The-Wild` dataset from HuggingFace
- Cache preprocessed spectrograms to `./cache/` for fast subsequent epochs
- Save best model to `checkpoints/best_model.pth`
- Log metrics to TensorBoard under `logs/`

### 3. Monitor with TensorBoard

```bash
tensorboard --logdir logs/
```

### 4. Evaluate

```bash
# Evaluate on test split (uses checkpoints/best_model.pth by default)
python evaluate.py --split test

# Use a specific checkpoint
python evaluate.py --split test --checkpoint checkpoints/checkpoint_epoch_019.pth
```

Outputs saved to `outputs/`:
- `test_metrics.json`
- `test_confusion_matrix.png`
- `test_roc_curve.png`
- `test_pr_curve.png`

### 5. Inference

```bash
# Single audio file
python inference.py --audio path/to/audio.wav

# All audio files in a directory
python inference.py --audio recordings/ --output outputs/results.json

# Custom threshold
python inference.py --audio clip.flac --threshold 0.6
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Precision | TP / (TP + FP) for fake class |
| Recall | TP / (TP + FN) for fake class |
| F1 Score | Harmonic mean of precision & recall |
| ROC-AUC | Area under ROC curve |
| EER | Equal Error Rate (FAR = FRR crossover) |
| FPR | False Positive Rate |
| TNR | True Negative Rate (Specificity) |
| Confusion Matrix | Full TP/TN/FP/FN breakdown |

---

## Configuration

Edit `configs/config.yaml` to adjust any hyperparameter:

```yaml
training:
  epochs: 50
  batch_size: 16
  learning_rate: 1.0e-4
  early_stopping_patience: 10

model:
  transformer_heads: 8
  transformer_layers: 2
  cbam_reduction_ratio: 16
```

---

## Device Support

The framework auto-detects the best available device:

| Priority | Device |
|----------|--------|
| 1st | CUDA GPU |
| 2nd | Apple Silicon MPS |
| 3rd | CPU |

---

## Dataset

**mueller91/In-The-Wild** — A real-world audio deepfake detection benchmark containing genuine and synthesized speech recordings.

- Loaded via `datasets.load_dataset("mueller91/In-The-Wild")`
- Binary classification: `0 = Genuine`, `1 = Fake`
- Auto-split into train / val / test if not pre-split

---

## Reproducibility

All runs are seeded via `utils.set_seed(seed)`. The default seed is `42`, configurable in `configs/config.yaml` under `training.seed`.
