# Audio Deepfake Detection

Local-only audio forgery detection using:

- Optimized ResNet++ with ResNet50, CBAM, SE attention, and a transformer branch
- Optimized SVM baseline with MFCC, chroma, spectral contrast, RMS, and ZCR features
- Shared deterministic 70/15/15 train/validation/test splits
- Automated evaluation, model comparison, dashboard generation, and inference

## Dataset

The dataset must already exist locally at:

```bash
dataset/release_in_the_wild/
```

The loader recursively scans `.wav`, `.mp3`, and `.flac` files, reads labels from `meta.csv`, validates audio files, and logs skipped files to `outputs/skipped_files.log`.

## Train Everything

```bash
python train_pipeline.py
```

Outputs are saved under `outputs/`, including:

- `resnet_metrics.json`
- `svm_metrics.json`
- `model_comparison.json`
- `report.html`
- `best_model/`

## Predict

Single file:

```bash
python predict.py path/to/audio.wav
```

Folder:

```bash
python predict.py path/to/folder/
```

Batch predictions are saved to `outputs/predictions.csv`.
