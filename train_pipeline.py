"""Train ResNet++, train SVM, compare, deploy the best model, and report."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from audio_forgery.config import load_config
from audio_forgery.data import load_or_create_splits
from audio_forgery.evaluation import save_json
from audio_forgery.report import generate_report
from audio_forgery.resnet_pipeline import train_resnet
from audio_forgery.svm_pipeline import train_svm


def _setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(log_dir) / "pipeline.log", encoding="utf-8"),
        ],
    )


def _dataset_summary(splits: dict) -> dict[str, int]:
    all_samples = [sample for values in splits.values() for sample in values]
    return {
        "total_samples": len(all_samples),
        "train_samples": len(splits["train"]),
        "validation_samples": len(splits["val"]),
        "test_samples": len(splits["test"]),
        "real_samples": sum(sample.label == 0 for sample in all_samples),
        "fake_samples": sum(sample.label == 1 for sample in all_samples),
        "speakers": len({sample.speaker for sample in all_samples if sample.speaker}),
    }


def _deploy_best(best: str, resnet: dict, svm: dict, cfg: dict) -> None:
    best_dir = Path(cfg["paths"]["best_model_dir"])
    best_dir.mkdir(parents=True, exist_ok=True)
    if resnet.get("model_path"):
        shutil.copy2(resnet["model_path"], best_dir / "best_resnet.pth")
    if svm.get("model_path"):
        shutil.copy2(svm["model_path"], best_dir / "best_svm.joblib")
    metadata = {
        "best_model": best,
        "resnet_metrics": resnet.get("metrics", {}),
        "svm_metrics": svm.get("metrics", {}),
    }
    (best_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete local audio deepfake detection pipeline")
    parser.add_argument("--config", default="models/model2_resnet/configs/config.yaml")
    parser.add_argument("--skip-resnet", action="store_true", help="Only useful for quick SVM checks")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _setup_logging(cfg["paths"]["log_dir"])
    splits = load_or_create_splits(cfg)
    summary = _dataset_summary(splits)
    save_json(summary, Path(cfg["paths"]["output_dir"]) / "dataset_summary.json")

    svm_model_path = Path(cfg["paths"]["output_dir"]) / "svm_model.joblib"
    svm_metrics_path = Path(cfg["paths"]["output_dir"]) / "svm_metrics.json"
    if svm_model_path.exists() and svm_metrics_path.exists():
        import joblib
        logging.getLogger(__name__).info("Loaded cached SVM model and metrics.")
        with open(svm_metrics_path, "r", encoding="utf-8") as f:
            svm_metrics = json.load(f)
        svm = {"model": joblib.load(svm_model_path), "model_path": str(svm_model_path), "metrics": svm_metrics}
    else:
        svm = train_svm(splits, cfg)
    if args.skip_resnet:
        resnet = {"model_path": "", "metrics": svm["metrics"] | {"model_type": "resnet_skipped"}}
    else:
        resnet = train_resnet(splits, cfg)

    best = (
        "svm"
        if args.skip_resnet
        else "resnet"
        if resnet["metrics"].get("f1_score", 0.0) >= svm["metrics"].get("f1_score", 0.0)
        else "svm"
    )
    _deploy_best(best, resnet, svm, cfg)
    comparison = {"best_model": best, "resnet": resnet["metrics"], "svm": svm["metrics"]}
    save_json(comparison, Path(cfg["paths"]["output_dir"]) / "model_comparison.json")
    generate_report(summary, resnet["metrics"], svm["metrics"], best, cfg, Path(cfg["paths"]["output_dir"]) / "report.html")
    print(f"Best model: {best.upper()}")
    print(f"Report saved: {Path(cfg['paths']['output_dir']) / 'report.html'}")


if __name__ == "__main__":
    main()
