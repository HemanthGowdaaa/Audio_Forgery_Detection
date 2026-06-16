"""HTML dashboard generation with embedded Plotly charts."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.offline import plot


def _metric_table(title: str, metrics: dict[str, Any]) -> str:
    keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    rows = "".join(
        f"<tr><td>{key}</td><td>{float(metrics.get(key, 0.0)):.4f}</td></tr>"
        for key in keys
    )
    return f"<section><h2>{title}</h2><table><tbody>{rows}</tbody></table></section>"


def _confusion_chart(name: str, metrics: dict[str, Any]) -> str:
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Pred REAL", "Pred FAKE"],
        y=["True REAL", "True FAKE"],
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
    ))
    fig.update_layout(title=f"{name} Confusion Matrix", height=360, margin=dict(t=50, l=40, r=20, b=30))
    return plot(fig, include_plotlyjs=False, output_type="div")


def _curve_chart(title: str, resnet: dict[str, Any], svm: dict[str, Any], curve: str) -> str:
    fig = go.Figure()
    if curve == "roc":
        x_key, y_key = "roc_fpr", "roc_tpr"
        x_title, y_title = "False Positive Rate", "True Positive Rate"
    else:
        x_key, y_key = "pr_recall", "pr_precision"
        x_title, y_title = "Recall", "Precision"
    for name, metrics in (("ResNet++", resnet), ("SVM", svm)):
        curves = metrics.get("curves", {})
        fig.add_trace(go.Scatter(x=curves.get(x_key, []), y=curves.get(y_key, []), mode="lines", name=name))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, height=420)
    return plot(fig, include_plotlyjs=False, output_type="div")


def _comparison_chart(resnet: dict[str, Any], svm: dict[str, Any]) -> str:
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="ResNet++", x=metrics, y=[resnet.get(m, 0.0) for m in metrics]))
    fig.add_trace(go.Bar(name="SVM", x=metrics, y=[svm.get(m, 0.0) for m in metrics]))
    fig.update_layout(title="Model Comparison", barmode="group", yaxis_range=[0, 1], height=420)
    return plot(fig, include_plotlyjs=False, output_type="div")


def generate_report(
    summary: dict[str, Any],
    resnet_metrics: dict[str, Any],
    svm_metrics: dict[str, Any],
    best_model: str,
    cfg: dict,
    output_path: str | Path = "outputs/report.html",
) -> None:
    """Generate a self-contained HTML report."""
    plotly_js = plot(go.Figure(), include_plotlyjs=True, output_type="div").split("</script>", 1)[0] + "</script>"
    css = """
    body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:0;background:#f5f7fb;color:#172033}
    header{background:#172033;color:white;padding:32px 40px} main{max-width:1180px;margin:0 auto;padding:28px}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px}
    section{background:white;border:1px solid #dfe6f1;border-radius:8px;padding:20px;margin-bottom:18px;box-shadow:0 1px 2px #0001}
    h1,h2{margin:0 0 14px} table{width:100%;border-collapse:collapse} td,th{border-bottom:1px solid #e6edf5;padding:10px;text-align:left}
    .badge{display:inline-block;background:#0f766e;color:white;border-radius:6px;padding:6px 10px;font-weight:700}
    .muted{color:#667085}.hero{display:flex;gap:18px;flex-wrap:wrap}.stat{background:#edf4ff;border-radius:8px;padding:14px;min-width:160px}
    """
    dataset_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in summary.items()
    )
    config_text = html.escape(json.dumps({
        "sample_rate": cfg["dataset"]["sample_rate"],
        "duration": cfg["dataset"]["duration"],
        "seed": cfg["dataset"]["seed"],
        "epochs": cfg["training"]["epochs"],
        "batch_size": cfg["training"]["batch_size"],
        "svm_grid": cfg["svm"],
    }, indent=2))
    body = f"""
    <!doctype html><html><head><meta charset="utf-8"><title>Audio Deepfake Detection Report</title>
    <style>{css}</style>{plotly_js}</head><body>
    <header><h1>Audio Deepfake Detection Dashboard</h1><p class="muted">Local dataset pipeline, ResNet++ vs SVM</p></header>
    <main>
      <section><h2>Best Model</h2><span class="badge">{html.escape(best_model.upper())}</span></section>
      <div class="grid">
        <section><h2>Dataset Summary</h2><table>{dataset_rows}</table></section>
        <section><h2>Training Configuration</h2><pre>{config_text}</pre></section>
      </div>
      <div class="grid">{_metric_table("ResNet++ Results", resnet_metrics)}{_metric_table("SVM Results", svm_metrics)}</div>
      <section><h2>Model Comparison</h2>{_comparison_chart(resnet_metrics, svm_metrics)}</section>
      <div class="grid"><section>{_confusion_chart("ResNet++", resnet_metrics)}</section><section>{_confusion_chart("SVM", svm_metrics)}</section></div>
      <section><h2>ROC Curves</h2>{_curve_chart("ROC Curves", resnet_metrics, svm_metrics, "roc")}</section>
      <section><h2>Precision Recall Curves</h2>{_curve_chart("Precision Recall Curves", resnet_metrics, svm_metrics, "pr")}</section>
      <section><h2>Training Time</h2><table><tr><th>Model</th><th>Seconds</th></tr>
      <tr><td>ResNet++</td><td>{resnet_metrics.get("training_time_sec", 0)}</td></tr>
      <tr><td>SVM</td><td>{svm_metrics.get("training_time_sec", 0)}</td></tr></table></section>
      <section><h2>Dataset Statistics</h2><p>Skipped files are logged in <code>outputs/skipped_files.log</code>. Split manifests are saved in <code>outputs/*_manifest.csv</code>.</p></section>
    </main></body></html>
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(body, encoding="utf-8")
