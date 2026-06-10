"""
evaluate.py
Aggregate per-window predictions from window_predictions.csv into per-clip
accuracy metrics, following the same logic as merge_isolated / compute_video.

Usage:
    python evaluate.py --input window_predictions.csv
    python evaluate.py --input window_predictions.csv --method score  # raw logits instead of softmax
    python evaluate.py --input window_predictions.csv --output results/
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ─────────────────────────────────────────────
# 1. CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate window-level predictions at clip level")
    parser.add_argument("--input",  required=True, help="Path to window_predictions.csv")
    parser.add_argument("--method", choices=["prob", "score"], default="prob",
                        help="Aggregation method: 'prob' = softmax then mean, 'score' = mean of raw logits")
    parser.add_argument("--output", default=".", help="Directory to save result files")
    return parser.parse_args()


# ─────────────────────────────────────────────
# 2. LOAD CSV
# ─────────────────────────────────────────────

def load_predictions(csv_path: str):
    """
    Read window_predictions.csv.
    Expected columns: clip_path, win_idx, logit_0 … logit_N, predicted, ground_truth
    Returns a DataFrame.
    """
    df = pd.read_csv(csv_path)
    return df


# ─────────────────────────────────────────────
# 3. AGGREGATE  (window → clip)
# ─────────────────────────────────────────────

def aggregate(df: pd.DataFrame, method: str = "prob"):
    """
    For each unique clip_path:
        - collect all window logit vectors
        - apply softmax (method='prob') or keep raw (method='score')
        - average across windows
        - argmax → clip-level prediction

    Returns a list of dicts:
        [{clip_path, label, predicted, top1, top5, mean_feat}, ...]
    """
    logit_cols = [c for c in df.columns if c.startswith("logit_")]
    num_classes = len(logit_cols)

    results = []
    for clip_path, group in df.groupby("clip_path", sort=False):
        label = int(group["ground_truth"].iloc[0])
        feats = group[logit_cols].values.astype(float)   # [n_windows, n_classes]

        if method == "prob":
            feats = softmax(feats, axis=1)

        mean_feat = feats.mean(axis=0)                   # [n_classes]
        predicted = int(np.argmax(mean_feat))
        top1 = float(predicted == label)
        top5 = float(label in np.argsort(-mean_feat)[:5])

        results.append({
            "clip_path":  clip_path,
            "label":      label,
            "predicted":  predicted,
            "top1":       top1,
            "top5":       top5,
            "mean_feat":  mean_feat,
            "n_windows":  len(group),
        })

    return results, num_classes


# ─────────────────────────────────────────────
# 4. METRICS
# ─────────────────────────────────────────────

def compute_metrics(results: list, num_classes: int):
    y_true = [r["label"]     for r in results]
    y_pred = [r["predicted"] for r in results]

    final_top1 = float(np.mean([r["top1"] for r in results])) * 100
    final_top5 = float(np.mean([r["top5"] for r in results])) * 100

    # Per-class accuracy
    correct = [0] * num_classes
    total   = [0] * num_classes
    for r in results:
        total[r["label"]] += 1
        if r["top1"] == 1.0:
            correct[r["label"]] += 1
    class_acc = [
        correct[i] / total[i] * 100 if total[i] > 0 else float("nan")
        for i in range(num_classes)
    ]

    # Confusion matrix
    cm      = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # sklearn classification report
    report = classification_report(
        y_true, y_pred,
        labels=list(range(num_classes)),
        zero_division=0,
    )

    return {
        "top1":       final_top1,
        "top5":       final_top5,
        "class_acc":  class_acc,
        "cm":         cm,
        "cm_norm":    cm_norm,
        "report":     report,
        "y_true":     y_true,
        "y_pred":     y_pred,
    }


# ─────────────────────────────────────────────
# 5. SAVE / PRINT
# ─────────────────────────────────────────────

def print_results(metrics: dict, num_classes: int):
    print("\n" + "=" * 50)
    print(f"  Clip-level Top-1 Accuracy : {metrics['top1']:.2f}%")
    print(f"  Clip-level Top-5 Accuracy : {metrics['top5']:.2f}%")
    print("=" * 50)
    print("\nPer-class accuracy:")
    for i, acc in enumerate(metrics["class_acc"]):
        print(f"  Class {i:>2d} : {acc:.2f}%")
    print("\nClassification Report:")
    print(metrics["report"])


def save_results(metrics: dict, num_classes: int, output_dir: str, results: list):
    os.makedirs(output_dir, exist_ok=True)

    # ── clip-level predictions CSV ────────────────────────────────────────
    clip_df = pd.DataFrame([
        {"clip_path": r["clip_path"], "label": r["label"],
         "predicted": r["predicted"], "top1": r["top1"], "top5": r["top5"],
         "n_windows": r["n_windows"]}
        for r in results
    ])
    clip_csv = os.path.join(output_dir, "clip_predictions.csv")
    clip_df.to_csv(clip_csv, index=False)
    print(f"\nClip-level predictions saved to: {clip_csv}")

    # ── summary txt ──────────────────────────────────────────────────────
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Top-1: {metrics['top1']:.2f}%\n")
        f.write(f"Top-5: {metrics['top5']:.2f}%\n\n")
        f.write("Per-class accuracy:\n")
        for i, acc in enumerate(metrics["class_acc"]):
            f.write(f"  Class {i}: {acc:.2f}%\n")
        f.write("\nClassification Report:\n")
        f.write(metrics["report"])
    print(f"Summary saved to: {summary_path}")

    # ── confusion matrix plot ─────────────────────────────────────────────
    _plot_confusion_matrix(metrics["cm_norm"], num_classes, output_dir)


def _plot_confusion_matrix(cm_norm, num_classes, output_dir):
    fig, ax = plt.subplots(figsize=(max(6, num_classes), max(5, num_classes - 1)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xlabel="Predicted class",
        ylabel="True class",
        title="Normalized Confusion Matrix",
    )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))

    thresh = 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_norm[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}",
                        ha="center", va="center",
                        color="white" if val > thresh else "black",
                        fontsize=8)

    fig.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")


# 6. MAIN

def main():
    args = parse_args()

    print(f"Loading predictions from: {args.input}")
    df = load_predictions(args.input)
    print(f"  {len(df)} windows, {df['clip_path'].nunique()} clips, "
          f"{df['ground_truth'].nunique()} classes")

    print(f"Aggregating with method='{args.method}' ...")
    results, num_classes = aggregate(df, method=args.method)

    metrics = compute_metrics(results, num_classes)
    print_results(metrics, num_classes)
    save_results(metrics, num_classes, args.output, results)


if __name__ == "__main__":
    main()