"""
benchmarks/eval_metrics.py
============================
Compute Precision, Recall, F1 for focus detection against ground-truth labels.

Usage:
    python benchmarks/eval_metrics.py \
        --gt   benchmarks/data/gt_session.json \
        --pred benchmarks/data/pred_session.json
"""

import json
import argparse
import numpy as np
from pathlib import Path


def load_labels(path: str) -> list[str]:
    """Load a list of state labels from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [row["label"] for row in data]


def align_labels(gt: list[str], pred: list[str]) -> tuple[list[str], list[str]]:
    """Trim to the shorter list for frame alignment."""
    n = min(len(gt), len(pred))
    return gt[:n], pred[:n]


def compute_metrics(gt: list[str], pred: list[str]) -> dict:
    """
    Compute per-class Precision, Recall, F1 and macro averages.

    Parameters
    ----------
    gt   : list[str]  -- ground-truth labels
    pred : list[str]  -- predicted labels

    Returns
    -------
    dict with keys: precision, recall, f1, accuracy, confusion
    """
    classes = sorted(set(gt) | set(pred))
    results = {}

    for cls in classes:
        tp = sum(1 for g, p in zip(gt, pred) if g == cls and p == cls)
        fp = sum(1 for g, p in zip(gt, pred) if g != cls and p == cls)
        fn = sum(1 for g, p in zip(gt, pred) if g == cls and p != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        results[cls] = {"precision": precision, "recall": recall, "f1": f1,
                        "tp": tp, "fp": fp, "fn": fn}

    accuracy = sum(1 for g, p in zip(gt, pred) if g == p) / len(gt)
    macro_f1 = np.mean([v["f1"] for v in results.values()])

    return {
        "per_class": results,
        "accuracy":  round(accuracy, 4),
        "macro_f1":  round(macro_f1, 4),
    }


def print_results(metrics: dict) -> None:
    print(f"\n{'Class':<14} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 44)
    for cls, m in metrics["per_class"].items():
        print(f"{cls:<14} {m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>8.3f}")
    print("-" * 44)
    print(f"{'Accuracy':<14} {metrics['accuracy']:>10.3f}")
    print(f"{'Macro F1':<14} {metrics['macro_f1']:>10.3f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt",   required=True, help="Ground-truth JSON")
    parser.add_argument("--pred", required=True, help="Predictions JSON")
    args = parser.parse_args()

    gt_labels   = load_labels(args.gt)
    pred_labels = load_labels(args.pred)
    gt_aligned, pred_aligned = align_labels(gt_labels, pred_labels)

    metrics = compute_metrics(gt_aligned, pred_aligned)
    print_results(metrics)
