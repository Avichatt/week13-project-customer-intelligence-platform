"""
evaluate.py — ML model evaluation: metrics, quality gate, and plots.
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve
)
from sklearn.calibration import calibration_curve

from src import config


# ── Quality Gate ────────────────────────────────────────────────────────────

GATE_MIN_AUC = 0.75
GATE_MIN_AUC_SAMPLE = 0.70
GATE_AUC_IMPROVEMENT = 0.01
GATE_AUC_IMPROVEMENT_SAMPLE = 0.005
GATE_MIN_F1 = 0.30


def evaluate_gate(
    pr_auc_xgb: float,
    pr_auc_base: float,
    f1_xgb: float,
    f1_base: float,
    latency_xgb_ms: float,
    sample_mode: bool = False,
) -> dict:
    """
    Run the quality gate and return a dict with `passed` (bool) and `reason` (str).

    Gate logic:
      - XGBoost PR-AUC beats baseline by >= 3 percentage points (1.0% in sample mode)
      - XGBoost F1 drops by no more than 2 percentage points compared to baseline (5.0% in sample mode)
      - XGBoost average sample inference latency <= 50.0 ms
    """
    min_improvement = 0.01 if sample_mode else 0.03
    max_f1_drop = 0.05 if sample_mode else 0.02
    max_latency = 50.0

    reasons = []
    
    # 1. PR-AUC improvement
    improvement = pr_auc_xgb - pr_auc_base
    if improvement < min_improvement:
        reasons.append(
            f"XGBoost PR-AUC {pr_auc_xgb:.4f} improvement over baseline {pr_auc_base:.4f} is {improvement:.4f}, "
            f"which is less than the required {min_improvement:.4f}"
        )
        
    # 2. F1 score drop
    f1_drop = f1_base - f1_xgb
    if f1_drop > max_f1_drop:
        reasons.append(
            f"XGBoost F1 {f1_xgb:.4f} dropped by {f1_drop:.4f} compared to baseline F1 {f1_base:.4f}, "
            f"which exceeds the maximum allowed drop of {max_f1_drop:.4f}"
        )
        
    # 3. Latency check
    if latency_xgb_ms > max_latency:
        reasons.append(
            f"XGBoost average sample inference latency {latency_xgb_ms:.2f} ms "
            f"exceeds limit of {max_latency:.2f} ms"
        )

    passed = len(reasons) == 0
    reason = "All gate criteria met." if passed else " | ".join(reasons)
    return {"passed": passed, "reason": reason}


# ── Plot helpers ─────────────────────────────────────────────────────────────

def plot_roc(y_test, y_prob_base, y_prob_xgb, out_dir: Path) -> Path:
    fpr_b, tpr_b, _ = roc_curve(y_test, y_prob_base)
    fpr_x, tpr_x, _ = roc_curve(y_test, y_prob_xgb)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_b, tpr_b, label=f"Baseline LR (AUC={roc_auc_score(y_test, y_prob_base):.3f})")
    plt.plot(fpr_x, tpr_x, label=f"XGBoost     (AUC={roc_auc_score(y_test, y_prob_xgb):.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend()
    out = out_dir / "roc_curve.png"
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    return out


def plot_pr(y_test, y_prob_base, y_prob_xgb, out_dir: Path) -> Path:
    p_b, r_b, _ = precision_recall_curve(y_test, y_prob_base)
    p_x, r_x, _ = precision_recall_curve(y_test, y_prob_xgb)
    plt.figure(figsize=(8, 6))
    plt.plot(r_b, p_b, label=f"Baseline LR (AUC={auc(r_b, p_b):.3f})")
    plt.plot(r_x, p_x, label=f"XGBoost     (AUC={auc(r_x, p_x):.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve"); plt.legend()
    out = out_dir / "pr_curve.png"
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    return out


def plot_confusion(y_test, y_pred_xgb, out_dir: Path) -> Path:
    cm = confusion_matrix(y_test, y_pred_xgb)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("XGBoost Confusion Matrix")
    out = out_dir / "confusion_matrix.png"
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    return out


def plot_calibration(y_test, y_prob_base, y_prob_xgb, out_dir: Path) -> Path:
    pt_b, pp_b = calibration_curve(y_test, y_prob_base, n_bins=10)
    pt_x, pp_x = calibration_curve(y_test, y_prob_xgb, n_bins=10)
    plt.figure(figsize=(8, 6))
    plt.plot(pp_b, pt_b, "s-", label="Baseline LR")
    plt.plot(pp_x, pt_x, "s-", label="XGBoost")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect")
    plt.xlabel("Mean Predicted Prob"); plt.ylabel("Fraction Positive")
    plt.title("Calibration Curve"); plt.legend()
    out = out_dir / "calibration_curve.png"
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    return out


def generate_all_plots(
    y_test, y_prob_base, y_prob_xgb, y_pred_xgb, out_dir: Path
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "roc": plot_roc(y_test, y_prob_base, y_prob_xgb, out_dir),
        "pr": plot_pr(y_test, y_prob_base, y_prob_xgb, out_dir),
        "cm": plot_confusion(y_test, y_pred_xgb, out_dir),
        "calibration": plot_calibration(y_test, y_prob_base, y_prob_xgb, out_dir),
    }


# ── Summary ──────────────────────────────────────────────────────────────────

def compute_metrics(y_test, y_prob_xgb, y_pred_xgb, y_prob_base) -> dict:
    p_xgb, r_xgb, _ = precision_recall_curve(y_test, y_prob_xgb)
    return {
        "baseline_auc": round(roc_auc_score(y_test, y_prob_base), 4),
        "xgb_auc": round(roc_auc_score(y_test, y_prob_xgb), 4),
        "xgb_pr_auc": round(auc(r_xgb, p_xgb), 4),
        "xgb_f1": round(f1_score(y_test, y_pred_xgb), 4),
        "classification_report": classification_report(y_test, y_pred_xgb, output_dict=True),
    }


def save_metrics(metrics: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ml_metrics.json"
    # classification_report has nested dicts — exclude from top-level JSON for readability
    saveable = {k: v for k, v in metrics.items() if k != "classification_report"}
    saveable["classification_report"] = metrics.get("classification_report", {})
    with open(path, "w") as f:
        json.dump(saveable, f, indent=2)
    return path
