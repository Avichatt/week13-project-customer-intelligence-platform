"""
predict.py — Inference layer for the XGBoost churn / campaign-conversion model.

Loads the persisted model package (preprocessor + model + metadata) that
train.py saves to data/processed/best_model.joblib and exposes a clean
`predict_single` and `predict_batch` API consumed by the FastAPI ML router.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from src import config


# ── Model loader (singleton) ─────────────────────────────────────────────────

_model_package: dict | None = None


def load_model_package(path: Path | None = None) -> dict:
    """
    Load (and cache) the model package dict:
      {
        "preprocessor": BankFeaturePipeline,
        "model":        XGBClassifier,
        "metrics":      {"roc_auc": ..., "pr_auc": ..., "f1": ...},
        "run_id":       str,
      }
    """
    global _model_package
    if _model_package is not None:
        return _model_package

    model_path = path or (config.PROCESSED_DATA_DIR / "best_model.joblib")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model package not found at {model_path}. "
            "Run `python pipelines/run_ml_pipeline.py` first."
        )
    _model_package = joblib.load(model_path)
    return _model_package


def reload_model(path: Path | None = None) -> dict:
    """Force a reload (useful after retraining without restarting the server)."""
    global _model_package
    _model_package = None
    return load_model_package(path)


# ── Prediction helpers ───────────────────────────────────────────────────────

def _raw_to_df(raw: dict[str, Any]) -> pd.DataFrame:
    """Convert a single raw feature dict to a one-row DataFrame."""
    return pd.DataFrame([raw])


def predict_single(features: dict[str, Any]) -> dict:
    """
    Run inference for one customer feature dict.

    Parameters
    ----------
    features : dict
        Raw input matching the bank-marketing schema (age, job, marital, …).

    Returns
    -------
    dict with keys:
      prediction     : int   (0 = no-conversion, 1 = conversion)
      probability    : float (model's positive-class probability)
      model_run_id   : str   (MLflow run ID of the serving model)
      model_metrics  : dict  (roc_auc, pr_auc, f1 of the serving model)
    """
    pkg = load_model_package()
    preprocessor = pkg["preprocessor"]
    model = pkg["model"]

    df = _raw_to_df(features)
    X_proc = preprocessor.transform(df)
    prob = float(model.predict_proba(X_proc)[0, 1])
    pred = int(prob >= 0.5)

    return {
        "prediction": pred,
        "probability": round(prob, 4),
        "model_run_id": pkg.get("run_id", "unknown"),
        "model_metrics": pkg.get("metrics", {}),
    }


def predict_batch(records: list[dict[str, Any]]) -> list[dict]:
    """
    Run batch inference.

    Parameters
    ----------
    records : list of feature dicts

    Returns
    -------
    list of prediction dicts (same structure as predict_single)
    """
    if not records:
        return []

    pkg = load_model_package()
    preprocessor = pkg["preprocessor"]
    model = pkg["model"]

    df = pd.DataFrame(records)
    X_proc = preprocessor.transform(df)
    probs = model.predict_proba(X_proc)[:, 1]
    preds = (probs >= 0.5).astype(int)

    run_id = pkg.get("run_id", "unknown")
    metrics = pkg.get("metrics", {})

    return [
        {
            "prediction": int(p),
            "probability": round(float(pr), 4),
            "model_run_id": run_id,
            "model_metrics": metrics,
        }
        for p, pr in zip(preds, probs)
    ]


def get_model_info() -> dict:
    """Return metadata about the currently-loaded model (no inference)."""
    pkg = load_model_package()
    return {
        "run_id": pkg.get("run_id", "unknown"),
        "metrics": pkg.get("metrics", {}),
        "model_type": type(pkg.get("model")).__name__,
        "preprocessor_type": type(pkg.get("preprocessor")).__name__,
    }
