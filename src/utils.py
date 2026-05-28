from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"
METRICS_DIR = OUTPUTS_DIR / "metrics"
RANDOM_STATE = 42
POSITIVE_LABEL = ">50K"


def ensure_project_dirs() -> None:
    for path in (DATA_DIR, OUTPUTS_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR, METRICS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def measure_fit(fit_callable: Callable[[], None]) -> tuple[float, float]:
    process = psutil.Process(os.getpid())
    start_rss = process.memory_info().rss
    start = time.perf_counter()
    fit_callable()
    elapsed = time.perf_counter() - start
    end_rss = process.memory_info().rss
    rss_delta_mb = max(0.0, end_rss - start_rss) / (1024 ** 2)
    return elapsed, rss_delta_mb


def measure_inference(predict_callable: Callable[[], Any], repeats: int = 20) -> float:
    predict_callable()
    start = time.perf_counter()
    for _ in range(repeats):
        predict_callable()
    elapsed = time.perf_counter() - start
    return elapsed / repeats


def compute_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
) -> dict[str, float]:
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    y_score_array = np.asarray(y_score)
    binary_true = (y_true_array == POSITIVE_LABEL).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision": float(precision_score(y_true_array, y_pred_array, pos_label=POSITIVE_LABEL, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred_array, pos_label=POSITIVE_LABEL, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred_array, pos_label=POSITIVE_LABEL, zero_division=0)),
        "roc_auc": float(roc_auc_score(binary_true, y_score_array)),
    }
