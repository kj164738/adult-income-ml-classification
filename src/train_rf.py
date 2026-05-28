from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from data import dataset_summary, load_dataset, split_dataset
from preprocess import build_preprocessor
from utils import (
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    POSITIVE_LABEL,
    RANDOM_STATE,
    TABLES_DIR,
    compute_classification_metrics,
    ensure_project_dirs,
    measure_fit,
    measure_inference,
    save_dataframe,
    save_json,
)


PARAM_GRID = {
    "n_estimators": [200, 500],
    "max_depth": [None, 20, 40],
    "min_samples_leaf": [1, 5],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Random Forest baseline for Adult income classification.")
    parser.add_argument("--csv-path", type=str, default=None, help="Optional local CSV path for the Adult dataset.")
    parser.add_argument("--smoke-test", action="store_true", help="Use a small synthetic dataset for offline validation.")
    return parser.parse_args()


def build_pipeline(x_train: pd.DataFrame, **model_params: int | None) -> Pipeline:
    model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **model_params)
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x_train, scale_numeric=False)),
            ("model", model),
        ]
    )


def save_prediction_file(path: Path, y_true: pd.Series, y_pred: pd.Series, y_score: pd.Series) -> None:
    save_dataframe(
        path,
        pd.DataFrame(
            {
                "y_true": y_true,
                "y_pred": y_pred,
                "y_score": y_score,
            }
        ),
    )


def plot_feature_importance(pipeline: Pipeline, output_path: Path, top_k: int = 15) -> None:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importance_frame = (
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(top_k)
        .sort_values("importance", ascending=True)
    )
    plt.figure(figsize=(7.0, 4.5))
    sns.barplot(data=importance_frame, x="importance", y="feature", hue="feature", dodge=False, palette="crest", legend=False)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title("Random Forest top feature importances")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    if output_path.suffix.lower() != ".pdf":
        plt.savefig(output_path.with_suffix(".pdf"))
    plt.close()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    features, target, data_source = load_dataset(smoke_test=args.smoke_test, csv_path=args.csv_path)
    split = split_dataset(features, target)
    x_train = split["x_train"]
    x_val = split["x_val"]
    x_test = split["x_test"]
    y_train = split["y_train"]
    y_val = split["y_val"]
    y_test = split["y_test"]

    search_rows: list[dict[str, float | int | str | None]] = []
    best_score: tuple[float, float] | None = None
    best_result: dict[str, object] | None = None

    keys = list(PARAM_GRID.keys())
    for values in itertools.product(*PARAM_GRID.values()):
        params = dict(zip(keys, values))
        pipeline = build_pipeline(x_train, **params)
        train_time_seconds, memory_delta_mb = measure_fit(lambda: pipeline.fit(x_train, y_train))
        y_val_pred = pipeline.predict(x_val)
        y_val_score = pipeline.predict_proba(x_val)[:, 1]
        val_metrics = compute_classification_metrics(y_val, y_val_pred, y_val_score)
        row = {
            "model": "Random Forest",
            **params,
            **val_metrics,
            "train_time_seconds": train_time_seconds,
            "memory_delta_mb": memory_delta_mb,
        }
        search_rows.append(row)
        candidate_score = (val_metrics["f1"], val_metrics["roc_auc"])
        if best_score is None or candidate_score > best_score:
            best_score = candidate_score
            best_result = {
                "params": params,
                "pipeline": pipeline,
                "validation_metrics": val_metrics,
                "train_time_seconds": train_time_seconds,
                "memory_delta_mb": memory_delta_mb,
            }

    if best_result is None:
        raise RuntimeError("Random Forest search did not produce a valid model.")

    best_pipeline: Pipeline = best_result["pipeline"]  # type: ignore[assignment]
    model_path = MODELS_DIR / "random_forest.joblib"
    joblib.dump(best_pipeline, model_path)
    model_size_mb = model_path.stat().st_size / (1024 ** 2)

    y_test_pred = best_pipeline.predict(x_test)
    y_test_score = best_pipeline.predict_proba(x_test)[:, 1]
    test_metrics = compute_classification_metrics(y_test, y_test_pred, y_test_score)
    avg_inference_seconds = measure_inference(lambda: best_pipeline.predict_proba(x_test))
    inference_ms_per_sample = avg_inference_seconds * 1000 / len(x_test)

    save_dataframe(TABLES_DIR / "rf_search_results.csv", pd.DataFrame(search_rows))
    save_prediction_file(METRICS_DIR / "rf_validation_predictions.csv", y_val, best_pipeline.predict(x_val), best_pipeline.predict_proba(x_val)[:, 1])
    save_prediction_file(METRICS_DIR / "rf_test_predictions.csv", y_test, y_test_pred, y_test_score)
    plot_feature_importance(best_pipeline, FIGURES_DIR / "rf_feature_importance.png")

    summary_payload = {
        "model_name": "Random Forest",
        "data_source": data_source,
        "smoke_test": args.smoke_test,
        "target_label_positive": POSITIVE_LABEL,
        "dataset_summary": dataset_summary(features, target),
        "best_params": best_result["params"],
        "selection_metric": "validation_f1",
        "validation_metrics": best_result["validation_metrics"],
        "test_metrics": test_metrics,
        "complexity": {
            "train_time_seconds": best_result["train_time_seconds"],
            "memory_delta_mb": best_result["memory_delta_mb"],
            "avg_inference_seconds": avg_inference_seconds,
            "inference_ms_per_sample": inference_ms_per_sample,
            "model_size_mb": model_size_mb,
        },
    }
    save_json(METRICS_DIR / "rf_summary.json", summary_payload)
    print(f"Saved Random Forest model to {model_path}")
    print(f"Best params: {best_result['params']}")
    print(f"Validation F1: {best_result['validation_metrics']['f1']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
