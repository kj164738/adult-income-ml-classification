from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
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
    "hidden_layer_sizes": [(128, 64), (256, 128)],
    "alpha": [1e-4, 1e-3],
    "learning_rate_init": [1e-3, 5e-4],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MLP baseline for Adult income classification.")
    parser.add_argument("--csv-path", type=str, default=None, help="Optional local CSV path for the Adult dataset.")
    parser.add_argument("--smoke-test", action="store_true", help="Use a small synthetic dataset for offline validation.")
    return parser.parse_args()


def build_pipeline(x_train: pd.DataFrame, **model_params: object) -> Pipeline:
    model = MLPClassifier(
        random_state=RANDOM_STATE,
        max_iter=100,
        early_stopping=True,
        solver="adam",
        **model_params,
    )
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x_train, scale_numeric=True)),
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


def decode_predictions(label_encoder: LabelEncoder, values: pd.Series | list[int] | object) -> pd.Series:
    decoded = label_encoder.inverse_transform(pd.Series(values).astype(int))
    return pd.Series(decoded)


def plot_training_curve(pipeline: Pipeline, output_path: Path) -> None:
    model: MLPClassifier = pipeline.named_steps["model"]
    loss_frame = pd.DataFrame({"epoch": range(1, len(model.loss_curve_) + 1), "loss": model.loss_curve_})
    plt.figure(figsize=(6.5, 4.0))
    sns.lineplot(data=loss_frame, x="epoch", y="loss", marker="o", linewidth=1.5, color="#1f77b4")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("MLP training curve")
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
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    positive_class_index = int(label_encoder.transform([POSITIVE_LABEL])[0])

    search_rows: list[dict[str, object]] = []
    best_score: tuple[float, float] | None = None
    best_result: dict[str, object] | None = None

    keys = list(PARAM_GRID.keys())
    for values in itertools.product(*PARAM_GRID.values()):
        params = dict(zip(keys, values))
        pipeline = build_pipeline(x_train, **params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            train_time_seconds, memory_delta_mb = measure_fit(lambda: pipeline.fit(x_train, y_train_encoded))
        y_val_pred_encoded = pipeline.predict(x_val)
        y_val_pred = decode_predictions(label_encoder, y_val_pred_encoded)
        y_val_score = pipeline.predict_proba(x_val)[:, positive_class_index]
        val_metrics = compute_classification_metrics(y_val, y_val_pred, y_val_score)
        row = {
            "model": "MLP",
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
        raise RuntimeError("MLP search did not produce a valid model.")

    best_pipeline: Pipeline = best_result["pipeline"]  # type: ignore[assignment]
    model_path = MODELS_DIR / "mlp_classifier.joblib"
    joblib.dump(best_pipeline, model_path)
    model_size_mb = model_path.stat().st_size / (1024 ** 2)

    y_test_pred_encoded = best_pipeline.predict(x_test)
    y_test_pred = decode_predictions(label_encoder, y_test_pred_encoded)
    y_test_score = best_pipeline.predict_proba(x_test)[:, positive_class_index]
    test_metrics = compute_classification_metrics(y_test, y_test_pred, y_test_score)
    avg_inference_seconds = measure_inference(lambda: best_pipeline.predict_proba(x_test))
    inference_ms_per_sample = avg_inference_seconds * 1000 / len(x_test)

    save_dataframe(TABLES_DIR / "mlp_search_results.csv", pd.DataFrame(search_rows))
    save_prediction_file(
        METRICS_DIR / "mlp_validation_predictions.csv",
        y_val,
        y_val_pred,
        best_pipeline.predict_proba(x_val)[:, positive_class_index],
    )
    save_prediction_file(METRICS_DIR / "mlp_test_predictions.csv", y_test, y_test_pred, y_test_score)
    plot_training_curve(best_pipeline, FIGURES_DIR / "mlp_training_curve.png")

    summary_payload = {
        "model_name": "MLP",
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
    save_json(METRICS_DIR / "mlp_summary.json", summary_payload)
    print(f"Saved MLP model to {model_path}")
    print(f"Best params: {best_result['params']}")
    print(f"Validation F1: {best_result['validation_metrics']['f1']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
