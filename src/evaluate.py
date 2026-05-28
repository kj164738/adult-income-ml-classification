from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

from utils import FIGURES_DIR, METRICS_DIR, TABLES_DIR, ensure_project_dirs, load_json, save_json


def load_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_results_frames(rf_summary: dict[str, object], mlp_summary: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results_rows = []
    complexity_rows = []
    for summary in (rf_summary, mlp_summary):
        test_metrics = summary["test_metrics"]
        complexity = summary["complexity"]
        results_rows.append(
            {
                "Model": summary["model_name"],
                "Accuracy": test_metrics["accuracy"],
                "Precision": test_metrics["precision"],
                "Recall": test_metrics["recall"],
                "F1": test_metrics["f1"],
                "ROC-AUC": test_metrics["roc_auc"],
            }
        )
        complexity_rows.append(
            {
                "Model": summary["model_name"],
                "Train time (s)": complexity["train_time_seconds"],
                "Inference (ms/sample)": complexity["inference_ms_per_sample"],
                "Memory delta (MB)": complexity["memory_delta_mb"],
                "Model size (MB)": complexity["model_size_mb"],
            }
        )
    return pd.DataFrame(results_rows), pd.DataFrame(complexity_rows)


def save_markdown_table(path: Path, frame: pd.DataFrame, floatfmt: str = ".4f") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(frame.to_markdown(index=False, floatfmt=floatfmt))
        handle.write("\n")


def save_latex_tables(results_frame: pd.DataFrame, complexity_frame: pd.DataFrame) -> None:
    results_tex = results_frame.to_latex(
        index=False,
        float_format=lambda value: f"{value:.4f}",
        caption="Test-set classification metrics for the two compared models.",
        label="tab:main_results",
    )
    complexity_tex = complexity_frame.to_latex(
        index=False,
        float_format=lambda value: f"{value:.4f}",
        caption="Complexity and efficiency comparison on the held-out test set.",
        label="tab:complexity_results",
    )
    (TABLES_DIR / "main_results_table.tex").write_text(results_tex, encoding="utf-8")
    (TABLES_DIR / "complexity_table.tex").write_text(complexity_tex, encoding="utf-8")


def plot_roc_curves(rf_predictions: pd.DataFrame, mlp_predictions: pd.DataFrame) -> None:
    plt.figure(figsize=(6.2, 4.5))
    for label, frame, color in (
        ("Random Forest", rf_predictions, "#1b9e77"),
        ("MLP", mlp_predictions, "#d95f02"),
    ):
        y_true = (frame["y_true"] == ">50K").astype(int)
        fpr, tpr, _ = roc_curve(y_true, frame["y_score"])
        plt.plot(fpr, tpr, linewidth=2.0, label=label, color=color)
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves on the test set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=300)
    plt.savefig(FIGURES_DIR / "roc_curve.pdf")
    plt.close()


def main() -> None:
    ensure_project_dirs()
    rf_summary = load_json(METRICS_DIR / "rf_summary.json")
    mlp_summary = load_json(METRICS_DIR / "mlp_summary.json")
    rf_predictions = load_predictions(METRICS_DIR / "rf_test_predictions.csv")
    mlp_predictions = load_predictions(METRICS_DIR / "mlp_test_predictions.csv")

    results_frame, complexity_frame = build_results_frames(rf_summary, mlp_summary)
    results_frame.to_csv(TABLES_DIR / "main_results_table.csv", index=False)
    complexity_frame.to_csv(TABLES_DIR / "complexity_table.csv", index=False)
    save_markdown_table(TABLES_DIR / "main_results_table.md", results_frame)
    save_markdown_table(TABLES_DIR / "complexity_table.md", complexity_frame)
    save_latex_tables(results_frame, complexity_frame)
    plot_roc_curves(rf_predictions, mlp_predictions)

    winner_row = results_frame.sort_values(by=["F1", "ROC-AUC"], ascending=False).iloc[0]
    summary_payload = {
        "best_model_by_test_f1": winner_row["Model"],
        "results_table_path": str(TABLES_DIR / "main_results_table.csv"),
        "complexity_table_path": str(TABLES_DIR / "complexity_table.csv"),
        "roc_curve_path": str(FIGURES_DIR / "roc_curve.pdf"),
    }
    save_json(METRICS_DIR / "final_summary.json", summary_payload)
    print("Saved final evaluation artifacts.")


if __name__ == "__main__":
    main()
