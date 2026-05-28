# JC3509 Assessment: Adult Income Classification

This repository contains the completed JC3509 Machine Learning assessment project for comparing **Random Forest** and **Multi-Layer Perceptron (MLP)** on the **Adult Income** binary classification task.

## Group Members

- Group member A: Kang Jinjia
- Group member B: Li Yongdong

## Project Summary

The project predicts whether an individual's annual income is greater than `$50K` using the public Adult dataset. The comparison uses the same preprocessing pipeline, train/validation/test split, and evaluation protocol for both models.

The final experiment compares:

- Random Forest
- Multi-Layer Perceptron (MLP)

The submitted paper reports predictive metrics, confusion-matrix behaviour, ROC curves, feature importance, training time, inference speed, memory usage, and model size.

## Project Structure

```text
JC3509_Assessment/
  data/
    adult_openml_cached.csv
  outputs/
    figures/
    metrics/
    models/
    tables/
  paper/
    final_single_file.tex
    final_single_file_regenerated.tex
    ML4_corrected.tex
    cvpr.sty
    refs.bib
    ieeenat_fullname.bst
  src/
    data.py
    preprocess.py
    train_rf.py
    train_mlp.py
    evaluate.py
    utils.py
  README.md
  requirements.txt
```

## Environment Setup

Create a Python environment and install the required packages:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The experiments were run with Python 3.12.7 on Windows 11.

## Reproducing the Experiments

Run the full experiment from the project root:

```powershell
python src/train_rf.py
python src/train_mlp.py
python src/evaluate.py
```

The scripts use the cached Adult dataset at:

```text
data/adult_openml_cached.csv
```

If the cache is not present, the code can fetch the Adult dataset from OpenML and save a local copy.

## Outputs

Running the scripts produces the following artifacts:

- `outputs/models/random_forest.joblib`
- `outputs/models/mlp_classifier.joblib`
- `outputs/metrics/rf_summary.json`
- `outputs/metrics/mlp_summary.json`
- `outputs/metrics/final_summary.json`
- `outputs/tables/main_results_table.csv`
- `outputs/tables/complexity_table.csv`
- `outputs/figures/roc_curve.pdf`
- `outputs/figures/rf_feature_importance.pdf`
- `outputs/figures/mlp_training_curve.pdf`

The paper uses the generated tables and figures to support the final comparison.

## Evaluation Protocol

- Dataset split: 70% training, 15% validation, 15% test
- Random seed: 42
- Model selection metric: validation F1-score
- Final test metrics: Accuracy, Precision, Recall, F1-score, and ROC-AUC
- Efficiency metrics: training time, inference time per sample, memory growth, and serialized model size

Both models use the same data split and evaluation pipeline to ensure a fair comparison.

## Paper

The final paper is written in the CVPR-style format required by the assessment. The main single-file version is:

```text
paper/final_single_file_regenerated.tex
```

The equivalent corrected version is also provided as:

```text
paper/ML4_corrected.tex
```

The final PDF should be compiled from one of these files in Overleaf or another LaTeX environment.

## Notes

- The code is designed to be reproducible from a clean Python environment.
- The cached dataset is included to avoid relying on network access during marking.
- The generated outputs are included so the reported paper results can be checked directly.
