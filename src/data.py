from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, make_classification
from sklearn.model_selection import train_test_split

from utils import DATA_DIR, POSITIVE_LABEL, RANDOM_STATE


ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]
LOCAL_CACHE_CSV = DATA_DIR / "adult_openml_cached.csv"


def load_cached_adult_csv(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(csv_path)
    frame["income"] = frame["income"].astype(str).str.replace(".", "", regex=False).str.strip()
    features = frame.drop(columns=["income"])
    target = frame["income"]
    return features, target


def load_local_adult_csv(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(csv_path, header=None, names=ADULT_COLUMNS, na_values="?", skipinitialspace=True)
    frame = frame.dropna(how="all")
    frame["income"] = frame["income"].astype(str).str.replace(".", "", regex=False).str.strip()
    features = frame.drop(columns=["income"])
    target = frame["income"]
    return features, target


def load_openml_adult() -> tuple[pd.DataFrame, pd.Series]:
    if LOCAL_CACHE_CSV.exists():
        return load_cached_adult_csv(LOCAL_CACHE_CSV)
    dataset = fetch_openml(name="adult", version=2, as_frame=True, data_home=str(DATA_DIR / "openml"))
    features = dataset.data.copy()
    target = dataset.target.astype(str).str.replace(".", "", regex=False).str.strip()
    cached_frame = features.copy()
    cached_frame["income"] = target
    LOCAL_CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
    cached_frame.to_csv(LOCAL_CACHE_CSV, index=False)
    return features, target


def make_smoke_test_dataset(n_samples: int = 600) -> tuple[pd.DataFrame, pd.Series]:
    numeric, labels = make_classification(
        n_samples=n_samples,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        class_sep=1.2,
        random_state=RANDOM_STATE,
    )
    rng = np.random.default_rng(RANDOM_STATE)
    frame = pd.DataFrame(numeric, columns=["age", "education-num", "capital-gain", "capital-loss", "hours-per-week", "fnlwgt"])
    frame["workclass"] = np.where(labels == 1, "Private", rng.choice(["Government", "Self-emp"], size=n_samples))
    frame["education"] = np.where(frame["education-num"] > frame["education-num"].median(), "Bachelors", "HS-grad")
    frame["marital-status"] = np.where(labels == 1, "Married", "Single")
    frame["occupation"] = rng.choice(["Tech", "Sales", "Admin"], size=n_samples)
    frame["relationship"] = np.where(labels == 1, "Husband", "Not-in-family")
    frame["race"] = rng.choice(["White", "Black", "Asian-Pac-Islander"], size=n_samples)
    frame["sex"] = rng.choice(["Male", "Female"], size=n_samples)
    frame["native-country"] = rng.choice(["United-States", "Mexico", "India"], size=n_samples)
    frame.loc[frame.index[::17], "workclass"] = "?"
    frame.loc[frame.index[::19], "occupation"] = "?"
    frame.loc[frame.index[::23], "capital-gain"] = np.nan
    target = pd.Series(np.where(labels == 1, POSITIVE_LABEL, "<=50K"), name="income")
    return frame, target


def normalize_missing_values(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    object_like = cleaned.select_dtypes(include=["object", "string", "category"]).columns
    for column in object_like:
        cleaned[column] = cleaned[column].astype(str).str.strip()
        cleaned[column] = cleaned[column].replace({"?": np.nan, "nan": np.nan})
    return cleaned


def load_dataset(smoke_test: bool = False, csv_path: str | None = None) -> tuple[pd.DataFrame, pd.Series, str]:
    if smoke_test:
        features, target = make_smoke_test_dataset()
        source = "synthetic_smoke_test"
    elif csv_path:
        features, target = load_local_adult_csv(Path(csv_path))
        source = str(Path(csv_path).resolve())
    else:
        features, target = load_openml_adult()
        source = "openml:adult-v2"
    features = normalize_missing_values(features)
    return features, target, source


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    random_state: int = RANDOM_STATE,
) -> dict[str, pd.DataFrame | pd.Series]:
    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        target,
        test_size=0.30,
        stratify=target,
        random_state=random_state,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state,
    )
    return {
        "x_train": x_train.reset_index(drop=True),
        "x_val": x_val.reset_index(drop=True),
        "x_test": x_test.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_val": y_val.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
    }


def dataset_summary(features: pd.DataFrame, target: pd.Series) -> dict[str, int]:
    numeric_columns = features.select_dtypes(include=["number"]).columns
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    positive_count = int((target == POSITIVE_LABEL).sum())
    return {
        "num_rows": int(len(features)),
        "num_features": int(features.shape[1]),
        "num_numeric_features": int(len(numeric_columns)),
        "num_categorical_features": int(len(categorical_columns)),
        "positive_examples": positive_count,
        "negative_examples": int(len(target) - positive_count),
    }
