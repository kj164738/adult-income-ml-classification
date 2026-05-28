from __future__ import annotations

import inspect

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def infer_feature_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [column for column in features.columns if column not in numeric_features]
    return numeric_features, categorical_features


def build_one_hot_encoder() -> OneHotEncoder:
    signature = inspect.signature(OneHotEncoder)
    if "sparse_output" in signature.parameters:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(features: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric_features, categorical_features = infer_feature_types(features)
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", build_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
