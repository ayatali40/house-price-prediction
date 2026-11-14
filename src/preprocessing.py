"""Preprocessing pipelines fitted on the training split only."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import ID_COLUMN, TARGET_COLUMN

NON_FEATURE_COLUMNS = [ID_COLUMN, TARGET_COLUMN]


def split_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    features = df.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in df.columns])
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric, categorical


def make_pipeline(model, feature_frame: pd.DataFrame, scale_numeric: bool = True) -> Pipeline:
    numeric, categorical = split_feature_types(feature_frame)

    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline([("preprocessor", preprocessor), ("model", model)])
