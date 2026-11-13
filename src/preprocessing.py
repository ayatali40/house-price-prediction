"""Scikit-learn preprocessing.

Imputation, encoding and scaling all live inside the pipeline, so they are
fitted on the training split and nowhere else.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import ID_COLUMN, TARGET_COLUMN

NON_FEATURE_COLUMNS = [ID_COLUMN, TARGET_COLUMN]


def split_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split the feature columns into numeric and categorical lists."""
    features = df.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in df.columns])
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric, categorical


def make_pipeline(model, feature_frame: pd.DataFrame, scale_numeric: bool = True) -> Pipeline:
    """Build the preprocessing + model pipeline for a training frame.

    Numbers get median imputation, which shrugs off outliers. Categories get
    mode imputation and one-hot encoding; unseen levels are ignored rather
    than raising an error. Scaling is on for linear models and off for trees,
    which don't care about scale.

    The column lists come from ``feature_frame`` (the training split), so each
    fitted pipeline remembers exactly what it was built for.
    """
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
