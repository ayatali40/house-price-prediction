"""Cleaning rules for the House Prices data."""

from __future__ import annotations

import pandas as pd

STRUCTURAL_CATEGORICAL_NA = [
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "MasVnrType",
]

STRUCTURAL_NUMERIC_NA = [
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "GarageCars",
    "GarageArea",
]

GARAGE_YEAR_COLUMN = "GarageYrBlt"
BUILD_YEAR_COLUMN = "YearBuilt"


def fill_structural_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in STRUCTURAL_CATEGORICAL_NA:
        if column in df.columns:
            df[column] = df[column].fillna("None")
    for column in STRUCTURAL_NUMERIC_NA:
        if column in df.columns:
            df[column] = df[column].fillna(0)
    if GARAGE_YEAR_COLUMN in df.columns and BUILD_YEAR_COLUMN in df.columns:
        df[GARAGE_YEAR_COLUMN] = df[GARAGE_YEAR_COLUMN].fillna(df[BUILD_YEAR_COLUMN])
    return df


def compute_lot_frontage_medians(df: pd.DataFrame) -> pd.Series:
    """Neighborhood medians for LotFrontage, learned from training data only."""
    if "LotFrontage" not in df.columns or "Neighborhood" not in df.columns:
        return pd.Series(dtype=float)
    return df.groupby("Neighborhood")["LotFrontage"].median()


def impute_lot_frontage(
    df: pd.DataFrame,
    neighborhood_medians: pd.Series | None = None,
) -> pd.DataFrame:
    """Fill LotFrontage using neighborhood medians.

    Pass medians learned on the training set when scoring new data.
    If medians is None, they are computed from ``df`` (training path only).
    """
    df = df.copy()
    if "LotFrontage" not in df.columns or "Neighborhood" not in df.columns:
        return df

    if neighborhood_medians is None:
        neighborhood_medians = compute_lot_frontage_medians(df)

    mapped = df["Neighborhood"].map(neighborhood_medians)
    df["LotFrontage"] = df["LotFrontage"].fillna(mapped)
    global_median = (
        float(neighborhood_medians.median())
        if len(neighborhood_medians)
        else float(df["LotFrontage"].median())
    )
    if pd.isna(global_median):
        global_median = 0.0
    df["LotFrontage"] = df["LotFrontage"].fillna(global_median)
    return df


def remove_price_outliers(
    df: pd.DataFrame,
    area_threshold: float = 4000,
    price_threshold: float = 300_000,
) -> pd.DataFrame:
    """Drop large houses that sold unusually cheap (training only)."""
    if {"GrLivArea", "SalePrice"}.issubset(df.columns):
        mask = (df["GrLivArea"] > area_threshold) & (df["SalePrice"] < price_threshold)
        df = df.loc[~mask].copy()
    return df


def clean_data(
    df: pd.DataFrame,
    remove_outliers: bool = False,
    lot_frontage_medians: pd.Series | None = None,
) -> pd.DataFrame:
    """Run cleaning steps. Pass train medians when preparing test/new data."""
    df = fill_structural_missing(df)
    df = impute_lot_frontage(df, neighborhood_medians=lot_frontage_medians)
    if remove_outliers:
        df = remove_price_outliers(df)
    return df
