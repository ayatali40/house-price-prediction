"""Cleaning rules for the House Prices data.

The dataset uses NaN for two very different things:

1. The house really doesn't have the feature (no garage, no basement, no pool).
2. Someone didn't record the value (mostly LotFrontage).

Those need different treatment, so the rules are split below. The lists are
module constants so the training and prediction paths use exactly the same
columns.
"""

from __future__ import annotations

import pandas as pd

# A NaN here means the house doesn't have the thing, so make that explicit.
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

# Same idea for amounts: no garage means zero area and zero cars.
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

# No garage year usually means the garage went up with the house.
GARAGE_YEAR_COLUMN = "GarageYrBlt"
BUILD_YEAR_COLUMN = "YearBuilt"


def fill_structural_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Turn "feature absent" NaNs into explicit "None" or 0 values."""
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


def impute_lot_frontage(df: pd.DataFrame) -> pd.DataFrame:
    """Fill LotFrontage with the median of the same neighborhood.

    Frontage follows the street layout, so a local median beats one global
    number.
    """
    df = df.copy()
    if "LotFrontage" not in df.columns or "Neighborhood" not in df.columns:
        return df
    neighborhood_median = df.groupby("Neighborhood")["LotFrontage"].transform("median")
    df["LotFrontage"] = df["LotFrontage"].fillna(neighborhood_median)
    df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())
    return df


def remove_price_outliers(
    df: pd.DataFrame,
    area_threshold: float = 4000,
    price_threshold: float = 300_000,
) -> pd.DataFrame:
    """Drop the famous Ames oddities: huge houses that sold cheap.

    Two training rows have more than 4,000 sq ft of living area but went for
    under $300k, way below what houses that size normally fetch. Keeping them
    would drag the regression line toward a pattern that doesn't generalise.
    The filter needs SalePrice, so it only works on labelled data.
    """
    if {"GrLivArea", "SalePrice"}.issubset(df.columns):
        mask = (df["GrLivArea"] > area_threshold) & (df["SalePrice"] < price_threshold)
        df = df.loc[~mask].copy()
    return df


def clean_data(df: pd.DataFrame, remove_outliers: bool = False) -> pd.DataFrame:
    """Run every cleaning step and return a new DataFrame.

    Parameters
    ----------
    df:
        Raw data (training or competition test).
    remove_outliers:
        Only makes sense for labelled training data.
    """
    df = fill_structural_missing(df)
    df = impute_lot_frontage(df)
    if remove_outliers:
        df = remove_price_outliers(df)
    return df
