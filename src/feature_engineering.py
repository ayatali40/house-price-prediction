"""Feature engineering for House Prices."""

from __future__ import annotations

import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    df["TotalBathrooms"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["BsmtHalfBath"]
    )

    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodeledAge"] = df["YrSold"] - df["YearRemodAdd"]

    df["TotalPorchSF"] = (
        df["WoodDeckSF"]
        + df["OpenPorchSF"]
        + df["EnclosedPorch"]
        + df["3SsnPorch"]
        + df["ScreenPorch"]
    )

    return df


def cast_categorical_codes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "MSSubClass" in df.columns:
        df["MSSubClass"] = df["MSSubClass"].astype(str)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    return cast_categorical_codes(add_engineered_features(df))
