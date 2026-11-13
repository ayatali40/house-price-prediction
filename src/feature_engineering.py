"""Extra features for the House Prices data.

Only five are added. Each one is something you'd know when the house is
listed, so no future information sneaks in.
"""

from __future__ import annotations

import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the five combined features.

    - TotalSF: basement plus both floors. Total usable space is what buyers
      compare, and one column captures it better than three.
    - TotalBathrooms: full and half baths, above and below ground, with a half
      counting as 0.5.
    - HouseAge: years between construction and sale.
    - RemodeledAge: years since the last remodel (or since construction).
    - TotalPorchSF: all porch and deck areas added up.
    """
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
    """MSSubClass is a code, not a measurement, so keep it as a category."""
    df = df.copy()
    if "MSSubClass" in df.columns:
        df["MSSubClass"] = df["MSSubClass"].astype(str)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature-engineering step."""
    return cast_categorical_codes(add_engineered_features(df))
