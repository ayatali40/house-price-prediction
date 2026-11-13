"""Small synthetic fixtures for the tests.

They are only here to exercise the code, never to stand in for the real
competition data or to produce reported results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import STRUCTURAL_CATEGORICAL_NA, STRUCTURAL_NUMERIC_NA


@pytest.fixture
def house_frame() -> pd.DataFrame:
    """Eight tiny houses covering the fields the pipeline touches."""
    rows = []
    for i in range(8):
        row = {
            "Id": i + 1,
            "MSSubClass": 20 if i % 2 == 0 else 60,
            "MSZoning": ["RL", "RM", "FV"][i % 3],
            "Neighborhood": ["NAmes", "CollgCr"][i % 2],
            "LotFrontage": np.nan if i == 0 else 60.0 + i,
            "LotArea": 8000 + i * 100,
            "OverallQual": 5 + i % 4,
            "GrLivArea": 1200 + i * 200,
            "YearBuilt": 1990 + i,
            "YearRemodAdd": 2000 + i,
            "YrSold": 2010,
            "1stFlrSF": 900 + i * 50,
            "2ndFlrSF": 300,
            "TotalBsmtSF": np.nan if i == 1 else 800,
            "FullBath": 2,
            "HalfBath": 0,
            "BsmtFullBath": 1,
            "BsmtHalfBath": 0,
            "WoodDeckSF": 100,
            "OpenPorchSF": 20,
            "EnclosedPorch": 0,
            "3SsnPorch": 0,
            "ScreenPorch": 0,
            "GarageYrBlt": np.nan,
            "SalePrice": 150_000 + i * 20_000,
        }
        for column in STRUCTURAL_CATEGORICAL_NA:
            row[column] = np.nan
        for column in STRUCTURAL_NUMERIC_NA:
            row[column] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)
