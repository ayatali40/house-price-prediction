"""Load the raw Kaggle House Prices files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_PATH = RAW_DIR / "train.csv"
TEST_PATH = RAW_DIR / "test.csv"

TARGET_COLUMN = "SalePrice"
ID_COLUMN = "Id"


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Download train.csv and test.csv from "
            "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data "
            "and place them in data/raw/."
        )
    return pd.read_csv(path)


def load_raw_data(
    train_path: str | Path = TRAIN_PATH,
    test_path: str | Path = TEST_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_csv(train_path), load_csv(test_path)


def validate_training_data(df: pd.DataFrame) -> None:
    required = {ID_COLUMN, TARGET_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Training data missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Training data is empty.")
    if df[ID_COLUMN].duplicated().any():
        raise ValueError("Training data has duplicate Id values.")
    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Training data has missing SalePrice values.")
    if not pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
        raise ValueError("SalePrice must be numeric.")
    if (df[TARGET_COLUMN] <= 0).any():
        raise ValueError("SalePrice must be positive.")


def validate_test_data(df: pd.DataFrame) -> None:
    if ID_COLUMN not in df.columns:
        raise ValueError(f"Test data missing '{ID_COLUMN}' column.")
    if df.empty:
        raise ValueError("Test data is empty.")
