"""Tests for loading and validation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import load_csv, validate_test_data, validate_training_data


def test_load_csv_reads_file(tmp_path):
    path = tmp_path / "small.csv"
    pd.DataFrame({"Id": [1, 2], "SalePrice": [100_000, 120_000]}).to_csv(path, index=False)

    loaded = load_csv(path)

    assert loaded.shape == (2, 2)
    assert loaded["Id"].tolist() == [1, 2]


def test_load_csv_missing_file_explains_where_to_put_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/raw"):
        load_csv(tmp_path / "train.csv")


def test_validate_training_data_accepts_valid_frame(house_frame):
    validate_training_data(house_frame)


def test_validate_training_data_requires_target(house_frame):
    with pytest.raises(ValueError, match="SalePrice"):
        validate_training_data(house_frame.drop(columns=["SalePrice"]))


def test_validate_training_data_rejects_duplicate_ids(house_frame):
    frame = house_frame.copy()
    frame.loc[1, "Id"] = frame.loc[0, "Id"]
    with pytest.raises(ValueError, match="duplicate Id"):
        validate_training_data(frame)


def test_validate_training_data_rejects_non_positive_price(house_frame):
    frame = house_frame.copy()
    frame.loc[0, "SalePrice"] = 0
    with pytest.raises(ValueError, match="positive"):
        validate_training_data(frame)


def test_validate_test_data_requires_id(house_frame):
    with pytest.raises(ValueError, match="'Id'"):
        validate_test_data(house_frame.drop(columns=["Id"]))

    validate_test_data(house_frame.drop(columns=["SalePrice"]))
