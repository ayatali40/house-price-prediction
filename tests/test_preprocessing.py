"""Tests for cleaning and preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data_cleaning import clean_data, remove_price_outliers
from src.feature_engineering import engineer_features
from src.preprocessing import make_pipeline


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    return engineer_features(clean_data(frame)).drop(columns=["SalePrice"])


def test_structural_missing_values_become_meaningful_values(house_frame):
    cleaned = clean_data(house_frame)

    assert (cleaned["PoolQC"] == "None").all()
    assert (cleaned["GarageType"] == "None").all()
    assert (cleaned["GarageArea"] == 0).all()
    assert (cleaned["TotalBsmtSF"].fillna(0) >= 0).all()
    assert cleaned["GarageYrBlt"].eq(cleaned["YearBuilt"]).all()
    assert cleaned["LotFrontage"].notna().all()


def test_engineered_features_have_expected_values(house_frame):
    featured = engineer_features(clean_data(house_frame))
    first = featured.iloc[0]

    # Every structural numeric field is absent in the fixture, so the basement
    # contributes 0.
    assert first["TotalSF"] == 0 + 900 + 300
    assert first["TotalBathrooms"] == 2
    assert first["HouseAge"] == 2010 - 1990
    assert first["RemodeledAge"] == 2010 - 2000
    assert first["TotalPorchSF"] == 100 + 20
    assert featured["MSSubClass"].dtype == object


def test_pipeline_imputes_and_encodes_without_nan(house_frame):
    X = _features(house_frame)
    pipeline = make_pipeline(LinearRegression(), X)
    pipeline.fit(X, np.log1p(house_frame["SalePrice"]))

    transformed = pipeline.named_steps["preprocessor"].transform(X)

    assert transformed.shape[0] == len(X)
    assert not np.isnan(transformed).any()


def test_pipeline_handles_unseen_categories(house_frame):
    X = _features(house_frame)
    pipeline = make_pipeline(LinearRegression(), X)
    pipeline.fit(X, np.log1p(house_frame["SalePrice"]))

    unseen = X.copy()
    unseen.loc[0, "MSZoning"] = "BRAND_NEW_ZONE"
    predictions = pipeline.predict(unseen)

    assert np.isfinite(predictions).all()


def test_price_outliers_are_removed_only_with_labelled_data(house_frame):
    frame = house_frame.copy()
    frame.loc[0, ["GrLivArea", "SalePrice"]] = [5_000, 200_000]

    filtered = remove_price_outliers(frame)

    assert len(filtered) == len(frame) - 1
    assert filtered[filtered["Id"] == 1].empty
