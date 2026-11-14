"""Tests for metrics, plots and prediction."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from src.data_cleaning import clean_data, compute_lot_frontage_medians, fill_structural_missing
from src.evaluation import (
    plot_actual_vs_predicted,
    plot_model_comparison,
    regression_metrics,
)
from src.feature_engineering import engineer_features
from src.predict import predict
from src.preprocessing import make_pipeline


def test_metrics_are_zero_for_perfect_predictions():
    metrics = regression_metrics([100.0, 200.0], [100.0, 200.0])

    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["mse"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


def test_metrics_on_a_small_known_example():
    metrics = regression_metrics([100.0, 200.0], [110.0, 190.0])

    assert metrics["mae"] == pytest.approx(10.0)
    assert metrics["mse"] == pytest.approx(100.0)
    assert metrics["rmse"] == pytest.approx(10.0)
    assert metrics["r2"] == pytest.approx(0.96)


def test_prediction_pipeline_returns_prices_for_each_house(house_frame):
    structural = fill_structural_missing(house_frame)
    medians = compute_lot_frontage_medians(structural)
    featured = engineer_features(clean_data(house_frame, lot_frontage_medians=medians))
    X = featured.drop(columns=["SalePrice"])
    pipeline = make_pipeline(LinearRegression(), X)
    pipeline.fit(X, np.log1p(featured["SalePrice"]))

    artifact = {"pipeline": pipeline, "lot_frontage_medians": medians}
    predictions = predict(artifact, house_frame)

    assert list(predictions.columns) == ["Id", "PredictedSalePrice"]
    assert len(predictions) == len(house_frame)
    assert np.isfinite(predictions["PredictedSalePrice"]).all()
    assert (predictions["PredictedSalePrice"] > 0).all()


def test_prediction_requires_id_column(house_frame):
    with pytest.raises(ValueError, match="'Id'"):
        predict(model_artifact=None, df=house_frame.drop(columns=["Id"]))


def test_evaluation_plots_are_saved(tmp_path):
    import pandas as pd

    y_true = np.array([100_000, 150_000, 200_000])
    y_pred = np.array([110_000, 145_000, 205_000])

    scatter = plot_actual_vs_predicted(y_true, y_pred, path=tmp_path / "scatter.png")
    comparison = plot_model_comparison(
        pd.DataFrame(
            {
                "model": ["A", "B"],
                "train_rmse": [10_000, 12_000],
                "val_rmse": [15_000, 14_000],
            }
        ),
        path=tmp_path / "comparison.png",
    )

    assert scatter.is_file() and scatter.stat().st_size > 0
    assert comparison.is_file() and comparison.stat().st_size > 0
