"""Train and save house price models.

    python -m src.train
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor

from src.data_cleaning import (
    clean_data,
    compute_lot_frontage_medians,
    fill_structural_missing,
)
from src.data_loader import TARGET_COLUMN, load_raw_data, validate_training_data
from src.evaluation import (
    predict_prices,
    plot_actual_vs_predicted,
    plot_model_comparison,
    plot_residual_distribution,
    plot_residuals_vs_predicted,
    regression_metrics,
)
from src.feature_engineering import engineer_features
from src.preprocessing import make_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = MODELS_DIR / "best_model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def get_models() -> dict[str, tuple[object, bool]]:
    return {
        "Linear Regression": (LinearRegression(), True),
        "Ridge Regression": (Ridge(alpha=10.0), True),
        "Lasso Regression": (Lasso(alpha=0.0005, max_iter=50_000), True),
        "Decision Tree": (
            DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=RANDOM_STATE),
            False,
        ),
        "Random Forest": (
            RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
            False,
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                random_state=RANDOM_STATE,
            ),
            False,
        ),
    }


def prepare_training_data(
    train_df: pd.DataFrame,
    remove_outliers: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Clean, engineer features, return X, y, and lot-frontage medians."""
    validate_training_data(train_df)

    # Structural fill first, then learn neighborhood medians on training rows only.
    structural = fill_structural_missing(train_df)
    lot_frontage_medians = compute_lot_frontage_medians(structural)

    cleaned = clean_data(
        train_df,
        remove_outliers=remove_outliers,
        lot_frontage_medians=lot_frontage_medians,
    )
    featured = engineer_features(cleaned)
    X = featured.drop(columns=[TARGET_COLUMN])
    y = featured[TARGET_COLUMN]
    return X, y, lot_frontage_medians


def prepare_test_features(
    test_df: pd.DataFrame,
    lot_frontage_medians: pd.Series,
) -> pd.DataFrame:
    return engineer_features(
        clean_data(test_df, lot_frontage_medians=lot_frontage_medians)
    )


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cv: int = 5,
) -> tuple[pd.DataFrame, dict[str, object]]:
    log_y_train = np.log1p(y_train)
    rows = []
    fitted = {}

    for name, (estimator, scale_numeric) in get_models().items():
        pipeline = make_pipeline(estimator, X_train, scale_numeric=scale_numeric)
        pipeline.fit(X_train, log_y_train)

        train_metrics = regression_metrics(y_train, predict_prices(pipeline, X_train))
        val_metrics = regression_metrics(y_val, predict_prices(pipeline, X_val))

        cv_pipeline = make_pipeline(
            clone(estimator), X_train, scale_numeric=scale_numeric
        )
        cv_scores = cross_val_score(
            cv_pipeline,
            X_train,
            log_y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
        )

        rows.append(
            {
                "model": name,
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
                "cv_rmse_log": float(-cv_scores.mean()),
            }
        )
        fitted[name] = pipeline

    return pd.DataFrame(rows), fitted


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    X_train: pd.DataFrame,
    lot_frontage_medians: pd.Series,
) -> dict:
    estimator, scale_numeric = get_models()[model_name]
    pipeline = make_pipeline(clone(estimator), X_train, scale_numeric=scale_numeric)
    pipeline.fit(X, np.log1p(y))

    artifact = {
        "pipeline": pipeline,
        "lot_frontage_medians": lot_frontage_medians,
        "model_name": model_name,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    return artifact


def _select_best(results: pd.DataFrame) -> str:
    return str(results.sort_values(["val_rmse", "cv_rmse_log"]).iloc[0]["model"])


def main() -> None:
    train_df, test_df = load_raw_data()
    X, y, lot_frontage_medians = prepare_training_data(train_df, remove_outliers=True)
    X_train, X_val, y_train, y_val = split_data(X, y)

    results, fitted = compare_models(X_train, y_train, X_val, y_val)
    results = results.sort_values("val_rmse").reset_index(drop=True)

    pd.set_option("display.float_format", "{:,.2f}".format)
    print(f"Training rows: {len(X_train)} | Validation rows: {len(X_val)}")
    print("\nModel comparison (metrics in dollars, lower RMSE is better):")
    print(
        results[["model", "train_rmse", "val_rmse", "val_mae", "val_r2", "cv_rmse_log"]]
        .to_string(index=False)
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    plot_model_comparison(results)

    best_name = _select_best(results)
    best_row = results.loc[results["model"] == best_name].iloc[0]
    best_val_predictions = predict_prices(fitted[best_name], X_val)
    plot_actual_vs_predicted(
        y_val,
        best_val_predictions,
        title=f"Actual vs Predicted — {best_name} (validation)",
    )
    plot_residual_distribution(
        y_val,
        best_val_predictions,
        title=f"Residual Distribution — {best_name} (validation)",
    )
    plot_residuals_vs_predicted(
        y_val,
        best_val_predictions,
        title=f"Residuals vs Predicted — {best_name} (validation)",
    )

    final_artifact = train_final_model(X, y, best_name, X_train, lot_frontage_medians)
    final_pipeline = final_artifact["pipeline"]

    test_features = prepare_test_features(test_df, lot_frontage_medians)
    test_predictions = predict_prices(final_pipeline, test_features)
    submission = pd.DataFrame({"Id": test_df["Id"], "PredictedSalePrice": test_predictions})
    submission.to_csv(REPORTS_DIR / "test_predictions.csv", index=False)

    summary = {
        "training_rows": int(len(X)),
        "train_split_rows": int(len(X_train)),
        "validation_split_rows": int(len(X_val)),
        "feature_columns": int(X.shape[1]),
        "target_transform": "log1p",
        "selection_metric": "validation RMSE (dollars)",
        "best_model": best_name,
        "best_validation_metrics": {
            "mae": float(best_row["val_mae"]),
            "rmse": float(best_row["val_rmse"]),
            "r2": float(best_row["val_r2"]),
            "cv_rmse_log": float(best_row["cv_rmse_log"]),
        },
        "all_models": results.to_dict(orient="records"),
    }
    with open(REPORTS_DIR / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nBest model by validation RMSE: {best_name}")
    print(f"Validation MAE: ${best_row['val_mae']:,.0f}")
    print(f"Validation RMSE: ${best_row['val_rmse']:,.0f}")
    print(f"Validation R^2: {best_row['val_r2']:.3f}")
    print(f"Saved pipeline to {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved test predictions to {(REPORTS_DIR / 'test_predictions.csv').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
