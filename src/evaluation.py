"""Metrics and diagnostic plots."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "images"
FIGURE_DPI = 130


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def predict_prices(model, features: pd.DataFrame) -> np.ndarray:
    return np.expm1(model.predict(features))


def plot_actual_vs_predicted(
    y_true,
    y_pred,
    path: str | Path = IMAGES_DIR / "eval_actual_vs_predicted.png",
    title: str = "Actual vs Predicted Sale Price",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.45, edgecolor="white", linewidth=0.3)
    limits = [0, max(y_true.max(), y_pred.max()) * 1.02]
    ax.plot(limits, limits, color="crimson", linestyle="--", label="Perfect prediction")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Actual sale price ($)")
    ax.set_ylabel("Predicted sale price ($)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


def plot_residual_distribution(
    y_true,
    y_pred,
    path: str | Path = IMAGES_DIR / "eval_residual_distribution.png",
    title: str = "Distribution of Residuals",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=40, color="#4C72B0", edgecolor="white")
    ax.axvline(0, color="crimson", linestyle="--", label="Zero error")
    ax.set_xlabel("Residual ($)")
    ax.set_ylabel("Number of houses")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


def plot_residuals_vs_predicted(
    y_true,
    y_pred,
    path: str | Path = IMAGES_DIR / "eval_residuals_vs_predicted.png",
    title: str = "Residuals vs Predicted Price",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = np.asarray(y_true, dtype=float) - y_pred

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.45, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="crimson", linestyle="--")
    ax.set_xlabel("Predicted sale price ($)")
    ax.set_ylabel("Residual ($)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


def plot_model_comparison(
    results: pd.DataFrame,
    path: str | Path = IMAGES_DIR / "model_comparison.png",
    title: str = "Training vs Validation RMSE by Model",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    positions = np.arange(len(results))
    width = 0.38
    ax.bar(positions - width / 2, results["train_rmse"], width, label="Training RMSE")
    ax.bar(positions + width / 2, results["val_rmse"], width, label="Validation RMSE")
    ax.set_xticks(positions)
    ax.set_xticklabels(results["model"], rotation=20, ha="right")
    ax.set_ylabel("RMSE ($)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path
