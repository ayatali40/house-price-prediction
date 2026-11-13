"""Load the saved pipeline and predict prices for new houses.

Run from the project root:

    python -m src.predict --input data/raw/test.csv --output reports/test_predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_cleaning import clean_data
from src.data_loader import ID_COLUMN, TEST_PATH
from src.feature_engineering import engineer_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"


def load_model(path: str | Path = MODEL_PATH):
    """Load the fitted pipeline, with a nudge toward src.train if it's absent."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {path}\nRun 'python -m src.train' first."
        )
    return joblib.load(path)


def predict(model, df: pd.DataFrame) -> pd.DataFrame:
    """Return Id and predicted price for every row of ``df``.

    Cleaning and feature engineering are repeated here, then the saved
    pipeline takes care of imputation, encoding, scaling and prediction.
    """
    if ID_COLUMN not in df.columns:
        raise ValueError(f"Input data must contain an '{ID_COLUMN}' column.")

    features = engineer_features(clean_data(df))
    predictions = np.expm1(model.predict(features))
    return pd.DataFrame({"Id": df[ID_COLUMN], "PredictedSalePrice": predictions})


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict house sale prices.")
    parser.add_argument("--input", default=str(TEST_PATH), help="CSV file with house features")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "reports" / "test_predictions.csv"),
        help="Where to write predictions",
    )
    args = parser.parse_args()

    model = load_model()
    data = pd.read_csv(args.input)
    predictions = predict(model, data)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)

    print(f"Predicted {len(predictions)} houses from {args.input}")
    print(f"Saved to {args.output}\n")
    print(predictions.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
