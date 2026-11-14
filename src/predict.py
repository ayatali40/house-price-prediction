"""Predict sale prices for new houses.

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
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model not found: {path}\nRun 'python -m src.train' first."
        )
    return joblib.load(path)


def predict(model_artifact, df: pd.DataFrame) -> pd.DataFrame:
    if ID_COLUMN not in df.columns:
        raise ValueError(f"Input must contain an '{ID_COLUMN}' column.")

    # Support both plain pipeline (old) and dict artifact (new).
    if isinstance(model_artifact, dict):
        pipeline = model_artifact["pipeline"]
        medians = model_artifact.get("lot_frontage_medians")
    else:
        pipeline = model_artifact
        medians = None

    features = engineer_features(
        clean_data(df, lot_frontage_medians=medians)
    )
    predictions = np.expm1(pipeline.predict(features))
    return pd.DataFrame({"Id": df[ID_COLUMN], "PredictedSalePrice": predictions})


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict house sale prices.")
    parser.add_argument("--input", default=str(TEST_PATH), help="CSV with house features")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "reports" / "test_predictions.csv"),
        help="Output path",
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
