# House Price Prediction

A regression project built on the Kaggle **House Prices: Advanced Regression Techniques** dataset. It takes the raw competition files through data checks, exploration, cleaning, feature engineering and model comparison, then saves the pipeline with the best validation score.

This started as a learning project, and it's organised that way: small reusable modules, a notebook that explains each decision, tests, and a saved model you can actually call.

## The problem

Given a house's size, quality, age, neighbourhood, garage, basement and the rest of its details, estimate its sale price.

A useful model has to be accurate in dollars, handle houses it hasn't seen, and produce the same result on every run. It estimates prices; it doesn't explain them. When a feature looks important here, that means it's associated with price in this data, not that changing it will move the price by a fixed amount.

## The data

Kaggle competition: [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

- `train.csv`: 1,460 rows × 81 columns, includes `SalePrice`
- `test.csv`: 1,459 rows × 80 columns, no target

The CSVs aren't in this repository. Download them and put them here:

```
data/raw/train.csv
data/raw/test.csv
```

The raw files stay untouched. Every change happens on a copy in memory.

## Tools

Python 3.10 · pandas · NumPy · scikit-learn · Matplotlib · seaborn · joblib · Jupyter · pytest

## Layout

```
house-price-prediction/
├── data/
│   ├── raw/                      # train.csv and test.csv (git-ignored)
│   └── processed/                # nothing here yet
├── notebooks/
│   └── house_price_prediction.ipynb
├── src/
│   ├── data_loader.py            # load the CSVs, check the basics
│   ├── data_cleaning.py          # missing values, LotFrontage, outliers
│   ├── feature_engineering.py    # five extra features
│   ├── preprocessing.py          # imputation, encoding, scaling
│   ├── train.py                  # compare models, refit, save
│   ├── evaluation.py             # metrics and plots
│   └── predict.py                # reusable prediction function
├── models/                       # best_model.joblib (generated)
├── images/                       # EDA and evaluation charts
├── reports/                      # metrics, comparison table, predictions
├── tests/                        # 17 pytest tests
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## How it works

1. Load the CSVs and fail loudly if either one is missing.
2. Look at the shapes, types, distributions, category counts and the target.
3. Check missing values, duplicates, odd values and outliers.
4. Clean using what each field means, not blanket mean or mode filling.
5. Add five features that a buyer would know at listing time.
6. Split the data 80/20 once, before fitting anything.
7. Train six regressors on the same split, features and target.
8. Evaluate with MAE, MSE, RMSE, R² and a few residual plots.
9. Pick the model with the best validation RMSE, refit it on all labelled rows and save it with joblib.
10. Predict the competition test set and sanity-check a few real rows.

## Decisions worth knowing

**Missing values.** A NaN in `PoolQC`, `Alley`, `Fence` or the garage and basement quality columns usually means the house doesn't have that feature. Those become an explicit `"None"` category, and the matching area and count columns become 0. `LotFrontage` is different: it's missing for 259 rows, and frontage follows the street layout, so it's filled with the median of the same neighbourhood. A single missing `Electrical` value gets the most common category.

**Ties to the house.** A missing `GarageYrBlt` becomes `YearBuilt`, since the garage normally went up with the house.

**Outliers.** Two houses have more than 4,000 sq ft of living area but sold below $300k. They look like partial or unusual sales and would pull the fit in the wrong direction, so they're removed from training only. Nothing is removed from the test set.

**Log target.** `SalePrice` has skewness 1.88; `log1p` brings it down to 0.12. Models train on the log scale, and predictions are converted back with `expm1` before any dollar metric is calculated. It's a choice that was checked on validation data, not an assumption.

**Extra features.** `TotalSF` (basement plus both floors), `TotalBathrooms` (half baths count 0.5), `HouseAge`, `RemodeledAge` and `TotalPorchSF`. All are known when the house is listed.

**Leakage.** Imputation, one-hot encoding and scaling sit inside a scikit-learn `Pipeline` fitted on the training split. Unseen categories are ignored rather than causing an error. `SalePrice` never goes in as a feature.

## Results

Validation split: 292 houses. Metrics are in dollars, and lower MAE/RMSE is better.

| Model | Train RMSE | Val MAE | Val RMSE | Val R² | CV RMSE (log) |
|---|---:|---:|---:|---:|---:|
| **Lasso Regression** | 17,528 | **13,881** | **19,364** | **0.932** | 0.114 |
| Ridge Regression | 16,774 | 14,326 | 20,032 | 0.927 | 0.114 |
| Gradient Boosting | 8,353 | 14,523 | 20,241 | 0.926 | 0.125 |
| Linear Regression | 15,462 | 15,529 | 22,207 | 0.911 | 0.130 |
| Random Forest | 10,207 | 16,433 | 23,655 | 0.899 | 0.137 |
| Decision Tree | 19,971 | 24,495 | 40,448 | 0.704 | 0.176 |

**Why Lasso.** It has the lowest validation RMSE, with Ridge close behind. Both regularised linear models beat the tree ensembles. The single decision tree is the clearest overfitter: its training error looks reasonable next to the linear models, but its validation error is far worse. Gradient Boosting also fits the training data very tightly (RMSE $8.4k) without matching that on validation.

The final pipeline was refit on all 1,458 labelled rows and saved to `models/best_model.joblib`. Predictions for the 1,459 test rows are in `reports/test_predictions.csv` (mean about $179k, median about $156k). There's **no test accuracy number** here, because those labels aren't public.

## Running it

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash; use .venv\Scripts\activate for cmd
pip install -r requirements.txt

python -m src.train                  # compare, select, refit, save (about 30 s)
python -m src.predict                # writes reports/test_predictions.csv
python -m pytest tests -q            # 17 tests
jupyter notebook notebooks/house_price_prediction.ipynb
```

`src.train` writes the comparison table, a JSON summary, the model file and the evaluation plots. The notebook was run top to bottom on this dataset and finished with no errors.

## Tests

The 17 tests cover loading and validation, structural missing values, the engineered feature values, imputation and encoding, unseen categories, outlier filtering, metric calculations on known examples, plot creation, and the prediction function. They use small synthetic fixtures; none of that data is used for reported results.

## What this project covers

- Turning a business question into a regression workflow.
- Checking data quality and handling missing values based on what they mean.
- EDA with charts that have titles, labels and a takeaway.
- Leak-free preprocessing with `Pipeline` and `ColumnTransformer`.
- Comparing linear, regularised and tree-based models on one split.
- Spotting overfitting with training/validation gaps, cross-validation and residuals.
- Saving a model and using it for new predictions.
- Testing the pieces with pytest.

## Limitations

- 1,458 labelled rows for 84 predictors. Validation numbers carry real uncertainty.
- One city, one period: Ames, Iowa, 2006–2010. Don't assume these relationships travel.
- Test labels are private, so validation is the only honest accuracy evidence.
- The chosen model is additive. It can't represent complicated interactions.
- Errors grow with price: average absolute error is about $10.8k in the cheapest validation quartile and $21.8k in the priciest.
- A linear model can extrapolate. One test house is predicted at about $1.73M, above anything in training, so extreme estimates deserve a second look.
- Associations aren't causal.

## Credits and license

Dataset © Kaggle / Dean De Cock, used under the competition rules. The code is released under the MIT License. Please don't redistribute the raw competition CSVs.
