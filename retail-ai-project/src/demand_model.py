"""
demand_model.py
---------------
Trains an XGBoost regressor to predict monthly units_sold
based on features from feature_engineering.py.

Pipeline:
  1. Load raw data -> engineer features
  2. Train/test split (last 6 months = test)
  3. Train XGBoost
  4. Evaluate (MAE, RMSE, R²)
  5. Return model + predictions for dashboard use
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ===================================================================
# Feature columns used by the model
# ===================================================================
FEATURE_COLS = [
    "month", "quarter", "year",
    "month_sin", "month_cos",
    "lag_1", "lag_2", "lag_3",
    "rolling_mean_3", "rolling_std_3",
    "population_density", "avg_income", "competitors",
    "foot_traffic", "accessibility_score", "parking_availability",
    "category_encoded",
]

TARGET_COL = "units_sold"


# ===================================================================
# Train / Test split  (time-based, NOT random)
# ===================================================================
def split_data(df: pd.DataFrame, test_months: int = 6):
    """
    Split by time: the last `test_months` months go to test,
    everything before goes to train.

    This is the correct way to split time-series data —
    you never train on future data.
    """
    cutoff_date = df["date"].max() - pd.DateOffset(months=test_months)

    train = df[df["date"] <= cutoff_date].copy()
    test  = df[df["date"] > cutoff_date].copy()

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_test  = test[FEATURE_COLS]
    y_test  = test[TARGET_COL]

    return X_train, X_test, y_train, y_test, train, test


# ===================================================================
# Model training
# ===================================================================
def train_demand_model(X_train, y_train) -> XGBRegressor:
    """
    Train an XGBoost regressor with sensible defaults.

    Key hyperparameters explained:
      n_estimators  : number of boosting rounds (trees)
      max_depth     : how deep each tree can go (prevents overfitting)
      learning_rate : step size — smaller = more cautious learning
      subsample     : use 80% of data per tree (reduces overfitting)
    """
    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,           # suppress training logs
    )
    model.fit(X_train, y_train)
    return model


# ===================================================================
# Evaluation
# ===================================================================
def evaluate_model(model, X_test, y_test) -> dict:
    """Return key regression metrics."""
    y_pred = model.predict(X_test)

    metrics = {
        "MAE":  round(mean_absolute_error(y_test, y_pred), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        "R2":   round(r2_score(y_test, y_pred), 4),
    }
    return metrics, y_pred


def get_feature_importance(model) -> pd.DataFrame:
    """Return feature importances sorted descending."""
    imp = pd.DataFrame({
        "Feature":    FEATURE_COLS,
        "Importance": model.feature_importances_,
    })
    imp = imp.sort_values("Importance", ascending=False).reset_index(drop=True)
    imp["Importance"] = imp["Importance"].round(4)
    return imp


# ===================================================================
# Convenience: full pipeline in one call (used by dashboard)
# ===================================================================
def run_demand_pipeline(demand_df: pd.DataFrame):
    """
    End-to-end: split -> train -> evaluate -> return everything.

    Returns
    -------
    model       : trained XGBRegressor
    metrics     : dict with MAE, RMSE, R2
    test_df     : test set DataFrame with a 'predicted' column added
    importance  : feature importance DataFrame
    """
    X_train, X_test, y_train, y_test, train_df, test_df = split_data(demand_df)

    model = train_demand_model(X_train, y_train)
    metrics, y_pred = evaluate_model(model, X_test, y_test)

    test_df = test_df.copy()
    test_df["predicted"] = y_pred.round(0).astype(int)

    importance = get_feature_importance(model)

    return model, metrics, test_df, importance


# ===================================================================
# Quick-run
# ===================================================================
if __name__ == "__main__":
    from data_processing import generate_locations, generate_sales
    from feature_engineering import engineer_demand_features

    # Build data
    locations = generate_locations()
    sales = generate_sales(locations)
    demand_df = engineer_demand_features(sales, locations)

    # Run pipeline
    model, metrics, test_df, importance = run_demand_pipeline(demand_df)

    # --- Metrics ---
    print("=" * 50)
    print("  DEMAND MODEL EVALUATION")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:>5s}: {v}")

    # --- Feature importance (top 10) ---
    print("\n--- Top 10 Feature Importances ---")
    print(importance.head(10).to_string(index=False))

    # --- Sample predictions ---
    print("\n--- Sample Predictions vs Actuals ---")
    sample_cols = ["location_id", "date", "product_category",
                   "units_sold", "predicted"]
    print(test_df[sample_cols].head(12).to_string(index=False))

    # --- Overall stats ---
    print(f"\nTrain size: {len(demand_df) - len(test_df)} rows")
    print(f"Test size : {len(test_df)} rows")
