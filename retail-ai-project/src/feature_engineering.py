"""
feature_engineering.py
----------------------
Creates derived features for two purposes:
  1. Location suitability scoring  (location-level)
  2. Demand prediction with XGBoost (time-series level)

All transformations are pure functions — they take a DataFrame in
and return a new DataFrame out, making them easy to test and explain.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# ===================================================================
# 1. LOCATION-LEVEL FEATURES  (used by location_model.py)
# ===================================================================

def engineer_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the raw locations DataFrame and adds:

      competition_score  : Lower is better  -> inverted & scaled 0-1
      affordability      : income_level / rent -> higher means customers can spend more
      market_potential    : population * income_level (proxy for total spending power)
      access_parking     : combined accessibility + parking score
      value_score        : foot_traffic / rent  -> bang for buck

    Finally, all numeric features are Min-Max scaled to [0, 1] so they
    can be combined fairly in a weighted scoring model.
    """
    out = df.copy()

    # --- Derived features ---
    out["competition_score"] = 1 / (1 + out["competitors"])        # fewer competitors = higher score
    out["affordability"]     = out["avg_income"] / out["rent_per_sqft"] # income relative to rent
    out["market_potential"]  = out["population_density"] * out["avg_income"]    # total spending power proxy
    out["access_parking"]   = out["accessibility_score"] + out["parking_availability"]    # combined convenience
    out["value_score"]      = out["foot_traffic"] / out["rent_per_sqft"] # traffic per rupee of rent

    # --- Normalize key features to 0-1 range ---
    features_to_scale = [
        "population_density", "avg_income", "competitors",
        "accessibility_score", "parking_availability", "foot_traffic", "rent_per_sqft",
        "competition_score", "affordability", "market_potential",
        "access_parking", "value_score",
    ]

    scaler = MinMaxScaler()
    out[features_to_scale] = scaler.fit_transform(out[features_to_scale])

    return out


# ===================================================================
# 2. DEMAND / TIME-SERIES FEATURES  (used by demand_model.py)
# ===================================================================

def engineer_demand_features(sales_df: pd.DataFrame,
                              locations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw sales data and enriches it with:

      Time features:
        - month, quarter, year
        - month_sin / month_cos   (cyclical encoding so Jan is close to Dec)

      Lag features (per location x category):
        - lag_1, lag_2, lag_3      (sales 1/2/3 months ago)

      Rolling features:
        - rolling_mean_3           (3-month moving average)
        - rolling_std_3            (3-month volatility)

      Location context:
        - population, income_level, competitors, foot_traffic
          (merged from locations table)

    Rows with NaN (from lags) are dropped so the model gets clean input.
    """
    df = sales_df.copy()

    # --- Time features ---
    df["month"]   = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"]    = df["date"].dt.year

    # Cyclical encoding — lets the model know month 12 is close to month 1
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # --- Lag features (per location + category group) ---
    group_cols = ["location_id", "product_category"]
    df = df.sort_values(["location_id", "product_category", "date"])

    for lag in [1, 2, 3]:
        df[f"lag_{lag}"] = df.groupby(group_cols)["units_sold"].shift(lag)

    # --- Rolling statistics ---
    df["rolling_mean_3"] = (
        df.groupby(group_cols)["units_sold"]
          .transform(lambda x: x.rolling(3).mean())
    )
    df["rolling_std_3"] = (
        df.groupby(group_cols)["units_sold"]
          .transform(lambda x: x.rolling(3).std())
    )

    # --- Merge location context ---
    loc_features = locations_df[
        ["location_id", "population_density", "avg_income",
         "competitors", "foot_traffic", "accessibility_score", "parking_availability"]
    ]
    df = df.merge(loc_features, on="location_id", how="left")

    # --- Encode product category as numeric ---
    df["category_encoded"] = df["product_category"].astype("category").cat.codes

    # --- Drop rows with NaN from lag/rolling (first 3 months per group) ---
    df = df.dropna().reset_index(drop=True)

    return df


# ===================================================================
# Quick-run: preview the engineered features
# ===================================================================
if __name__ == "__main__":
    from data_processing import generate_locations, generate_sales

    # Generate raw data
    locations = generate_locations()
    sales = generate_sales(locations)

    # --- Location features ---
    loc_feat = engineer_location_features(locations)
    print("=== LOCATION FEATURES (first 5) ===\n")
    display_cols = [
        "location_name", "competition_score", "affordability",
        "market_potential", "access_parking", "value_score",
    ]
    print(loc_feat[display_cols].head(5).to_string(index=False))

    # --- Demand features ---
    demand_feat = engineer_demand_features(sales, locations)
    print("\n\n=== DEMAND FEATURES (first 8) ===\n")
    display_cols2 = [
        "location_id", "date", "product_category", "units_sold",
        "month", "month_sin", "lag_1", "rolling_mean_3",
    ]
    print(demand_feat[display_cols2].head(8).to_string(index=False))

    print(f"\nLocation features shape : {loc_feat.shape}")
    print(f"Demand features shape   : {demand_feat.shape}")
    print(f"Demand feature columns  : {list(demand_feat.columns)}")
