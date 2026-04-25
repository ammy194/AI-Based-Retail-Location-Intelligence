"""
data_processing.py
------------------
Handles synthetic dataset creation, loading, and cleaning.

Generates:
  1. locations.csv  — Realistic Karnataka regions (majority Bengaluru)
  2. sales.csv      — 36 months of historical sales per location
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

KARNATAKA_LOCATIONS = [
    # Bengaluru (25 locations)
    ("Bengaluru", "Indiranagar", 12.9784, 77.6408),
    ("Bengaluru", "Koramangala", 12.9352, 77.6245),
    ("Bengaluru", "Whitefield", 12.9698, 77.7500),
    ("Bengaluru", "Electronic City", 12.8399, 77.6770),
    ("Bengaluru", "BTM Layout", 12.9166, 77.6101),
    ("Bengaluru", "Jayanagar", 12.9299, 77.5826),
    ("Bengaluru", "JP Nagar", 12.9063, 77.5857),
    ("Bengaluru", "Yelahanka", 13.1007, 77.5963),
    ("Bengaluru", "Hebbal", 13.0354, 77.5988),
    ("Bengaluru", "Marathahalli", 12.9569, 77.7011),
    ("Bengaluru", "Sarjapur Road", 12.9226, 77.6734),
    ("Bengaluru", "HSR Layout", 12.9121, 77.6446),
    ("Bengaluru", "Bannerghatta Road", 12.8878, 77.5971),
    ("Bengaluru", "Malleshwaram", 13.0031, 77.5643),
    ("Bengaluru", "Rajajinagar", 12.9982, 77.5530),
    ("Bengaluru", "KR Puram", 13.0083, 77.7031),
    ("Bengaluru", "Banashankari", 12.9175, 77.5562),
    ("Bengaluru", "Basavanagudi", 12.9406, 77.5684),
    ("Bengaluru", "Bellandur", 12.9304, 77.6784),
    ("Bengaluru", "Ulsoor", 12.9815, 77.6226),
    ("Bengaluru", "CV Raman Nagar", 12.9855, 77.6639),
    ("Bengaluru", "Domlur", 12.9609, 77.6387),
    ("Bengaluru", "Kalyan Nagar", 13.0280, 77.6399),
    ("Bengaluru", "RT Nagar", 13.0247, 77.5948),
    ("Bengaluru", "Vijayanagar", 12.9756, 77.5354),
    
    # Other Cities (8 locations)
    ("Mysuru", "Gokulam", 12.3259, 76.6331),
    ("Mysuru", "Kuvempunagar", 12.2858, 76.6264),
    ("Mangaluru", "Kadri", 12.8824, 74.8560),
    ("Mangaluru", "Hampankatta", 12.8704, 74.8436),
    ("Hubballi-Dharwad", "Vidya Nagar", 15.3647, 75.1240),
    ("Hubballi-Dharwad", "Navanagar", 15.3948, 75.1130),
    ("Belagavi", "Tilakwadi", 15.8366, 74.5065),
    ("Belagavi", "Camp", 15.8562, 74.5312),
]

PREMIUM_AREAS = [
    "Indiranagar", "Koramangala", "Whitefield", "Marathahalli", "Domlur", 
    "HSR Layout", "Malleshwaram", "Gokulam", "Kadri"
]
OUTSKIRT_AREAS = [
    "Electronic City", "Yelahanka", "KR Puram", "Navanagar", "Camp"
]
# All others are considered Residential

PRODUCT_CATEGORIES = ["Electronics Store", "Clothing Store", "Grocery Store", "Café"]
SEED = 42

def generate_locations(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    n = len(KARNATAKA_LOCATIONS)
    
    rows = []
    for idx, (city, area, lat, lon) in enumerate(KARNATAKA_LOCATIONS):
        # Realistic Rule-Based Generation
        if area in PREMIUM_AREAS:
            pop_den = rng.randint(8000, 15000)
            avg_inc = rng.randint(80000, 150000)
            foot_trf = rng.randint(100, 200)
            rent = rng.randint(150, 300)
            comp = rng.randint(8, 15)
            access = rng.randint(7, 10)
            parking = rng.randint(1, 4) # Hard to find parking in premium
        elif area in OUTSKIRT_AREAS:
            pop_den = rng.randint(2000, 8000)
            avg_inc = rng.randint(30000, 60000)
            foot_trf = rng.randint(20, 60)
            rent = rng.randint(30, 80)
            comp = rng.randint(1, 5)
            access = rng.randint(4, 7)
            parking = rng.randint(4, 6) # Easy parking
        else: # Residential
            pop_den = rng.randint(10000, 20000) # High density
            avg_inc = rng.randint(50000, 90000)
            foot_trf = rng.randint(50, 120)
            rent = rng.randint(70, 150)
            comp = rng.randint(4, 10)
            access = rng.randint(6, 9)
            parking = rng.randint(2, 5)

        rows.append({
            "location_id":          idx + 1,
            "city":                 city,
            "area":                 area,
            "location_name":        f"{area}, {city}",
            "latitude":             lat,
            "longitude":            lon,
            "population_density":   pop_den,
            "avg_income":           avg_inc,
            "competitors":          comp,
            "accessibility_score":  access,
            "parking_availability": parking,
            "foot_traffic":         foot_trf,
            "rent_per_sqft":        rent,
        })

    return pd.DataFrame(rows)

def generate_sales(locations_df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.date_range(start="2023-01-01", periods=36, freq="MS")

    rows = []
    for _, loc in locations_df.iterrows():
        # Base Sales logic based on formula: population_density + avg_income + foot_traffic - competition + noise
        base_demand_factor = (loc["population_density"] / 1000) + (loc["avg_income"] / 2000) + (loc["foot_traffic"] * 0.5) - (loc["competitors"] * 2)
        base_demand_factor = max(base_demand_factor, 10) # Floor
        
        for cat in PRODUCT_CATEGORIES:
            cat_mult = {
                "Electronics Store": 0.3, # Sells fewer units, higher price
                "Clothing Store": 0.8,
                "Grocery Store": 2.5,     # Sells many units
                "Café": 1.5
            }[cat]
            
            for i, date in enumerate(dates):
                trend    = 1 + 0.005 * i
                season   = 1 + 0.15 * np.sin(2 * np.pi * i / 12)
                noise    = rng.normal(1.0, 0.10)
                
                units    = int(base_demand_factor * cat_mult * trend * season * noise)
                units    = max(units, 0)
                
                # Realistic pricing
                price_multiplier = {
                    "Electronics Store": rng.uniform(2000, 10000),
                    "Clothing Store": rng.uniform(500, 2500),
                    "Grocery Store": rng.uniform(50, 300),
                    "Café": rng.uniform(150, 400)
                }[cat]
                
                revenue  = round(units * price_multiplier, 2)

                rows.append({
                    "location_id":      loc["location_id"],
                    "date":             date,
                    "product_category": cat,
                    "units_sold":       units,
                    "revenue":          revenue,
                })

    return pd.DataFrame(rows)

def save_datasets(locations_df: pd.DataFrame, sales_df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    locations_df.to_csv(DATA_DIR / "locations.csv", index=False)
    sales_df.to_csv(DATA_DIR / "sales.csv", index=False)
    print(f"[OK] Saved locations.csv  ({len(locations_df)} rows)")
    print(f"[OK] Saved sales.csv      ({len(sales_df)} rows)")

def load_locations() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "locations.csv")

def load_sales() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["date"])

if __name__ == "__main__":
    print(">> Generating realistic synthetic datasets...\n")
    locations = generate_locations()
    sales = generate_sales(locations)
    save_datasets(locations, sales)

    print("\n--- Locations Preview ---")
    print(locations[["city", "area", "population_density", "avg_income", "competitors", "accessibility_score", "foot_traffic"]].head(8).to_string(index=False))

    print("\n--- Sales Preview ---")
    print(sales.head(5).to_string(index=False))

    print(f"\nTotal locations : {len(locations)}")
    print(f"Total sales rows: {len(sales)}")
