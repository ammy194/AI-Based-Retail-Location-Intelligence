"""
location_model.py
-----------------
Scores and ranks candidate retail store locations using a
Weighted Scoring Model (WSM).

Why WSM instead of ML?
  - We have no labeled data (no "good" / "bad" location labels).
  - A weighted score is fully transparent and easy to explain in a viva.
  - Weights can be tuned by domain experts or stakeholders based on business type.

The model uses the normalized features produced by feature_engineering.py.
"""

import pandas as pd

# ===================================================================
# Dynamic Weights based on Business Type
# ===================================================================
def get_business_weights(business_type: str = "General") -> dict:
    """Returns dynamic weights based on business type."""
    if business_type == "Clothing Store":
        return {
            "market_potential":  0.30,   # Needs high income area
            "competition_score": 0.15,
            "affordability":     0.10,
            "foot_traffic":      0.30,   # Very reliant on walk-ins
            "access_parking":    0.05,
            "value_score":       0.10,
        }
    elif business_type == "Electronics Store":
        return {
            "market_potential":  0.35,   # Needs high purchasing power
            "competition_score": 0.15,
            "affordability":     0.10,
            "foot_traffic":      0.10,
            "access_parking":    0.20,   # Needs good access/parking for large items
            "value_score":       0.10,
        }
    elif business_type == "Grocery Store":
        return {
            "market_potential":  0.30,   # Relies heavily on high population density
            "competition_score": 0.20,   # Grocery stores suffer from nearby competition
            "affordability":     0.10,
            "foot_traffic":      0.10,
            "access_parking":    0.20,   # Needs convenience
            "value_score":       0.10,
        }
    elif business_type == "Café":
        return {
            "market_potential":  0.15,
            "competition_score": 0.15,
            "affordability":     0.10,
            "foot_traffic":      0.40,   # Heavily reliant on foot traffic
            "access_parking":    0.10,
            "value_score":       0.10,
        }
    else: # General
        return {
            "market_potential":  0.25,
            "competition_score": 0.20,
            "affordability":     0.15,
            "foot_traffic":      0.15,
            "access_parking":    0.15,
            "value_score":       0.10,
        }

def score_locations(loc_features_df: pd.DataFrame,
                    business_type: str = "General") -> pd.DataFrame:
    """
    Compute a suitability score for each location based on business type.
    """
    weights = get_business_weights(business_type)
    df = loc_features_df.copy()

    # Weighted sum of normalized features
    df["suitability_score"] = sum(
        df[feature] * weight
        for feature, weight in weights.items()
    )

    # Rank (1 = best)
    df["rank"] = df["suitability_score"].rank(ascending=False).astype(int)
    df = df.sort_values("rank").reset_index(drop=True)

    return df

def get_weight_table(business_type: str = "General") -> pd.DataFrame:
    """Return weights as a neat DataFrame for display."""
    weights = get_business_weights(business_type)
    wt = pd.DataFrame(
        list(weights.items()),
        columns=["Feature", "Weight"],
    )
    wt["Contribution (%)"] = (wt["Weight"] * 100).astype(int)
    return wt.sort_values("Weight", ascending=False).reset_index(drop=True)

def generate_recommendation(row: pd.Series, business_type: str) -> dict:
    """
    Generates dynamic reasons and conclusions based on real location data
    and business requirements.
    """
    score = row["suitability_score"]
    pop_den = row["population_density"]
    income = row["avg_income"]
    foot_traffic = row["foot_traffic"]
    access = row["accessibility_score"]
    comp = row["competitors"]
    
    reasons = []
    recommendation = "Moderate"
    
    # 1. Hard Business Rules
    if business_type == "Clothing Store":
        if income > 100000 and foot_traffic > 120:
            recommendation = "Highly Recommended"
            reasons.append(f"Exceptional foot traffic ({foot_traffic} visitors/day) guarantees high walk-in visibility.")
            reasons.append(f"Premium income levels (₹{income:,.0f}) indicate very strong purchasing power for retail.")
        elif income < 60000 or foot_traffic < 60:
            recommendation = "Not Recommended"
            reasons.append("Low foot traffic and income severely limits potential customer base for clothing.")
            
    elif business_type == "Electronics Store":
        if income > 100000 and access >= 7:
            recommendation = "Highly Recommended"
            reasons.append(f"High average income (₹{income:,.0f}) is ideal for high-ticket electronic purchases.")
            reasons.append(f"Excellent accessibility (score: {access}/10) makes it easy for customers to transport large items.")
        elif income < 70000:
            recommendation = "Not Recommended"
            reasons.append("Insufficient income level to sustain consistent electronics sales.")
            
    elif business_type == "Grocery Store":
        if pop_den > 12000 and access >= 7:
            recommendation = "Highly Recommended"
            reasons.append(f"Dense population ({pop_den} people/sq km) creates massive daily demand for groceries.")
            reasons.append(f"High accessibility (score: {access}/10) ensures daily shopping convenience.")
        elif pop_den < 6000:
            recommendation = "Not Recommended"
            reasons.append("Low population density limits the daily volume required for a grocery store.")
            
    elif business_type == "Café":
        if foot_traffic > 130:
            recommendation = "Highly Recommended"
            reasons.append(f"Massive foot traffic ({foot_traffic} visitors/day) is perfectly suited for spontaneous café visits.")
            reasons.append(f"Good population density ({pop_den} people/sq km) provides a steady local customer base.")
        elif foot_traffic < 60:
            recommendation = "Not Recommended"
            reasons.append("Extremely low foot traffic makes a café business unviable here.")
            
    # Add general fallback reasons if list is empty
    if not reasons:
        reasons.append(f"Population density is {pop_den} people/sq km.")
        reasons.append(f"Average income stands at ₹{income:,.0f}.")
        reasons.append(f"Accessibility is rated at {access}/10.")
    
    # Always add competition reason
    if comp <= 3:
        reasons.append(f"Very low competition (only {comp} competitors) presents a highly strategic opportunity.")
    elif comp >= 10:
        reasons.append(f"High saturation ({comp} competitors) means marketing costs will be higher.")
        if recommendation == "Highly Recommended":
            recommendation = "Moderate" # Downgrade if heavily saturated

    # Fallback to score-based recommendation if still not caught by strict rules
    if recommendation == "Moderate":
        if score > 0.65:
            recommendation = "Highly Recommended"
        elif score < 0.35:
            recommendation = "Not Recommended"

    # 2. Final Conclusion Generation
    if recommendation == "Highly Recommended":
        conclusion = (
            f"This location is highly suitable for a {business_type}. "
            f"The combination of strong local demographics and favorable infrastructure provides an excellent foundation. "
            f"Therefore, setting up the business here is strongly recommended."
        )
    elif recommendation == "Moderate":
        conclusion = (
            f"This location presents moderate potential for a {business_type}. "
            f"While there are some strong indicators, careful consideration of market saturation and costs is required. "
            f"A pilot or phased rollout could mitigate risks."
        )
    else:
        conclusion = (
            f"This location is not recommended for a {business_type}. "
            f"Critical factors such as foot traffic or purchasing power fall below the viable threshold. "
            f"It is advised to explore more premium or accessible areas."
        )

    return {
        "score": score,
        "recommendation": recommendation,
        "reasons": reasons,
        "conclusion": conclusion
    }


# ===================================================================
# Quick-run: score and rank all locations
# ===================================================================
if __name__ == "__main__":
    from data_processing import generate_locations
    from feature_engineering import engineer_location_features

    # Generate & engineer
    locations = generate_locations()
    loc_feat = engineer_location_features(locations)

    # Score & rank for a specific business
    bt = "Café"
    ranked = score_locations(loc_feat, business_type=bt)

    # --- Display ranking ---
    print("=" * 65)
    print(f"  LOCATION SUITABILITY RANKING ({bt})")
    print("=" * 65)
    display_cols = ["rank", "location_name", "suitability_score"]
    print(ranked[display_cols].to_string(index=False))

    # --- Display weights ---
    print("\n--- Weight Configuration ---")
    print(get_weight_table(bt).to_string(index=False))
