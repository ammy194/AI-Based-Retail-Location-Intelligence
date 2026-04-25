"""
visualization.py
-----------------
Reusable Plotly chart functions and Folium map for the Streamlit dashboard.

- Plotly charts return plotly.graph_objects.Figure
- Map function returns a folium.Map object
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import folium


# ===================================================================
# 1. LOCATION SUITABILITY — horizontal bar chart
# ===================================================================
def plot_location_ranking(ranked_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of suitability scores, colour-coded:
      green = top 3, amber = top 50%, red = bottom 50%.
    """
    df = ranked_df.sort_values("suitability_score", ascending=True).copy()

    # Colour by tier
    def tier_color(rank):
        # Dynamic color based on rank (Highlight top 3)
        if rank <= 3:
            return "#2ecc71"   # green — best
        elif rank <= len(ranked_df) * 0.5:
            return "#f39c12"   # amber — average
        else:
            return "#e74c3c"   # red   — weak

    df["color"] = df["rank"].apply(tier_color)

    fig = go.Figure(go.Bar(
        x=df["suitability_score"],
        y=df["location_name"],
        orientation="h",
        marker_color=df["color"],
        text=df["suitability_score"].round(3),
        textposition="outside",
    ))
    fig.update_layout(
        title="Location Suitability Ranking",
        xaxis_title="Suitability Score",
        yaxis_title="",
        height=500,
        margin=dict(l=200),
        template="plotly_white",
    )
    return fig


# ===================================================================
# 2. DEMAND TREND — actual vs predicted line chart
# ===================================================================
def plot_demand_trend(test_df: pd.DataFrame,
                      location_id: int = None,
                      category: str = None) -> go.Figure:
    """
    Line chart: actual units_sold vs predicted over time.
    Optionally filter by location and/or category.
    """
    df = test_df.copy()
    title_parts = ["Demand: Actual vs Predicted"]

    if location_id is not None:
        df = df[df["location_id"] == location_id]
        title_parts.append(f"Location {location_id}")
    if category is not None:
        df = df[df["product_category"] == category]
        title_parts.append(category)

    # Aggregate by date (in case multiple categories/locations)
    agg = df.groupby("date").agg(
        actual=("units_sold", "sum"),
        predicted=("predicted", "sum"),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg["date"], y=agg["actual"],
        mode="lines+markers", name="Actual",
        line=dict(color="#3498db", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=agg["date"], y=agg["predicted"],
        mode="lines+markers", name="Predicted",
        line=dict(color="#e74c3c", width=2, dash="dash"),
    ))
    fig.update_layout(
        title=" | ".join(title_parts),
        xaxis_title="Date",
        yaxis_title="Units Sold",
        height=400,
        template="plotly_white",
        legend=dict(orientation="h", y=1.12),
    )
    return fig


# ===================================================================
# 3. FEATURE IMPORTANCE — bar chart
# ===================================================================
def plot_feature_importance(importance_df: pd.DataFrame,
                            top_n: int = 10) -> go.Figure:
    """Top-N feature importances from XGBoost with human-readable labels."""
    
    # Dictionary mapping for renaming
    feature_mapping = {
        "lag_1": "Sales (1 Month Ago)",
        "lag_2": "Sales (2 Months Ago)",
        "lag_3": "Sales (3 Months Ago)",
        "rolling_mean_3": "Average Sales (Last 3 Months)",
        "rolling_std_3": "Sales Variability (Last 3 Months)",
        "foot_traffic": "Foot Traffic (Number of Visitors)",
        "month": "Month of Year",
        "month_sin": "Seasonality (Sine Component)",
        "month_cos": "Seasonality (Cosine Component)",
        "quarter": "Quarter of Year",
        "population": "Population",
        "income_level": "Income Level",
        "competitors": "Competitors",
        "accessibility_score": "Accessibility",
        "parking_availability": "Parking Availability",
        "category_encoded": "Product Category"
    }

    df = importance_df.copy()
    
    # Replace feature names BEFORE plotting
    df["Feature"] = df["Feature"].map(lambda x: feature_mapping.get(x, x))
    
    # Sort descending to get the true top_n, then sort ascending so Plotly displays the highest at the very top
    df = df.sort_values("Importance", ascending=False).head(top_n).sort_values("Importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=df["Importance"],
        y=df["Feature"],
        orientation="h",
        marker_color="#8e44ad",
        text=df["Importance"].round(4),
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Top {top_n} Feature Importances (XGBoost)",
        xaxis_title="Importance",
        yaxis_title="",
        height=400,
        margin=dict(l=220),  # Increased left margin for longer text labels
        template="plotly_white",
    )
    return fig


# ===================================================================
# 4. CATEGORY-WISE SALES — grouped bar chart
# ===================================================================
def plot_category_sales(sales_df: pd.DataFrame) -> go.Figure:
    """Total units sold by product category across all locations."""
    cat_totals = (
        sales_df.groupby("product_category")["units_sold"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
    fig = go.Figure(go.Bar(
        x=cat_totals["product_category"],
        y=cat_totals["units_sold"],
        marker_color=colors[:len(cat_totals)],
        text=cat_totals["units_sold"],
        textposition="outside",
    ))
    fig.update_layout(
        title="Total Units Sold by Category",
        xaxis_title="Product Category",
        yaxis_title="Total Units Sold",
        height=400,
        template="plotly_white",
    )
    return fig


# ===================================================================
# 5. MONTHLY REVENUE TREND — area chart
# ===================================================================
def plot_revenue_trend(sales_df: pd.DataFrame) -> go.Figure:
    """Monthly total revenue across all locations (area chart)."""
    monthly = (
        sales_df.groupby("date")["revenue"]
        .sum()
        .reset_index()
    )

    fig = go.Figure(go.Scatter(
        x=monthly["date"],
        y=monthly["revenue"],
        fill="tozeroy",
        line=dict(color="#2ecc71", width=2),
        fillcolor="rgba(46,204,113,0.2)",
    ))
    fig.update_layout(
        title="Monthly Revenue Trend (All Locations)",
        xaxis_title="Date",
        yaxis_title="Revenue (Rs.)",
        height=400,
        template="plotly_white",
    )
    return fig


# ===================================================================
# 6. WEIGHT BREAKDOWN — pie/donut chart for location model weights
# ===================================================================
def plot_weight_breakdown(weight_df: pd.DataFrame) -> go.Figure:
    """Donut chart showing location model weight distribution."""
    fig = go.Figure(go.Pie(
        labels=weight_df["Feature"],
        values=weight_df["Weight"],
        hole=0.45,
        marker_colors=["#3498db", "#2ecc71", "#e74c3c",
                        "#f39c12", "#9b59b6", "#1abc9c"],
        textinfo="label+percent",
    ))
    fig.update_layout(
        title="Location Scoring Weight Distribution",
        height=400,
        template="plotly_white",
    )
    return fig


# ===================================================================
# 7. FOLIUM MAP — interactive store location map
# ===================================================================
def _tier_color(rank: int) -> str:
    """Map marker colour by rank tier."""
    if rank <= 5:
        return "green"
    elif rank <= 10:
        return "orange"
    return "red"


def _tier_icon(rank: int) -> str:
    """Map marker icon by rank tier."""
    if rank <= 3:
        return "star"
    return "info-sign"


def create_location_map(ranked_df: pd.DataFrame) -> folium.Map:
    """
    Create an interactive Folium map with:
      - Color-coded markers (green/orange/red by tier)
      - Rich HTML popups with score details
      - Auto-fitted bounds to show all locations

    Parameters
    ----------
    ranked_df : DataFrame with columns:
        latitude, longitude, location_name, rank,
        suitability_score, plus feature scores.

    Returns
    -------
    folium.Map — render in Streamlit with st_folium(map)
    """
    # Centre the map on Bengaluru
    center_lat = 12.9716
    center_lon = 77.5946

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles="CartoDB positron",
    )

    # Add a marker for each location
    for _, row in ranked_df.iterrows():
        color = _tier_color(row["rank"])
        icon  = _tier_icon(row["rank"])

        # Build a nice HTML popup
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; width: 220px;">
            <h4 style="margin:0 0 6px 0; color: #2c3e50;">
                #{int(row['rank'])} {row['location_name']}
            </h4>
            <hr style="margin:4px 0;">
            <table style="font-size:12px; width:100%;">
                <tr><td><b>Score</b></td>
                    <td style="text-align:right;">{row['suitability_score']:.3f}</td></tr>
                <tr><td><b>Market Potential</b></td>
                    <td style="text-align:right;">{row['market_potential']:.3f}</td></tr>
                <tr><td><b>Competition</b></td>
                    <td style="text-align:right;">{row['competition_score']:.3f}</td></tr>
                <tr><td><b>Affordability</b></td>
                    <td style="text-align:right;">{row['affordability']:.3f}</td></tr>
                <tr><td><b>Access+Parking</b></td>
                    <td style="text-align:right;">{row['access_parking']:.3f}</td></tr>
                <tr><td><b>Foot Traffic</b></td>
                    <td style="text-align:right;">{row['foot_traffic']:.3f}</td></tr>
            </table>
        </div>
        """

        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"#{int(row['rank'])} {row['location_name']}",
            icon=folium.Icon(color=color, icon=icon, prefix="glyphicon"),
        ).add_to(m)

    # Fit map to show all markers
    bounds = [
        [ranked_df["latitude"].min(), ranked_df["longitude"].min()],
        [ranked_df["latitude"].max(), ranked_df["longitude"].max()],
    ]
    m.fit_bounds(bounds, padding=(30, 30))

    return m


# ===================================================================
# Quick-run: validate all charts + map
# ===================================================================
if __name__ == "__main__":
    from data_processing import generate_locations, generate_sales
    from feature_engineering import engineer_location_features, engineer_demand_features
    from location_model import score_locations, get_weight_table
    from demand_model import run_demand_pipeline

    # --- Build everything ---
    locations = generate_locations()
    sales = generate_sales(locations)

    loc_feat = engineer_location_features(locations)
    ranked = score_locations(loc_feat)

    demand_df = engineer_demand_features(sales, locations)
    model, metrics, test_df, importance = run_demand_pipeline(demand_df)

    # --- Generate charts ---
    fig1 = plot_location_ranking(ranked)
    fig2 = plot_demand_trend(test_df)
    fig3 = plot_feature_importance(importance)
    fig4 = plot_category_sales(sales)
    fig5 = plot_revenue_trend(sales)
    fig6 = plot_weight_breakdown(get_weight_table())

    # --- Generate map ---
    store_map = create_location_map(ranked)
    map_path = "data/location_map.html"
    store_map.save(map_path)

    print("[OK] All 6 Plotly charts created successfully.")
    print(f"[OK] Folium map saved to {map_path}")
    print("     Open it in your browser to preview!")
    print("\nFunctions available:")
    print("  Charts : plot_location_ranking, plot_demand_trend,")
    print("           plot_feature_importance, plot_category_sales,")
    print("           plot_revenue_trend, plot_weight_breakdown")
    print("  Map    : create_location_map")
