"""
app.py
------
Streamlit dashboard entry point.
Run with:  streamlit run app.py

Ties together all modules:
  - data_processing   (load data)
  - feature_engineering (build features)
  - location_model     (score & rank locations)
  - demand_model       (XGBoost predictions)
  - visualization      (Plotly charts + Folium map)
"""

import sys
from pathlib import Path

# Ensure src/ is on the path so imports work cleanly
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st
import pandas as pd

from data_processing import generate_locations, generate_sales, PRODUCT_CATEGORIES, DATA_DIR
from feature_engineering import engineer_location_features, engineer_demand_features
from location_model import score_locations, get_weight_table, generate_recommendation
from demand_model import run_demand_pipeline
from visualization import (
    plot_location_ranking,
    plot_demand_trend,
    plot_feature_importance,
    plot_category_sales,
    plot_revenue_trend,
    plot_weight_breakdown,
    create_location_map,
)

try:
    from streamlit_folium import st_folium
    HAS_ST_FOLIUM = True
except ImportError:
    HAS_ST_FOLIUM = False


# ===================================================================
# PAGE CONFIG
# ===================================================================
st.set_page_config(
    page_title="Karnataka Retail AI Dashboard",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===================================================================
# CACHED DATA LOADING  (runs once, then cached)
# ===================================================================
@st.cache_data
def load_base_data():
    """Generates the base datasets just once."""
    locations = generate_locations()
    sales = generate_sales(locations)
    loc_feat = engineer_location_features(locations)
    return locations, sales, loc_feat

locations, sales, loc_feat = load_base_data()


# ===================================================================
# SIDEBAR FILTERS
# ===================================================================
st.sidebar.title("Configuration")

# Business Type filter (New Feature)
selected_business = st.sidebar.selectbox(
    "Business Type",
    options=PRODUCT_CATEGORIES,
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.title("Location Filters")

# Compute Rankings dynamically based on selected business type
ranked = score_locations(loc_feat, business_type=selected_business)

# City filter (Default to Bengaluru)
all_cities = sorted(ranked["city"].unique().tolist())
default_city_index = all_cities.index("Bengaluru") if "Bengaluru" in all_cities else 0

selected_city = st.sidebar.selectbox(
    "Select City",
    options=["All Karnataka"] + all_cities,
    index=default_city_index + 1,  # +1 because of 'All Karnataka'
)

# Filter ranked dataframe based on selected city
if selected_city != "All Karnataka":
    display_ranked = ranked[ranked["city"] == selected_city].reset_index(drop=True)
    # Recalculate rank within the city
    display_ranked["rank"] = display_ranked["suitability_score"].rank(ascending=False).astype(int)
else:
    display_ranked = ranked.copy()

# Location/Area filter
all_locations = display_ranked["location_name"].tolist()
selected_location = st.sidebar.selectbox(
    "Select Area",
    options=["All Areas"] + all_locations,
    index=0,
)


# ===================================================================
# DEMAND MODEL (Trained on the fly for simplicity or cached)
# ===================================================================
@st.cache_data
def load_demand_model(_sales, _locations):
    demand_df = engineer_demand_features(_sales, _locations)
    model, metrics, test_df, importance = run_demand_pipeline(demand_df)
    return metrics, test_df, importance

metrics, test_df, importance = load_demand_model(sales, locations)


# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Performance (XGBoost)")
st.sidebar.metric("R² Score", f"{metrics['R2']:.4f}")
st.sidebar.metric("MAE", f"{metrics['MAE']:.2f} units")
st.sidebar.metric("RMSE", f"{metrics['RMSE']:.2f} units")

st.sidebar.markdown("---")
st.sidebar.caption("Built with Python, XGBoost, Plotly & Folium")


# ===================================================================
# HEADER
# ===================================================================
st.title("AI-Powered Retail Dashboard: Karnataka Region")
st.markdown(
    f"Identify the **best store locations** and **predict future demand** "
    f"specifically for a **{selected_business}**. "
    "Focus: **Bengaluru & Major Karnataka Cities**."
)


# ===================================================================
# ROW 1: KPI METRICS
# ===================================================================
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

best_overall = ranked.iloc[0]
best_in_selection = display_ranked.iloc[0]

# Filter sales based on city selection and business type for KPIs
if selected_city != "All Karnataka":
    city_loc_ids = display_ranked["location_id"].tolist()
    display_sales = sales[(sales["location_id"].isin(city_loc_ids)) & (sales["product_category"] == selected_business)]
else:
    display_sales = sales[sales["product_category"] == selected_business]

total_revenue = display_sales["revenue"].sum()
total_units = display_sales["units_sold"].sum()

if selected_city == "All Karnataka":
    col1.metric("Top Locality (Karnataka)", best_overall["area"], f"Score: {best_overall['suitability_score']:.3f}")
else:
    col1.metric(f"Top Locality ({selected_city})", best_in_selection["area"], f"Score: {best_in_selection['suitability_score']:.3f}")

col2.metric(f"Total Revenue ({selected_city})", f"₹ {total_revenue:,.0f}")
col3.metric("Total Units Sold", f"{total_units:,}")
col4.metric("Best Overall (Karnataka)", best_overall["area"], f"{best_overall['city']}")


# ===================================================================
# ROW 2: LOCATION DETAIL & RECOMMENDATION (if a specific location is selected)
# ===================================================================
if selected_location != "All Areas":
    st.markdown("---")
    st.header(f"Smart Recommendation: {selected_location}")

    loc_row = display_ranked[display_ranked["location_name"] == selected_location].iloc[0]
    rec_data = generate_recommendation(loc_row, selected_business)

    # Status color mapping
    status_color = {
        "Highly Recommended": "🟢",
        "Moderate": "🟡",
        "Not Recommended": "🔴"
    }.get(rec_data["recommendation"], "⚪")

    d1, d2, d3 = st.columns([1, 1, 2])
    with d1:
        st.metric(f"Rank (in {selected_city})", f"#{int(loc_row['rank'])} of {len(display_ranked)}")
        st.metric("Suitability Score", f"{loc_row['suitability_score']:.3f}")
    
    with d2:
        st.markdown(f"### {status_color} {rec_data['recommendation']}")
        st.markdown(f"**Business Type:** {selected_business}")
    
    with d3:
        st.markdown("### Data-Driven Reasons:")
        for reason in rec_data["reasons"]:
            st.markdown(f"- {reason}")
            
    st.info(f"**Conclusion:** {rec_data['conclusion']}")


# ===================================================================
# ROW 3: MAP + LOCATION RANKING  (side by side)
# ===================================================================
st.markdown("---")
if selected_city == "All Karnataka":
    st.header(f"Store Location Analysis: All Karnataka ({selected_business})")
else:
    st.header(f"Store Location Analysis: {selected_city} ({selected_business})")

map_col, rank_col = st.columns([3, 2])

with map_col:
    st.subheader("Location Map")
    store_map = create_location_map(display_ranked)
    if HAS_ST_FOLIUM:
        st_folium(store_map, width=700, height=450, returned_objects=[])
    else:
        # Fallback: render as HTML
        import streamlit.components.v1 as components
        map_html = store_map._repr_html_()
        components.html(map_html, height=450)

with rank_col:
    st.subheader(f"Suitability Ranking")
    st.plotly_chart(plot_location_ranking(display_ranked), use_container_width=True)


# ===================================================================
# ROW 4: DEMAND PREDICTION
# ===================================================================
st.markdown("---")
st.header("Demand Prediction (XGBoost)")

# Apply filters
loc_id_filter = None
cat_filter = selected_business # Sync category filter with business type

if selected_location != "All Areas":
    loc_id_filter = display_ranked[display_ranked["location_name"] == selected_location]["location_id"].iloc[0]

pred_col, imp_col = st.columns([3, 2])

with pred_col:
    st.plotly_chart(
        plot_demand_trend(test_df, location_id=loc_id_filter, category=cat_filter),
        use_container_width=True,
    )

with imp_col:
    st.plotly_chart(
        plot_feature_importance(importance),
        use_container_width=True,
    )


# ===================================================================
# ROW 5: SALES INSIGHTS
# ===================================================================
st.markdown("---")
st.header(f"Sales Insights ({selected_city})")

sales_col1, sales_col2 = st.columns(2)

with sales_col1:
    st.plotly_chart(plot_category_sales(display_sales), use_container_width=True)

with sales_col2:
    st.plotly_chart(plot_revenue_trend(display_sales), use_container_width=True)


# ===================================================================
# ROW 6: MODEL INFO
# ===================================================================
st.markdown("---")
st.header("Model & Scoring Details")

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.subheader(f"Scoring Weights ({selected_business})")
    st.plotly_chart(plot_weight_breakdown(get_weight_table(selected_business)), use_container_width=True)
    st.dataframe(get_weight_table(selected_business), use_container_width=True, hide_index=True)

with info_col2:
    st.subheader("XGBoost Model Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score", f"{metrics['R2']:.4f}")
    m2.metric("MAE", f"{metrics['MAE']:.2f}")
    m3.metric("RMSE", f"{metrics['RMSE']:.2f}")

    st.markdown("""
    **What these metrics mean:**
    - **R² = {r2}** — the model explains {r2_pct}% of demand variation
    - **MAE = {mae}** — on average, predictions are off by ~{mae:.0f} units
    - **RMSE = {rmse}** — penalizes large errors more heavily
    """.format(r2=metrics["R2"], r2_pct=round(metrics["R2"]*100, 1),
               mae=metrics["MAE"], rmse=metrics["RMSE"]))

    st.subheader("Prediction Sample")
    sample_cols = ["location_id", "date", "product_category", "units_sold", "predicted"]
    st.dataframe(
        test_df[sample_cols].head(20),
        use_container_width=True,
        hide_index=True,
    )


# ===================================================================
# FOOTER
# ===================================================================
st.markdown("---")
st.caption(
    "AI-Powered Retail Dashboard: Karnataka Region | "
    "Built with Streamlit, XGBoost, Plotly & Folium"
)
