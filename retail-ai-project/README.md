# 🏪 AI-Powered Retail Store Location & Demand Dashboard

An end-to-end AI/ML project that **identifies optimal retail store locations** and **predicts future product demand** using machine learning, displayed through an interactive Streamlit dashboard.

---

## 🎯 Project Goal

1. **Location Analysis** — Score and rank 15 candidate store locations using a Weighted Scoring Model based on population, income, competition, accessibility, and foot traffic.
2. **Demand Prediction** — Train an XGBoost regression model on 36 months of historical sales data to forecast future product demand.
3. **Interactive Dashboard** — Visualize insights through charts, maps, and filters using Streamlit, Plotly, and Folium.

---

## 📸 Dashboard Features

| Feature | Description |
|---|---|
| **KPI Metrics** | Best location, total revenue, total units, avg suitability score |
| **Interactive Map** | Folium map with color-coded markers (green/orange/red by tier) |
| **Location Ranking** | Horizontal bar chart with suitability scores |
| **Demand Prediction** | Actual vs Predicted line chart with location/category filters |
| **Feature Importance** | Top-10 XGBoost feature importances |
| **Sales Insights** | Category-wise sales + monthly revenue trend |
| **Model Details** | Scoring weights donut chart + XGBoost metrics |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Pandas, NumPy | Data manipulation |
| Scikit-learn | Feature scaling (MinMaxScaler) |
| XGBoost | Demand prediction model |
| Plotly | Interactive charts |
| Folium | Map visualization |
| Streamlit | Dashboard framework |

---

## 📁 Project Structure

```
retail-ai-project/
│
├── data/
│   ├── locations.csv          # 15 candidate store locations
│   └── sales.csv              # 36 months × 4 categories × 15 locations
│
├── src/
│   ├── data_processing.py     # Dataset generation & loading
│   ├── feature_engineering.py # Derived features for ML
│   ├── location_model.py      # Weighted Scoring Model for locations
│   ├── demand_model.py        # XGBoost demand predictor
│   └── visualization.py       # Plotly charts + Folium map
│
├── app.py                     # Streamlit dashboard
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

---

## 🚀 How to Run

### 1. Clone & Setup
```bash
cd retail-ai-project
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. Generate Data (optional — dashboard does this automatically)
```bash
python src/data_processing.py
```

### 3. Launch Dashboard
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 🤖 Models Explained

### Location Suitability Model (Weighted Scoring)

Uses 6 normalized features with configurable weights:

| Feature | Weight | Logic |
|---|---|---|
| Market Potential | 25% | population × avg_income |
| Competition Score | 20% | 1 / (1 + num_competitors) |
| Affordability | 15% | avg_income / rent |
| Foot Traffic | 15% | Daily foot traffic |
| Access + Parking | 15% | Accessibility + parking scores |
| Value Score | 10% | foot_traffic / rent |

**Why not ML?** We have no labeled "good/bad" location data. A weighted model is transparent, explainable, and adjustable by business stakeholders.

### Demand Prediction Model (XGBoost)

| Detail | Value |
|---|---|
| Algorithm | XGBoost Regressor |
| Features | 17 (time, lag, rolling, location context) |
| Train/Test Split | Time-based (last 6 months = test) |
| R² Score | **0.9886** |
| MAE | 9.66 units |
| Top Feature | rolling_mean_3 (82.5% importance) |

**Why XGBoost?** Best-in-class for tabular data, fast training, built-in feature importance.

---

## 📊 Key Results

- **Best Location:** Connaught Place, Delhi (Score: 0.661) — zero competitors, high accessibility
- **Worst Location:** Saket, Delhi (Score: 0.127) — weak across all factors
- **Model Accuracy:** R² = 0.9886 (explains 98.9% of demand variation)
- **Most Important Feature:** 3-month rolling average of past sales

---

## 🎤 Viva Quick Answers

**Q: Why synthetic data?**
> Real retail data is proprietary. Our synthetic data simulates realistic patterns: seasonal trends, category differences, and location-driven demand.

**Q: Why Weighted Scoring instead of ML for locations?**
> No labeled training data exists. WSM is fully transparent — we can explain every score to a stakeholder.

**Q: Why time-based train/test split?**
> Random splits leak future data into training. Time-based splits respect the temporal nature of the problem.

**Q: What is cyclical encoding?**
> We encode months as sin/cos so the model knows December (12) is close to January (1), not far apart.

**Q: Why is rolling_mean_3 the top feature?**
> In retail, recent sales history is the strongest predictor. A 3-month average smooths noise while capturing trend.

---

## 📝 Future Improvements

- Connect to real retail datasets (Kaggle, government open data)
- Add more ML models for comparison (Random Forest, LSTM)
- Implement real-time data ingestion
- Add user authentication and role-based access
- Deploy on cloud (Streamlit Cloud, AWS, GCP)

---

*Built for hackathon/viva demonstration purposes.*
