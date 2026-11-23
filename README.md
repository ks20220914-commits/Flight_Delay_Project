# ✈️ AeroPredict AI: Enterprise Flight Delay Prediction & Analytics Platform

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange?style=for-the-badge)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Developed by:** [Kyrillos Saeed](https://www.linkedin.com/in/kyrillos-saeed)  
**Portfolio:** [Kyrillos Saeed Portfolio](https://ks20220914-commits.github.io/Kyrillos-Saeed.github.io/)  
**Live Application:** [Insert your Streamlit Cloud Link Here]

---

## 🌟 The "WOW" Factor (Project Highlights)
- **5.4+ Million** flights analyzed across 12 months.
- **0.8732 ROC-AUC** achieved using Optuna-tuned XGBoost.
- **71% Recall** for delayed flights (catching 7 out of 10 delays).
- **$12,000** estimated savings per avoided delay.
- Custom **Cyberpunk Dark Mode UI** with 7 interactive tabs and 40+ dynamic filters.

---

## 📸 Platform Screenshots
*(Replace these placeholders with your actual screenshots saved in the `assets/` folder)*

### 1. Live Dashboard & Prediction
![Dashboard](assets/dashboard.png)
*Real-time prediction with gauge charts, What-If simulator, and PDF report generation.*

### 2. Deep-Dive Analytics & Comparison
![Analytics](assets/analytics.png)
*Airport comparison, airline performance, and weather impact analysis.*

### 3. Crisis Simulator & Geospatial Map
![Crisis](assets/crisis.png)
*Simulating extreme operational conditions and visualizing US flight paths.*

---

## 📖 Executive Summary
Flight delays cost the global aviation industry billions of dollars annually in operational costs, crew rescheduling, and passenger compensation. **AeroPredict AI** is an end-to-end, production-ready Data Science platform designed to predict whether a commercial flight will arrive **more than 30 minutes late**.

By leveraging 12 months of historical flight data, integrating real-time weather APIs, and deploying a hyper-tuned XGBoost model, this platform achieves an **ROC-AUC of 0.8732** and a **Recall of 71%**. The system is deployed via a fully interactive Streamlit dashboard that provides actionable business intelligence, ROI analysis, and real-time predictions.

---

## 🏗️ Project Architecture (CRISP-DM Framework)
This project strictly follows the **CRISP-DM** methodology, covering all 6 phases from Business Understanding to Deployment.

text
Flight_Delay_Project/
│
├── app/                  # Streamlit Application (UI & Logic)
│   ├── streamlit_app.py  # Main Application Code
│   └── style.css         # Custom Cyberpunk CSS Styling
│
├── data/
│   ├── raw/              # Raw CSVs (BTS 12 Months + Open-Meteo Weather)
│   ├── interim/          # Merged, Cleaned, and Validated Data
│   └── processed/        # Final Engineered Features (Ready for ML)
│
├── models/               # Saved Trained Pipelines (.pkl)
│
├── notebooks/            # Jupyter Notebooks for R&D
│   ├── 03_eda.ipynb      # 30-Cell Advanced Exploratory Data Analysis
│   ├── 04_modeling.ipynb # 18-Cell Optuna & SHAP Modeling
│   └── 05_eval.ipynb     # Comprehensive Evaluation & Threshold Analysis
│
├── scripts/              # Automated Data Pipeline Scripts
│   ├── 01_fetch_weather.py
│   ├── 02_merge_clean.py
│   ├── 03_feature_engineering.py
│   └── test_model.py     # Unit Tests for Model Validation
│
├── requirements.txt      # Python Dependencies
└── README.md
`

---

🔍 Phase 1: Business Understanding

The Problem: Airlines struggle to anticipate cascading delays caused by aircraft rotation and weather conditions. 
The Goal: Build a predictive system to estimate the probability of a flight being delayed (Arrival Delay ≥ 30 mins) before it departs.
Success Criteria:

· Achieve a ROC-AUC > 0.80.
· Maintain a Recall (for the delayed class) > 65%.
· Provide a user-friendly dashboard for stakeholders.

---
🧹 Phase 2: Data Understanding & Preparation

1. Data Collection

· BTS On-Time Performance: Downloaded 12 monthly CSV files (Jan-Dec 2024).
· Open-Meteo Weather API: Integrated historical hourly weather data for 334 US airports.

2. Data Merging & Cleaning (Enterprise Pipeline)

· Error Handling: Built a robust pipeline using try-except and logging.
· Merging: Combined flight data with weather data using ORIGIN, FL_DATE, and hour.
· Data Validation: Dropped cancelled/diverted flights, removed missing ARR_DELAY values, and dropped duplicates.
· Dataset Size: Final cleaned dataset contained 5,424,851 rows.

3. Advanced Feature Engineering

· prev_arr_delay: The arrival delay of the aircraft's previous flight (Rotation Chain).
· turnaround_time: The time between the previous arrival and current departure.
· origin_hourly_flights: A proxy for airport congestion.

---

📊 Phase 3: Advanced Exploratory Data Analysis (EDA)

We conducted a 30-cell, highly visual EDA using Plotly and Datashader:

· Target Imbalance: 84% On-Time vs. 16% Delayed.
· Time Series Analysis: Heatmaps revealed delays peak on Sunday evenings and Friday afternoons.
· Weather Impact: 3D scatter plots showed heavy rain (>10mm) and high wind speeds (>40 km/h) drastically increase delay probabilities.
· Airport Pareto Analysis: Proved that 20% of hub airports (ATL, ORD, DFW) account for 80% of all delays.

---

🤖 Phase 4: Machine Learning Modeling

We framed this as a Binary Classification problem.

1. Algorithm Benchmarking (Baseline)

· XGBoost: 0.8636 ROC-AUC
· LightGBM: 0.8624 ROC-AUC
· CatBoost: 0.8569 ROC-AUC
  Decision: Proceeded with XGBoost.

2. Hyperparameter Tuning (Optuna)

We utilized Optuna to run 20 trials and find optimal hyperparameters.

· Tuned Params: max_depth=10, learning_rate=0.144, n_estimators=276.
· Best Tuning ROC-AUC: 0.8722.

3. Final Model Training

Using the tuned parameters and scale_pos_weight (to handle imbalance), we trained the final pipeline on 4.3 Million rows.

---

📈 Phase 5: Evaluation & Business ROI

The model was evaluated on a strict holdout set of 1.08 Million flights (20%).

· Final ROC-AUC: 0.8732 (Excellent predictive power).
· Accuracy: 86%.
· Recall (Delayed Class): 71%.
· Precision (Delayed Class): 49%.
· Business ROI: Estimated savings of up to $12,000 per avoided delay.
· Explainability (SHAP): SHAP confirmed prev_arr_delay has the highest impact.

---

🖥️ Phase 6: Deployment & Dashboard Features

The application is deployed using Streamlit with a custom Cyberpunk Dark Mode UI.

Core Features:

1. Real-Time AI Prediction: Instant delay probabilities.
2. What-If Simulator: Dynamically adjust previous flight delay to see risk impact.
3. Automated PDF Reports: Generate downloadable operational risk reports.
4. Airport Comparison: Compare delay rates and weather variances using Radar Charts.
5. Airline Performance: Analyze market share and operational efficiency.
6. Geo-Spatial Analysis: Visualizes flight paths across the US using Scattergeo plots.
7. Crisis Simulator: Simulate extreme conditions (e.g., Heavy Storm) and get emergency actions.
8. Enterprise Batch Prediction: Upload a CSV of thousands of flights and download predictions.

---

🚀 How to Run Locally

1. Clone the repository:
  
   git clone https://github.com/ks20220914-commits/Flight_Delay_Project.git
   cd Flight_Delay_Project
   
2. Create a virtual environment & install dependencies:
  
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   
3. Run the Streamlit App:
  
   streamlit run app/streamlit_app.py
   
---

🧪 Unit Testing

To ensure the model's robustness, we implemented unit tests using unittest. Run them using:

python scripts/test_model.py
---

🔮 Future Work

· Integrate Real-time Live Flight APIs (e.g., OpenSky Network).
· Deploy the model using FastAPI as a microservice.
· Incorporate Deep Learning (LSTM) to model flight rotation sequences.

---

Developed with by Kyrillos Saeed

