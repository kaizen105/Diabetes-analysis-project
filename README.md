# 🩺 Diabetes Readmission Prediction – Machine Learning Project

An end-to-end data science pipeline built on the **UCI Diabetes 130-US Hospitals dataset** (~100k encounters, ~70k patients). This project delivers a full analytical workflow — from data preparation and exploratory analysis through predictive modelling, SQL-driven business insights, interactive dashboarding, and a deployed clinical decision-support interface.

[![Streamlit App](https://img.shields.io/badge/Live_App-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://diabetes-analysis-project-mfz5kcrwnusng4wztuzydk.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PowerBI](https://img.shields.io/badge/Power_BI-Dashboard-F2C811?style=flat-square&logo=powerbi&logoColor=black)](powerbi/dashboard.pbix)

---

## 📂 Repository Structure

```
diabetes_analysis_project/
│
├── datasets/                   # Raw and processed data (ignored in .gitignore)
│   ├── diabetic_data.csv           # Raw source data
│   ├── diabetes_data_ml.csv        # Encoded dataset for ML
│   ├── diabetic_data_clean.csv     # Categorical dataset for Power BI
│   └── IDS_mapping.csv             # Diagnosis code mappings
│
├── notebook/                   # Jupyter notebooks & trained model (.pkl)
│   ├── ml_model.ipynb            # Feature engineering, model training
│   ├── diabetesproject.ipynb     # EDA and data preparation
│   └── diabetes_readmission.pkl    # Serialised final model (XGBoost)
│
├── Streamlit/                  # Streamlit application
│   └── interface.py                # Streamlit deployment interface
│
├── sql/                        # Analytical SQL queries
│   └── queries.sql                 # 10 analytical SQL queries
│
├── powerbi/                    # Interactive Power BI dashboard
│   └── dashboard.pbix              # Interactive Power BI dashboard
│
├── tests/                      # Unit tests for model and pipeline
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

---

## 🔄 Workflow

### 1. Data Preparation
The raw dataset contains common clinical data quality issues that required systematic remediation before modelling:
- Replaced placeholder missing values (`?`) and null entries across categorical and numeric fields.
- Engineered features from diagnosis codes using the IDS mapping reference.
- Addressed skewness and outliers in high-variance utilisation features (e.g., outpatient visits capped at 42, emergency visits at 76).
- Produced two purpose-built output datasets:
  - **`diabetes_data_ml.csv`** — fully encoded, for model training
  - **`diabetic_data_clean.csv`** — categorical labels preserved, for Power BI

### 2. Exploratory Data Analysis
Conducted univariate, bivariate, and multivariate analysis to characterise the patient population and identify drivers of readmission risk.

**Key population findings:**
| Metric | Value |
|---|---|
| Median hospital stay | 4 days (IQR: 2–6) |
| Avg. medications per encounter | 16 (max: 81) |
| Overall readmission rate | ~16% |
| Dominant admission source | Emergency |
| Highest-risk age group | 60–80 years |

Visualisations included distribution histograms, grouped bar plots by readmission status, boxplots for utilisation outlier detection, and a full correlation heatmap across numeric features.

---

### 3. Modelling & Evaluation

The final deployed model is **XGBoost Tuned (no SMOTE)**. 

#### Results Summary
| Model | Configuration | Test AUC |
|---|---|---|
| XGBoost | Tuned (no SMOTE) | 0.667 |
| Logistic Regression | Balanced (no SMOTE) | 0.650 |
| Logistic Regression | Balanced + SMOTE | 0.646 |

#### Algorithm Comparison & Model Selection Rationale
Initially, Logistic Regression (class_weight='balanced') + SMOTE was selected due to its perceived comparable performance and high interpretability. However, fixing a bug in the AUC calculation revealed a meaningful performance gap. With the corrected AUC metric, XGBoost Tuned achieves an AUC of 0.667, outperforming the Logistic Regression configurations (0.646–0.650). Based on this genuine performance advantage, the final model choice was revised to XGBoost Tuned. 

Additionally, SMOTE was tested across 6 configurations covering every model family. In every case, including XGBoost, SMOTE failed to improve the AUC over the non-SMOTE equivalent. Thus, SMOTE was excluded from the final pipeline.

#### Interpretability (SHAP)
With the shift to XGBoost, SHAP (`TreeExplainer`) is utilized for model interpretability. SHAP analysis performed independently on both the older Logistic Regression model and the new XGBoost model agreed strongly on the top predictors: `total_visits` and `discharge_disposition_id_Transferred to another facility`. The feature `A1Cresult_Not Measured` also consistently appears as a top driver in both. This cross-model agreement is a strong signal that these features represent genuine clinical drivers of readmission risk.

#### Benchmark Comparison
The performance of the final XGBoost model aligns closely with published benchmarks on the UCI Diabetes 130-US Hospitals dataset. Independent published studies typically report XGBoost achieving an AUC in the 0.63–0.67 range. The corrected Test AUC of 0.667 for this project's XGBoost pipeline places it at the upper end of these established benchmarks.

#### Known Limitations
* **Measurement Sparsity:** `max_glu_serum` was not measured in 94.8% of cases, and `A1Cresult` was not measured in 83.1% of cases. 
* **Signal in Missingness:** SHAP analysis confirmed that the "Not Measured" status for these tests carries predictive signal in itself, separate from the actual measured values.
* **Predictive Ceiling:** Even with the best-performing model, AUC (~0.67) remains modest, reflecting the genuine difficulty of predicting readmission from structured EHR data alone.
* **Class Imbalance:** The positive class (30-day readmission) represents roughly 11-16% of encounters.

---

### 4. Deployment

The final XGBoost model is serialised to `diabetes_readmission.pkl` and served via a Streamlit interface that accepts structured patient feature inputs and returns a readmission probability prediction.

**Run locally:**
```bash
# Clone and set up environment
git clone https://github.com/kaizen105/Diabetes-analysis-project.git
cd Diabetes-analysis-project

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Launch app
streamlit run Streamlit/interface.py
```

**Live deployment:** [diabetes-analysis-project.streamlit.app](https://diabetes-analysis-project-mfz5kcrwnusng4wztuzydk.streamlit.app/)

---

### 5. SQL Insights

Ten analytical SQL queries were written against the cleaned dataset to surface patterns that predictive models alone cannot communicate to business stakeholders.

**Key findings:**
| Question | Finding |
|---|---|
| Overall readmission rate | ~16% of encounters resulted in readmission |
| Highest-risk age group | 60–80 years — disproportionate share of readmissions |
| Diagnosis association | Circulatory and respiratory primary diagnoses strongly correlated with readmission |
| Admission source | Emergency admissions account for the largest proportion of readmitted patients |
| Gender distribution | Female ~54%, Male ~46% |
| Insulin usage | Patients with adjusted insulin regimens showed elevated readmission rates |

---

### 6. Dashboarding

An interactive Power BI dashboard provides stakeholder-facing visualisation of the dataset and SQL-derived insights. It is designed for clinical operations teams, not data practitioners.

**Dashboard views:**
- Patient demographics breakdown (age, gender, race)
- Readmission trends by age group, diagnosis category, and admission source
- Medication utilisation distribution
- Hospital stay duration analysis
- Insulin usage vs. readmission rate cross-tabulation

Dashboard file: `powerbi/dashboard.pbix`

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| **Data Processing** | Python, Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost, SHAP, Joblib |
| **Visualisation (EDA)** | Matplotlib, Seaborn, Plotly |
| **Deployment** | Streamlit |
| **Analytical Queries** | SQL |
| **Business Intelligence** | Microsoft Power BI |

---

## 🔮 Future Improvements

- **Feature engineering** — Consolidate sparse diagnosis codes into ICD chapter groupings; re-evaluate medication features as binary flags rather than counts
- **Advanced modelling** — Calibrated probability outputs via Platt scaling
- **Pipeline automation** — End-to-end sklearn `Pipeline` objects to prevent data leakage during cross-validation
- **Cloud deployment** — Containerise Streamlit app via Docker, deploy to AWS ECS or Azure App Service
- **Monitoring** — Add data drift detection for production model performance tracking
