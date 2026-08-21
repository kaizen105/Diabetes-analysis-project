# Diabetes Readmission Prediction – Machine Learning Project

## Project Overview

This project implements a **machine learning pipeline** to predict hospital readmission for diabetes patients within 30 days. The model leverages patient medical data, handles class imbalance, and provides predictions through a **Streamlit web application** for easy user interaction and clinical decision support.

## Key Features

* **Data Preprocessing:** Handles numerical and categorical features, applies scaling (`StandardScaler`) and transformation (`PowerTransformer`).
* **Machine Learning Pipeline:** Employs XGBoost within a **scikit-learn Pipeline** for robust, reproducible predictions.
* **Interactive Web App:** Streamlit dashboard for users to input patient features and receive real-time predictions.
* **Visualizations & Interpretability:** Optional charts via **Plotly** to explore feature distributions, and **SHAP (TreeExplainer)** for global and localized feature importance.

## Model Selection & Evaluation

### Results Summary
| Model | Configuration | Test AUC |
|---|---|---|
| XGBoost | Tuned (no SMOTE) | 0.667 |
| Logistic Regression | Balanced (no SMOTE) | 0.650 |
| Logistic Regression | Balanced + SMOTE | 0.646 |

### Algorithm Comparison & Model Selection Rationale
The final deployed model is **XGBoost Tuned (no SMOTE)**. 

Initially, Logistic Regression (class_weight='balanced') + SMOTE was selected as the final model due to its perceived comparable performance and high interpretability. However, a bug was subsequently identified in the AUC calculation: the `roc_auc_score()` was originally computed using hard class predictions (0/1) rather than predicted probabilities. This systematically underestimated the true AUC across all models. 

Catching and fixing this calculation error revealed a real and meaningful performance gap between the models. With the corrected AUC metric, XGBoost Tuned achieves an AUC of 0.667, outperforming the Logistic Regression configurations (0.646–0.650). Based on this genuine performance advantage, the final model choice was revised to XGBoost Tuned. 

Additionally, SMOTE was tested across 6 configurations covering every model family. In every case, including XGBoost, SMOTE failed to improve the AUC over the non-SMOTE equivalent (e.g., the XGBoost SMOTE version scored 0.644 vs. 0.667 without). Thus, SMOTE was excluded from the final pipeline. A plausible explanation is that SMOTE's linear interpolation between nearest neighbors is poorly suited to this feature space, which is overwhelmingly one-hot encoded categorical data — interpolating between two one-hot category vectors produces fractional, clinically non-sensical synthetic patients rather than realistic ones.

### Interpretability (SHAP)
With the shift to XGBoost, SHAP (`TreeExplainer`) is utilized for model interpretability instead of relying on linear coefficients. 

Notably, SHAP analysis performed independently on both the older Logistic Regression model and the new XGBoost model agreed strongly on the top predictors: `total_visits` and `discharge_disposition_id_Transferred to another facility`. The feature `A1Cresult_Not Measured` also consistently appears as a top driver in both. This cross-model agreement is a strong signal that these features represent genuine clinical drivers of readmission risk, rather than model-specific artifacts.

### Benchmark Comparison
The performance of the final XGBoost model aligns closely with published benchmarks on the UCI Diabetes 130-US Hospitals dataset. Independent published studies typically report XGBoost achieving an AUC in the 0.63–0.67 range, Logistic Regression at ~0.64, and Random Forest at ~0.63. The corrected Test AUC of 0.667 for this project's XGBoost pipeline places it at the upper end of these established benchmarks.

## Known Limitations
* **Measurement Sparsity:** Several clinical measurements have high rates of missing or unmeasured data. For instance, `max_glu_serum` was not measured in 94.8% of cases, and `A1Cresult` was not measured in 83.1% of cases. 
* **Signal in Missingness:** SHAP analysis confirmed that the "Not Measured" status for these tests carries predictive signal in itself, separate from the actual measured values. While the model successfully learns from this missingness, it suggests that testing protocols or clinical triage decisions are acting as latent indicators of patient condition.
* **Predictive Ceiling:** Even with corrected metrics and the best-performing model, AUC (~0.67) remains modest by absolute standards. Consistent with published literature on this dataset, this reflects the genuine difficulty of predicting readmission from structured EHR data alone, absent behavioral, social, or continuous physiological signals — not a modeling deficiency.
* **Class Imbalance:** The positive class (30-day readmission) represents roughly 11-16% of encounters. This constrains achievable precision/recall tradeoffs regardless of model choice, and motivated the use of class_weight='balanced' throughout the model comparison.

## Project Structure

```
diabetes_analysis_project/
│
├── datasets/                   # Raw and processed data (ignored in .gitignore)
├── notebook/                   # Jupyter notebooks & trained model (.pkl)
├── tests/                      # Unit tests for model and pipeline
├── Streamlit/                  # Streamlit application
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

## Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/diabetes_analysis_project.git
cd diabetes_analysis_project
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app locally:
```bash
streamlit run Streamlit/interface.py
```

## Usage
1. Enter the **patient’s medical features**.
2. Click **Predict**.
3. The app outputs the probability of hospital readmission within 30 days along with actionable clinical recommendations.

## Technologies & Libraries
* Python 3.10+
* scikit-learn, xgboost, shap, joblib, pandas, numpy
* Streamlit for frontend deployment
* Plotly for interactive visualizations

## Author
Yash Sharma – Data Analyst / Machine Learning Enthusiast
