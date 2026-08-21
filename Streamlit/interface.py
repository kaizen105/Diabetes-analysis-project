import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import urllib.request
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="DiabeTrack",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling and responsiveness
st.markdown("""
<style>
    /* Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #f8fafc;
        text-align: center;
        margin-top: 80px; 
        margin-bottom: 2rem;
    }

    /* Navbar Container */
    .nav-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 5px;
        margin-bottom: 50px;
        padding: 5px 15px;
        background: #14278b;
        border-radius: 0px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        position: relative;
        z-index: 10;
    }

    /* Navbar Button */
    .nav-button {
        padding: 12px 24px;
        border-radius: 0px;
        border: none;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        color: #f8fafc;
        background: #1b3c99;
    }
    .nav-button:hover {
        background: #3366ff;
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.4);
    }
    /* Info Card */
    .info-card {
        background: #14278b;
        color: #f8fafc;
        padding: 20px;
        border-radius: 0px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        margin: 15px 0;
        border-left: 5px solid #f39c12;
    }
    
    .home-info-card {
        min-height: 250px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    /* Metric Card */
    .metric-card {
        background: #14278b;
        color: #f8fafc;
        padding: 20px;
        border-radius: 0px;
        text-align: center;
        margin: 10px 0;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    /* Prediction Result */
    .prediction-result {
        padding: 20px;
        border-radius: 0px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 20px 0;
    }

    /* Risk Levels */
    .low-risk {
        background: #1b3c99;
        color: #a8e6cf;
        border-left: 5px solid #a8e6cf;
    }
    .medium-risk {
        background: #1b3c99;
        color: #ffd93d;
        border-left: 5px solid #ffd93d;
    }
    .high-risk {
        background: #1b3c99;
        color: #ff6b6b;
        border-left: 5px solid #ff6b6b;
    }
</style>
""", unsafe_allow_html=True)

# Navigation bar
st.markdown("""
<style>
/* Navbar container aligned to top-right */
.top-nav {
    position: absolute;
    top: -40px; 
    right: 30px;
    display: flex;
    gap: 15px;
    padding: 8px 12px;
    background: #010931;
    border-radius: 0px;
    z-index: 999;
}

/* Navbar links */
.top-nav a {
    padding: 8px 14px;
    border-radius: 0px;
    font-weight: 600;
    text-decoration: none;
    color: #f8fafc;
    background: #14278b;
    transition: all 0.3s ease;
    font-size: 0.95rem;
}

.top-nav a:hover {
    background: #3366ff;
}
</style>

<div class="top-nav">
    <a href="?page=Home">🏠 Home</a>
    <a href="?page=Prediction">🔮 Prediction</a>
    <a href="?page=Insights">📊 Insights</a>
    <a href="?page=Treatment">💊 Treatment</a>
</div>
""", unsafe_allow_html=True)

# Get directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_model_data():
    """Load the ML-ready dataset from GitHub repo."""
    try:
        url = "https://raw.githubusercontent.com/kaizen105/Diabetes-analysis-project/8eb4417d802bb66a0c8efff44f56a593d5b4ae15/datasets/diabetes_data_ml.csv"
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"❌ Error loading model data: {e}")
        return None

@st.cache_data
def load_insights_data():
    """Load the insights dataset from GitHub repo."""
    try:
        url = "https://raw.githubusercontent.com/kaizen105/Diabetes-analysis-project/8eb4417d802bb66a0c8efff44f56a593d5b4ae15/datasets/diabetic_data_clean.csv"
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"❌ Error loading insights data: {e}")
        return None

# ----------------------------
# Load model (binary pickle)
# ----------------------------
@st.cache_resource
def load_model():
    """Load the trained pipeline model from the local directory."""
    try:
        model_path = os.path.join(BASE_DIR, '../notebook/diabetes_readmission.pkl')
        model = joblib.load(model_path)

        if not hasattr(model, "predict"):
            st.error("❌ Loaded object is not a valid model.")
            return None
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Main content based on selected page
query_params = st.query_params
page = query_params.get("page", "Home")

if page == 'Home':
    st.markdown('<h1 class="main-header">🏥 Diabetes Care & Readmission Prevention Center</h1>', unsafe_allow_html=True)
    
    insights_df = load_insights_data()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="info-card home-info-card">
            <h2 style="color: ##1f4e79;">Welcome to Diabetes Care and readmission center</h2>
            <p style="font-size: 1.1rem; line-height: 1.6;">
                Our AI-powered platform helps healthcare providers and patients predict and prevent hospital readmissions 
                for diabetes patients. With cutting-edge machine learning and comprehensive care insights, we're 
                revolutionizing diabetes management.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if insights_df is not None:
            total_encounters = len(insights_df)
            total_patients = insights_df['patient_nbr'].nunique()
            readmit_30 = len(insights_df[insights_df['readmitted'] == '<30'])
            readmission_rate = (readmit_30 / total_encounters) * 100
            st.markdown(f"""
            <div class="info-card home-info-card">
                <h4>📊 Dataset Overview</h4>
                <p><strong>Total Encounters:</strong> {total_encounters:,}</p>
                <p><strong>Unique Patients:</strong> {total_patients:,}</p>
                <p><strong>30-day Readmission Rate:</strong> {readmission_rate:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("### 📈 Key Statistics")
    col1, col2, col3, col4 = st.columns(4)

    if insights_df is not None:
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{total_patients:,}</h3>
                <p>Unique Patients</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{readmit_30:,}</h3>
                <p>30-day Readmissions</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{readmission_rate:.1f}%</h3>
                <p>30-day Readmission Rate</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3 title="AUC (Area Under ROC Curve) — measures ability to correctly rank patients by readmission risk; more reliable than accuracy on imbalanced clinical data.">0.667</h3>
                <p>Test AUC</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### 🔬 Understanding Diabetes & Readmission")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>What is Diabetes?</h4>
            <p>Diabetes is a chronic condition that affects how your body processes blood sugar (glucose). 
            There are two main types:</p>
            <ul>
                <li><strong>Type 1:</strong> Autoimmune condition where the body doesn't produce insulin</li>
                <li><strong>Type 2:</strong> Body becomes resistant to insulin or doesn't produce enough</li>
            </ul>
            <p>Proper management is crucial to prevent complications and hospital readmissions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>Hospital Readmission Risk</h4>
            <p>Diabetes patients face higher readmission risks due to:</p>
            <ul>
                <li><strong>Complications:</strong> Diabetic ketoacidosis, hypoglycemia</li>
                <li><strong>Comorbidities:</strong> Heart disease, kidney problems</li>
                <li><strong>Medication Issues:</strong> Non-adherence, side effects</li>
                <li><strong>Lifestyle Factors:</strong> Diet, exercise, stress management</li>
            </ul>
            <p>Early prediction helps prevent costly and dangerous readmissions.</p>
        </div>
        """, unsafe_allow_html=True)

elif page == 'Prediction':
    st.markdown('<h1 class="main-header">🔮 Diabetes Readmission Prediction</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <p style="font-size: 1.1rem;">Enter patient information below to predict the likelihood of hospital readmission within 30 days.</p>
    </div>
    """, unsafe_allow_html=True)

    model_df = load_model_data()
    model = load_model()

    if model_df is None or model is None:
        st.stop()

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👤 Demographics")
            age_map = {
                '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
                '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
                '[80-90)': 85, '[90-100)': 95
            }
            age_val = st.select_slider('Age', options=list(age_map.values()), value=55)
            gender = st.selectbox('Gender', ['Male', 'Female'])
            gender_code = 1 if gender == "Male" else 0
            
            st.markdown("### 🏥 Clinical Data")
            time_in_hospital = st.slider("Time in Hospital (days)", 1, 15, 3)
            num_medications = st.slider("Number of Medications", 1, 30, 16)
            num_lab_procedures = st.slider("Number of Lab Procedures", 1, 50, 41)
            num_procedures = st.slider("Number of Procedures", 0, 10, 1)
            number_diagnoses = st.slider("Number of Diagnoses", 1, 10, 7)
            total_visits = st.slider("Previous Hospital Visits", 0, 20, 0)
        
        with col2:
            st.markdown("### 📝 Medical Details")
            diabetesMed = st.selectbox('Does patient have diabetes?', ['Yes', 'No'])
            insulin = st.selectbox('Insulin Usage', ['No', 'Up', 'Down', 'Steady'])
            change = st.selectbox('Change in Medication?', ['Ch', 'No'])
            admission_type = st.selectbox('Admission Type', ['Emergency', 'Elective', 'Urgent', 'Other'])
            insurance = st.selectbox("Insurance/Payer", ["Medicare (MC)", "Other"])

        predict_button = st.form_submit_button("🔍 Predict Readmission Risk", type="primary")

    if predict_button:
        all_columns = [
            'gender', 'age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
            'num_medications', 'number_diagnoses', 'total_visits', 'race_Asian',
            'race_Caucasian', 'race_Hispanic', 'race_Other', 'admission_type_id_Emergency',
            'admission_type_id_Not Available', 'discharge_disposition_id_Left AMA',
            'discharge_disposition_id_Not Available', 'discharge_disposition_id_Still patient/referred to this institution',
            'discharge_disposition_id_Transferred to another facility', 'admission_source_id_Not Available',
            'admission_source_id_Referral', 'admission_source_id_Transferred from hospital',
            'payer_code_CH', 'payer_code_CM', 'payer_code_CP', 'payer_code_DM', 'payer_code_FR',
            'payer_code_HM', 'payer_code_MC', 'payer_code_MD', 'payer_code_MP', 'payer_code_OG',
            'payer_code_OT', 'payer_code_Other', 'payer_code_PO', 'payer_code_SI', 'payer_code_SP',
            'payer_code_UN', 'payer_code_WC', 'insulin_No', 'insulin_Steady', 'insulin_Up',
            'change_No', 'diabetesMed_Yes', 'max_glu_serum_>300', 'max_glu_serum_Norm', 
            'max_glu_serum_Not Measured', 'A1Cresult_>8', 'A1Cresult_Norm', 'A1Cresult_Not Measured',
            'diag_1_Diabetes', 'diag_1_Digestive',
            'diag_1_Genitourinary', 'diag_1_Injury', 'diag_1_Musculoskeletal', 'diag_1_Neoplasms',
            'diag_1_Other', 'diag_1_Respiratory', 'diag_1_Unknown', 'diag_2_Diabetes',
            'diag_2_Digestive', 'diag_2_Genitourinary', 'diag_2_Injury', 'diag_2_Musculoskeletal',
            'diag_2_Neoplasms', 'diag_2_Other', 'diag_2_Respiratory', 'diag_2_Unknown',
            'diag_3_Diabetes', 'diag_3_Digestive', 'diag_3_Genitourinary', 'diag_3_Injury',
            'diag_3_Musculoskeletal', 'diag_3_Neoplasms', 'diag_3_Other', 'diag_3_Respiratory',
            'diag_3_Unknown'
        ]
        
        patient = {col: False for col in all_columns}
        
        patient['gender'] = gender_code
        patient['age'] = age
        patient['time_in_hospital'] = time_in_hospital
        patient['num_lab_procedures'] = num_lab_procedures
        patient['num_procedures'] = num_procedures
        patient['num_medications'] = num_medications
        patient['number_diagnoses'] = number_diagnoses
        patient['total_visits'] = total_visits

        race_mapping = {'Caucasian': 'race_Caucasian', 'Asian': 'race_Asian', 'Hispanic': 'race_Hispanic', 'Other': 'race_Other'}
        if race in race_mapping:
            patient[race_mapping[race]] = True

        if admission_type == "Emergency":
            patient['admission_type_id_Emergency'] = True
        else:
            patient['admission_type_id_Not Available'] = True

        if insurance == "Medicare (MC)":
            patient['payer_code_MC'] = True
        else:
            patient['payer_code_Other'] = True

        if has_diabetes:
            patient['diag_1_Diabetes'] = True
            patient['diag_2_Diabetes'] = True
            if on_diabetes_med:
                patient['diabetesMed_Yes'] = True
        else:
            patient['diag_1_Other'] = True

        patient['insulin_No'] = True
        patient['change_No'] = True
        patient['admission_source_id_Referral'] = True # Example default for required fields
        
        # New model missing column defaults
        patient['max_glu_serum_Not Measured'] = True
        patient['A1Cresult_Not Measured'] = True

        patient_df = pd.DataFrame([patient])
        patient_df = patient_df.astype({c: 'int64' for c in patient_df.select_dtypes('bool').columns})

        with st.spinner("Analyzing patient data..."):
            try:
                prediction = model.predict(patient_df)[0]
                probability = model.predict_proba(patient_df)[0][1]
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.stop()
            
            risk_level = "Low"
            risk_class = "low-risk"
            color = "#27ae60"
            if probability < 0.3:
                risk_level = "Low"
                risk_class = "low-risk"
                color = "#27ae60"
            elif probability < 0.7:
                risk_level = "Medium"
                risk_class = "medium-risk"
                color = "#f39c12"
            else:
                risk_level = "High"
                risk_class = "high-risk"
                color = "#e74c3c"

            st.markdown(f"""
            <div class="prediction-result {risk_class}">
                <h3>Prediction Results</h3>
                <h2 style="color:{color};">{risk_level} Risk ({probability*100:.1f}%)</h2>
                <p>Probability of 30-day readmission</p>
            </div>
            """, unsafe_allow_html=True)

            st.subheader("📋 Clinical Recommendations")
            if probability > 0.6:
                st.markdown("""
                - Enhanced discharge planning<br>
                - Early follow-up appointment<br>
                - Medication reconciliation<br>
                - Patient education on warning signs
                """)
            elif probability > 0.4:
                st.markdown("""
                - Standard discharge planning<br>
                - Routine follow-up scheduling<br>
                - Basic medication review
                """)
            else:
                st.markdown("""
                - Standard care protocols<br>
                - Routine follow-up as needed
                """)

elif page == 'Insights':
    st.markdown('<h1 class="main-header">📊 Data Insights & Analytics</h1>', unsafe_allow_html=True)
    
    insights_df = load_insights_data()
    
    if insights_df is None:
        st.error("❌ Could not load insights dataset. Please check file path.")
        st.stop()
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_encounters = len(insights_df)
    total_patients = insights_df['patient_nbr'].nunique()
    readmit_30 = len(insights_df[insights_df['readmitted'] == '<30'])
    readmit_rate = (readmit_30 / total_encounters) * 100
    avg_stay = insights_df['time_in_hospital'].mean()
    avg_age = insights_df['age'].mode()[0]
    
    with col1:
        with st.container(height=120, border=True):
            st.metric("Total Encounters", f"{total_encounters:,}")
    with col2:
        with st.container(height=120, border=True):
            st.metric("30-day Readmissions", f"{readmit_30:,}")
    with col3:
        with st.container(height=120, border=True):
            st.metric("30-day Readmission Rate", f"{readmit_rate:.1f}%")
    with col4:
        with st.container(height=120, border=True):
            st.metric("Avg Hospital Stay", f"{avg_stay:.1f} days")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            age_readmit = insights_df.groupby(['age', 'readmitted']).size().unstack(fill_value=0)
            age_readmit_pct = age_readmit.div(age_readmit.sum(axis=1), axis=0) * 100
            
            fig1 = px.bar(
                x=age_readmit_pct.index,
                y=age_readmit_pct['<30'] if '<30' in age_readmit_pct.columns else [0]*len(age_readmit_pct.index),
                title="30-day Readmission Rate by Age Group",
                labels={'x': 'Age Group', 'y': 'Readmission Rate (%)'}
            )
            fig1.update_traces(marker_color='#e74c3c')
            fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        with st.container(border=True):
            gender_counts = insights_df['gender'].value_counts()
            fig2 = px.pie(values=gender_counts.values, names=gender_counts.index, 
                          title="Patient Gender Distribution")
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig2, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            fig3 = px.histogram(insights_df, x='time_in_hospital', 
                                title="Distribution of Hospital Stay Length",
                                nbins=14)
            fig3.update_traces(marker_color='#3498db')
            fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        with st.container(border=True):
            insulin_readmit = pd.crosstab(insights_df['insulin'], insights_df['readmitted'], normalize='index') * 100
            
            fig4 = px.bar(insulin_readmit, 
                          title="Readmission Rate by Insulin Usage",
                          labels={'value': 'Percentage', 'index': 'Insulin Usage'})
            fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig4, use_container_width=True)
    
    st.subheader("📈 Demographic Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            race_counts = insights_df['race'].value_counts()
            fig5 = px.bar(x=race_counts.index, y=race_counts.values,
                          title="Patient Distribution by Race")
            fig5.update_traces(marker_color='#9b59b6')
            fig5.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig5, use_container_width=True)
    
    with col2:
        with st.container(border=True):
            admission_counts = insights_df['admission_type_id'].value_counts()
            fig6 = px.bar(x=admission_counts.index, y=admission_counts.values,
                          title="Admission Type Distribution")
            fig6.update_traces(marker_color='#f39c12')
            fig6.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig6, use_container_width=True)
            
    st.subheader("🧠 Model Feature Importance (SHAP)")
    model_df = load_model_data()
    model = load_model()
    if model_df is not None and model is not None:
        with st.spinner("Calculating SHAP Feature Importance..."):
            with st.container(border=True):
                try:
                    import shap
                    import matplotlib.pyplot as plt
                    
                    X = model_df.drop('readmitted', axis=1) if 'readmitted' in model_df.columns else model_df
                    
                    # Ensure X has all features expected by the model
                    missing_cols = ['max_glu_serum_>300', 'max_glu_serum_Norm', 'max_glu_serum_Not Measured', 
                                    'A1Cresult_>8', 'A1Cresult_Norm', 'A1Cresult_Not Measured']
                    for col in missing_cols:
                        if col not in X.columns:
                            X[col] = True if 'Not Measured' in col else False
                    
                    # Extract XGBoost step and transform sample for SHAP
                    # We'll assume it's a pipeline and XGBClassifier is the last step
                    if hasattr(model, 'steps'):
                        xgb_model = model.steps[-1][1]
                        # For TreeExplainer, we might just need the xgb model and X directly if there's no complex preprocessing 
                        # Or if we have simple one-hot encoding we can explain directly on it.
                        explainer = shap.TreeExplainer(xgb_model)
                        
                        # Use a small sample to avoid freezing the app
                        X_sample = X.sample(n=min(200, len(X)), random_state=42)
                        # Ensure features match model by transforming through the pipeline
                        import scipy.sparse
                        if len(model.steps) > 1:
                            preprocessor = model[:-1]
                            X_transformed = preprocessor.transform(X_sample)
                            if scipy.sparse.issparse(X_transformed):
                                X_transformed = X_transformed.toarray()
                            X_sample = pd.DataFrame(X_transformed, columns=xgb_model.feature_names_in_)
                        else:
                            X_sample = X_sample[xgb_model.feature_names_in_]
                        
                        # Convert to float to avoid any categorical/string plotting errors in matplotlib
                        X_sample = X_sample.astype(float)
                        
                        shap_values = explainer.shap_values(X_sample)
                        
                        # If shap_values is a list (multiclass or binary with two outputs), select the positive class
                        if isinstance(shap_values, list) and len(shap_values) == 2:
                            shap_values = shap_values[1]
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        fig.patch.set_alpha(0.0)
                        ax.set_facecolor('none')
                        ax.xaxis.label.set_color('white')
                        ax.yaxis.label.set_color('white')
                        ax.tick_params(colors='white')
                        for spine in ax.spines.values():
                            spine.set_edgecolor('white')
                            
                        shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not generate SHAP plot: {e}")
    
    st.subheader("🔍 Key Insights")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        high_risk_age = insights_df[insights_df['age'].isin(['[70-80)', '[80-90)', '[90-100)'])]
        if not high_risk_age.empty and '<30' in high_risk_age['readmitted'].unique():
            high_risk_readmit = len(high_risk_age[high_risk_age['readmitted'] == '<30']) / len(high_risk_age) * 100
        else:
            high_risk_readmit = 0
            
        st.markdown(f"""
        <div class="info-card">
            <h4>🎯 High-Risk Demographics</h4>
            <ul>
                <li>Elderly patients (70+): {high_risk_readmit:.1f}% readmission rate</li>
                <li>Average hospital stay: {avg_stay:.1f} days</li>
                <li>Most common age group: {avg_age}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with insight_col2:
        insulin_users = len(insights_df[insights_df['insulin'] != 'No'])
        insulin_pct = (insulin_users / total_patients) * 100
        
        st.markdown(f"""
        <div class="info-card">
            <h4>💊 Medication Insights</h4>
            <ul>
                <li>Insulin users: {insulin_pct:.1f}% of patients</li>
                <li>Average medications: {insights_df['num_medications'].mean():.1f}</li>
                <li>Diabetes medication: {len(insights_df[insights_df['diabetesMed'] == 'Yes']) / total_patients * 100:.1f}%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with insight_col3:
        emergency_admits = len(insights_df[insights_df['admission_type_id'] == 'Emergency'])
        emergency_pct = (emergency_admits / total_patients) * 100
        
        st.markdown(f"""
        <div class="info-card">
            <h4>🏥 Care Patterns</h4>
            <ul>
                <li>Emergency admissions: {emergency_pct:.1f}%</li>
                <li>Average lab procedures: {insights_df['num_lab_procedures'].mean():.0f}</li>
                <li>Multiple diagnoses common: {insights_df['number_diagnoses'].mean():.1f} avg</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
elif page == 'Treatment':
    st.markdown('<h1 class="main-header">💊 Diabetes Treatment & Readmission Prevention</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <p style="font-size: 1.1rem;">Explore recommended strategies for diabetes management and reducing hospital readmission risk.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("🍎 Lifestyle & Diet")
    st.markdown("""
    <ul>
        <li>Maintain a balanced diet rich in vegetables, whole grains, and lean proteins</li>
        <li>Limit sugar, refined carbs, and saturated fats</li>
        <li>Monitor carbohydrate intake to control blood sugar levels</li>
        <li>Stay hydrated and avoid sugary drinks</li>
        <li>Regular meal times to prevent glucose spikes</li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.subheader("🏃 Physical Activity")
    st.markdown("""
    <ul>
        <li>Engage in at least 150 minutes of moderate aerobic activity per week</li>
        <li>Incorporate strength training 2-3 times weekly</li>
        <li>Daily walking or light exercises help control blood sugar</li>
        <li>Always consult your doctor before starting a new exercise plan</li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.subheader("💊 Medications & Monitoring")
    st.markdown("""
    <ul>
        <li>Follow prescribed insulin or oral medication schedules strictly</li>
        <li>Monitor blood glucose levels regularly</li>
        <li>Keep track of HbA1c and other lab results</li>
        <li>Report any side effects or unusual symptoms promptly</li>
        <li>Ensure medication adherence to prevent complications</li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.subheader("🩺 Hospital Readmission Prevention")
    st.markdown("""
    <ul>
        <li>Attend follow-up appointments after discharge</li>
        <li>Educate patients and caregivers on warning signs</li>
        <li>Implement discharge planning and care coordination</li>
        <li>Maintain a structured home care plan including diet, meds, and monitoring</li>
        <li>Early intervention for any complications can prevent readmission</li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <p>💡 <strong>Tip:</strong> Personalized care plans significantly reduce 30-day readmissions. AI-powered predictions can help identify high-risk patients early.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🏥 Diabetes Care & Readmission Prevention Center | Powered by AI Healthcare Technology</p>
    <p><small>This tool is for educational purposes and should not replace professional medical advice.</small></p>
</div>
""", unsafe_allow_html=True)
