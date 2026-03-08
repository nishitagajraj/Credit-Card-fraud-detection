# ============================================================
# Credit Card Fraud Detection - Streamlit App
# Author: Nishita
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ── Page config ──
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="wide"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 10px; border-radius: 8px; }
    .fraud-box {
        background-color: #ff4b4b22;
        border: 2px solid #ff4b4b;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .safe-box {
        background-color: #00c85322;
        border: 2px solid #00c853;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.title("💳 Credit Card Fraud Detection")
st.markdown("**ML-powered fraud detection using Logistic Regression, Random Forest & XGBoost**")
st.divider()

# ── Sidebar ──
st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio("Go to", ["🔍 Predict Transaction", "📊 Model Performance", "ℹ️ About"])

# ── Load model ──
@st.cache_resource
def load_model():
    if os.path.exists("model/best_model.pkl"):
        model = joblib.load("model/best_model.pkl")
        features = joblib.load("model/feature_columns.pkl")
        return model, features
    return None, None

model, feature_cols = load_model()

# ════════════════════════════════════════════
# PAGE 1: PREDICT
# ════════════════════════════════════════════
if page == "🔍 Predict Transaction":
    st.subheader("🔍 Check a Transaction for Fraud")

    if model is None:
        st.warning("⚠️ Model not found. Please run `train.py` first to train and save the model.")
    else:
        st.markdown("Adjust the transaction features below and click **Predict**.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Transaction Details**")
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=25000.0, value=100.0, step=0.01)
            time = st.number_input("Time (seconds from first transaction)", min_value=0.0, max_value=200000.0, value=50000.0)

        with col2:
            st.markdown("**PCA Features V1–V14**")
            v_vals_1 = {}
            for i in range(1, 15):
                v_vals_1[f'V{i}'] = st.slider(f"V{i}", -30.0, 30.0, 0.0, 0.1)

        with col3:
            st.markdown("**PCA Features V15–V28**")
            v_vals_2 = {}
            for i in range(15, 29):
                v_vals_2[f'V{i}'] = st.slider(f"V{i}", -30.0, 30.0, 0.0, 0.1)

        if st.button("🚀 Predict", use_container_width=True):
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()

            input_data = {**v_vals_1, **v_vals_2}
            input_data['Amount_Scaled'] = (amount - 88.35) / 250.12
            input_data['Time_Scaled'] = (time - 94813) / 47488

            # Reorder to match training features
            input_df = pd.DataFrame([input_data])
            for col in feature_cols:
                if col not in input_df.columns:
                    input_df[col] = 0.0
            input_df = input_df[feature_cols]

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            st.divider()
            if prediction == 1:
                st.markdown(f"""
                <div class="fraud-box">
                    <h2>🚨 FRAUDULENT TRANSACTION DETECTED</h2>
                    <h3>Fraud Probability: {probability*100:.2f}%</h3>
                    <p>This transaction has been flagged as potentially fraudulent. Immediate action recommended.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="safe-box">
                    <h2>✅ LEGITIMATE TRANSACTION</h2>
                    <h3>Fraud Probability: {probability*100:.2f}%</h3>
                    <p>This transaction appears to be legitimate.</p>
                </div>
                """, unsafe_allow_html=True)

# ════════════════════════════════════════════
# PAGE 2: MODEL PERFORMANCE
# ════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.subheader("📊 Model Performance Dashboard")

    plots = {
        "Class Distribution": "plots/class_distribution.png",
        "Amount Distribution": "plots/amount_distribution.png",
        "Confusion Matrices": "plots/confusion_matrices.png",
        "ROC Curves": "plots/roc_curves.png",
        "Model Comparison": "plots/model_comparison.png",
        "Feature Importance": "plots/feature_importance.png",
        "Correlation Heatmap": "plots/correlation_heatmap.png",
    }

    available = {k: v for k, v in plots.items() if os.path.exists(v)}

    if not available:
        st.warning("⚠️ No plots found. Please run `train.py` first to generate them.")
    else:
        tabs = st.tabs(list(available.keys()))
        for tab, (name, path) in zip(tabs, available.items()):
            with tab:
                img = Image.open(path)
                st.image(img, use_column_width=True)

# ════════════════════════════════════════════
# PAGE 3: ABOUT
# ════════════════════════════════════════════
elif page == "ℹ️ About":
    st.subheader("ℹ️ About This Project")
    st.markdown("""
    ### 💳 Credit Card Fraud Detection using Machine Learning

    **Problem Statement:**  
    Credit card fraud causes billions in losses annually. This project builds an ML pipeline to detect fraudulent transactions in real-time using the Kaggle Credit Card Fraud dataset.

    **Dataset:**
    - 284,807 transactions over 2 days
    - Only 492 (0.17%) are fraudulent — highly imbalanced
    - Features V1–V28 are PCA-transformed for privacy

    **Key Challenges Addressed:**
    - ⚖️ **Class Imbalance** — handled using SMOTE (Synthetic Minority Oversampling)
    - 📏 **Feature Scaling** — StandardScaler on Amount and Time
    - 🎯 **Metric Selection** — ROC-AUC instead of accuracy (due to imbalance)

    **Models Trained:**
    | Model | Description |
    |-------|-------------|
    | Logistic Regression | Baseline linear model |
    | Random Forest | Ensemble of decision trees |
    | XGBoost | Gradient boosted trees (typically best) |

    **Tech Stack:**
    `Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `imbalanced-learn` · `Streamlit` · `Matplotlib` · `Seaborn`

    ---
    **Author:** Nishita | VIT Bhopal University  
    **GitHub:** [github.com/nishitagajraj](https://github.com/nishitagajraj)
    """)