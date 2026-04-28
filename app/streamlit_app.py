"""
app/streamlit_app.py
--------------------
Interactive Streamlit application for predicting customer churn.

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Path setup ────────────────────────────────────────────────────────────
APP_DIR     = os.path.dirname(__file__)
ROOT_DIR    = os.path.abspath(os.path.join(APP_DIR, ".."))
SRC_DIR     = os.path.join(ROOT_DIR, "src")
MODEL_PATH  = os.path.join(ROOT_DIR, "models", "churn_model.pkl")
VIZ_DIR     = os.path.join(ROOT_DIR, "visualization", "output")

sys.path.insert(0, SRC_DIR)

# PAGE CONFIG
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CUSTOM CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #f1f5f9; }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border-left: 5px solid #2563eb;
        margin-bottom: 12px;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #1e40af; }
    .metric-label { font-size: 0.85rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }

    .risk-high {
        background: #fef2f2;
        border-left-color: #dc2626;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 3px rgba(220,38,38,0.15);
    }
    .risk-low {
        background: #f0fdf4;
        border-left-color: #16a34a;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 3px rgba(22,163,74,0.15);
    }
    .risk-label { font-size: 1.6rem; font-weight: 700; }

    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 2px solid #e2e8f0;
    }

    .stButton > button {
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 2.5rem;
        border: none;
        font-size: 1rem;
        transition: background 0.2s;
    }
    .stButton > button:hover { background-color: #1d4ed8; }

    .insight-box {
        background: #eff6ff;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 0.92rem;
        color: #1e3a8a;
        border-left: 3px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)


# LOAD MODEL
@st.cache_resource
def load_bundle():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


# PREDICTION HELPER
def predict_single(features: dict, bundle: dict) -> tuple[float, str]:
    """Build a one-row DataFrame and return (probability, risk_label)."""
    from data_preprocessing import clean_data, encode_features, scale_features
    from feature_engineering import engineer_features

    df = pd.DataFrame([features])
    df["Churn"] = "No"  # dummy target

    df = clean_data(df)
    df = engineer_features(df)
    df, _ = encode_features(df)

    feature_names = bundle["feature_names"]
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_names]
    if "Churn" in X.columns:
        X = X.drop(columns=["Churn"])

    scaler = bundle["scaler"]
    scale_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges", "AverageChargePerMonth"]
                  if c in X.columns]
    X = X.copy()
    X[scale_cols] = scaler.transform(X[scale_cols])

    prob = bundle["model"].predict_proba(X)[0, 1]
    label = "High Risk" if prob >= 0.5 else "Low Risk"
    return float(prob), label


# SIDEBAR
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/signal.png", width=72)
    st.title("📡 Churn Predictor")
    st.caption("Telecom Customer Intelligence")
    st.markdown("---")
    st.markdown("**About this tool**")
    st.info(
        "Enter customer information to predict the probability "
        "that this customer will leave (churn) in the near future."
    )
    page = st.radio("Navigate", ["Predict Churn", "EDA Visualizations", "ℹ️ About"])


# PAGE: PREDICT CHURN
if "Predict" in page:
    st.title("Customer Churn Prediction")
    st.markdown("Fill in the customer profile below and click **Predict**.")

    bundle = load_bundle()
    if bundle is None:
        st.error(
            "⚠️ Model not found. Please run the training pipeline first:\n\n"
            "```bash\npython src/train_model.py\n```"
        )
        st.stop()

    # ── Input form ──────────────────────────────────────────────────────
    with st.form("churn_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown('<div class="section-title">👤 Demographics</div>', unsafe_allow_html=True)
            gender        = st.selectbox("Gender", ["Male", "Female"])
            senior        = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner       = st.selectbox("Has Partner", ["Yes", "No"])
            dependents    = st.selectbox("Has Dependents", ["Yes", "No"])

        with c2:
            st.markdown('<div class="section-title">📋 Account Info</div>', unsafe_allow_html=True)
            tenure        = st.slider("Tenure (months)", 0, 72, 12)
            contract      = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            payment       = st.selectbox("Payment Method", [
                                "Electronic check", "Mailed check",
                                "Bank transfer (automatic)", "Credit card (automatic)"])
            paperless     = st.selectbox("Paperless Billing", ["Yes", "No"])
            monthly       = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)

        with c3:
            st.markdown('<div class="section-title">🌐 Services</div>', unsafe_allow_html=True)
            phone         = st.selectbox("Phone Service", ["Yes", "No"])
            multi_lines   = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet      = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            security      = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            backup        = st.selectbox("Online Backup",   ["Yes", "No", "No internet service"])
            device_prot   = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support  = st.selectbox("Tech Support",    ["Yes", "No", "No internet service"])
            streaming_tv  = st.selectbox("Streaming TV",    ["Yes", "No", "No internet service"])
            streaming_mov = st.selectbox("Streaming Movies",["Yes", "No", "No internet service"])

        predict_btn = st.form_submit_button("🔍 Predict Churn", use_container_width=True)

    # ── Prediction ──────────────────────────────────────────────────────
    if predict_btn:
        total_charges = monthly * tenure
        features = dict(
            gender=gender, SeniorCitizen=1 if senior == "Yes" else 0,
            Partner=partner, Dependents=dependents,
            tenure=tenure, PhoneService=phone, MultipleLines=multi_lines,
            InternetService=internet, OnlineSecurity=security,
            OnlineBackup=backup, DeviceProtection=device_prot,
            TechSupport=tech_support, StreamingTV=streaming_tv,
            StreamingMovies=streaming_mov, Contract=contract,
            PaperlessBilling=paperless, PaymentMethod=payment,
            MonthlyCharges=monthly, TotalCharges=str(total_charges),
        )

        with st.spinner("Calculating churn probability …"):
            prob, label = predict_single(features, bundle)

        st.markdown("---")
        r1, r2, r3 = st.columns([2, 1, 1])

        risk_class = "risk-high" if label == "High Risk" else "risk-low"
        risk_icon  = "🔴" if label == "High Risk" else "🟢"
        risk_color = "#dc2626" if label == "High Risk" else "#16a34a"

        with r1:
            st.markdown(f"""
            <div class="{risk_class}">
                <div class="metric-label">Prediction Result</div>
                <div class="risk-label" style="color:{risk_color}">{risk_icon} {label}</div>
                <div style="font-size:0.9rem;color:#475569;margin-top:6px;">
                    Churn Probability: <strong>{prob:.1%}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.metric("Churn Probability", f"{prob:.1%}",
                      delta=f"{prob - 0.26:.1%} vs avg",
                      delta_color="inverse")

        with r3:
            st.metric("Retention Probability", f"{1 - prob:.1%}")

        # Gauge bar
        st.markdown("**Risk Gauge**")
        gauge_col, _ = st.columns([3, 1])
        with gauge_col:
            fig_g, ax_g = plt.subplots(figsize=(6, 0.6))
            fig_g.patch.set_alpha(0)
            ax_g.set_facecolor("#e5e7eb")
            ax_g.barh(0, prob, color=risk_color, height=0.6)
            ax_g.barh(0, 1 - prob, left=prob, color="#e5e7eb", height=0.6)
            ax_g.axvline(0.5, color="#475569", linewidth=1.5, linestyle="--")
            ax_g.set_xlim(0, 1)
            ax_g.set_ylim(-0.5, 0.5)
            ax_g.axis("off")
            ax_g.text(prob / 2, 0, f"{prob:.1%}", ha="center", va="center",
                      color="white", fontweight="bold", fontsize=11)
            st.pyplot(fig_g, use_container_width=True)
            plt.close(fig_g)

        # Recommendations
        st.markdown("---")
        st.markdown("**💡 Recommended Retention Actions**")
        recs = []
        if contract == "Month-to-month":
            recs.append("Offer a discounted annual or two-year contract.")
        if monthly > 70:
            recs.append("Propose a loyalty discount or bundled plan to reduce monthly cost.")
        if tenure < 12:
            recs.append("Enroll in new customer onboarding program to build engagement.")
        if tech_support == "No" and internet != "No":
            recs.append("Offer complimentary TechSupport trial for 3 months.")
        if security == "No" and internet != "No":
            recs.append("Highlight Online Security benefits; offer free trial.")
        if not recs:
            recs.append("Customer is stable. Continue standard engagement touchpoints.")
        for rec in recs:
            st.markdown(f'<div class="insight-box">✅ {rec}</div>', unsafe_allow_html=True)


# PAGE: EDA VISUALIZATIONS
elif "EDA" in page:
    st.title("Exploratory Data Analysis")
    st.markdown("Pre-generated visualizations from the training dataset.")

    plots = {
        "01_churn_distribution.png":          "Churn Distribution",
        "02_churn_by_contract.png":            "Churn by Contract Type",
        "03_churn_vs_monthly_charges.png":     "Monthly Charges vs Churn",
        "04_churn_vs_tenure.png":              "Tenure vs Churn",
        "05_correlation_heatmap.png":          "Correlation Heatmap",
        "06_churn_by_internet_service.png":    "Internet Service vs Churn",
        "roc_curves_comparison.png":           "ROC Curves — Model Comparison",
    }

    found = False
    for fname, title in plots.items():
        path = os.path.join(VIZ_DIR, fname)
        if os.path.exists(path):
            found = True
            st.subheader(title)
            st.image(path, use_column_width=True)
            st.markdown("---")

    if not found:
        st.warning(
            "No visualizations found. Run the training pipeline first:\n\n"
            "```bash\npython src/train_model.py\n```"
        )


# PAGE: ABOUT
else:
    st.title("ℹ️ About This Project")
    st.markdown("""
    ## Telecom Customer Churn Prediction

    This application uses machine learning to predict whether a telecom customer is
    likely to cancel their service (churn) based on account and usage data.

    ### Models Used
    | Model | Description |
    |-------|-------------|
    | **Logistic Regression** | Baseline linear model |
    | **Random Forest** | Ensemble of decision trees |
    | **XGBoost / GBM** | Gradient boosted trees (best performer) |

    The model with the highest **ROC-AUC** score is selected automatically.

    ### Dataset
    - **Telco Customer Churn** dataset (Kaggle)
    - ~7,000 customers with 20+ features

    ### Tech Stack
    - **Python** · Pandas · NumPy · Scikit-learn · XGBoost
    - **Streamlit** for deployment
    - **Matplotlib / Seaborn** for visualizations

    ### Key Churn Drivers
    - Contract type (month-to-month = highest risk)
    - Monthly charges (higher = more likely to churn)
    - Tenure (newer customers churn more)
    - Internet service type
    - Lack of online security or tech support

    """)
