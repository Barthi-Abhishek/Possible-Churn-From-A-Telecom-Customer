"""
streamlit_app.py
----------------
Interactive web application for the Customer Churn Prediction model.
Run with:  streamlit run app/streamlit_app.py
"""

import os
import sys
import streamlit as st

# Ensure src/ is importable from the app/ directory
APP_DIR     = os.path.dirname(__file__)
SRC_DIR     = os.path.join(APP_DIR, '..', 'src')
sys.path.insert(0, SRC_DIR)

from predict import predict_churn   # noqa: E402 (intentional late import)


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="📡",
    layout="centered",
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📡 Telco Customer Churn Predictor")
st.markdown(
    "Enter the customer's account details below and click **Predict** "
    "to find out whether they are at risk of churning."
)
st.divider()


# ── Input form ─────────────────────────────────────────────────────────────────
st.subheader("Customer Profile")

col1, col2, col3 = st.columns(3)

with col1:
    gender         = st.selectbox("Gender",          ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen",  ["No", "Yes"])
    partner        = st.selectbox("Partner",          ["Yes", "No"])
    dependents     = st.selectbox("Dependents",       ["No", "Yes"])
    phone_service  = st.selectbox("Phone Service",    ["Yes", "No"])

with col2:
    tenure           = st.slider("Tenure (months)",  0, 72, 12)
    monthly_charges  = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    total_charges    = st.number_input("Total Charges ($)", min_value=0.0,
                                       value=float(monthly_charges * tenure or 1.0),
                                       step=1.0)

with col3:
    multiple_lines    = st.selectbox("Multiple Lines",    ["No", "Yes", "No phone service"])
    internet_service  = st.selectbox("Internet Service",  ["DSL", "Fiber optic", "No"])
    online_security   = st.selectbox("Online Security",   ["No", "Yes", "No internet service"])
    online_backup     = st.selectbox("Online Backup",     ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

st.divider()

col4, col5 = st.columns(2)

with col4:
    tech_support     = st.selectbox("Tech Support",      ["No", "Yes", "No internet service"])
    streaming_tv     = st.selectbox("Streaming TV",      ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies",  ["No", "Yes", "No internet service"])

with col5:
    contract          = st.selectbox("Contract",         ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing",["Yes", "No"])
    payment_method    = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"]
    )

st.divider()


# ── Prediction ─────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Churn", type="primary", use_container_width=True):

    customer = {
        'SeniorCitizen'   : 1 if senior_citizen == "Yes" else 0,
        'tenure'          : tenure,
        'MonthlyCharges'  : monthly_charges,
        'TotalCharges'    : total_charges,
        'gender'          : gender,
        'Partner'         : partner,
        'Dependents'      : dependents,
        'PhoneService'    : phone_service,
        'MultipleLines'   : multiple_lines,
        'InternetService' : internet_service,
        'OnlineSecurity'  : online_security,
        'OnlineBackup'    : online_backup,
        'DeviceProtection': device_protection,
        'TechSupport'     : tech_support,
        'StreamingTV'     : streaming_tv,
        'StreamingMovies' : streaming_movies,
        'Contract'        : contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod'   : payment_method,
    }

    with st.spinner("Running model …"):
        try:
            result = predict_churn(customer)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

    # ── Display result ─────────────────────────────────────────────────────────
    prob = result['probability']

    if result['prediction'] == 1:
        st.error(f"### ⚠️ Customer likely to CHURN")
    else:
        st.success(f"### ✅ Customer likely to STAY")

    st.metric(label="Churn Probability", value=f"{prob:.1%}")

    # Progress bar coloured by risk
    bar_color = "red" if prob > 0.5 else "green"
    st.markdown(
        f"""
        <style>
        .stProgress > div > div > div > div {{
            background-color: {bar_color};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.progress(prob)

    # Interpretation
    if prob < 0.3:
        risk_label = "🟢 Low risk"
    elif prob < 0.6:
        risk_label = "🟡 Medium risk"
    else:
        risk_label = "🔴 High risk"

    st.info(f"**Risk level:** {risk_label}  |  "
            f"**Churn probability:** {prob:.2%}")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Telco Customer Churn Predictor · Built with Streamlit & scikit-learn · "
    "Model trained on the IBM Telco Customer Churn dataset."
)
