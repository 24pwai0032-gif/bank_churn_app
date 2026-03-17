"""
app.py – Streamlit frontend for Bank Churn Prediction
"""

import streamlit as st
import requests

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="centered",
)

API_URL = "http://127.0.0.1:8000/predict"

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("🏦 Bank Customer Churn Predictor")
st.markdown(
    "Fill in the customer details below and click **Predict** to find out "
    "whether this customer is likely to churn."
)
st.divider()

# ─────────────────────────────────────────────
# Input form
# ─────────────────────────────────────────────
with st.form("churn_form"):
    st.subheader("📋 Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input(
            "Credit Score", min_value=300, max_value=900, value=650, step=1,
            help="Customer's credit score (300–900)"
        )
        geography = st.selectbox(
            "Geography", options=["France", "Germany", "Spain"],
            help="Country where the customer is based"
        )
        gender = st.selectbox("Gender", options=["Male", "Female"])
        age = st.number_input(
            "Age", min_value=18, max_value=100, value=40, step=1
        )
        tenure = st.slider(
            "Tenure (years)", min_value=0, max_value=10, value=5,
            help="Number of years with the bank"
        )

    with col2:
        balance = st.number_input(
            "Account Balance ($)", min_value=0.0, max_value=300000.0,
            value=50000.0, step=500.0
        )
        num_products = st.selectbox(
            "Number of Products", options=[1, 2, 3, 4], index=1,
            help="Number of bank products the customer uses"
        )
        has_cr_card = st.radio(
            "Has Credit Card?", options=["Yes", "No"],
            horizontal=True
        )
        is_active = st.radio(
            "Is Active Member?", options=["Yes", "No"],
            horizontal=True
        )
        salary = st.number_input(
            "Estimated Salary ($)", min_value=0.0, max_value=500000.0,
            value=75000.0, step=1000.0
        )

    st.divider()
    submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

# ─────────────────────────────────────────────
# On submit
# ─────────────────────────────────────────────
if submitted:
    payload = {
        "CreditScore":     int(credit_score),
        "Geography":       geography,
        "Gender":          gender,
        "Age":             int(age),
        "Tenure":          int(tenure),
        "Balance":         float(balance),
        "NumOfProducts":   int(num_products),
        "HasCrCard":       1 if has_cr_card == "Yes" else 0,
        "IsActiveMember":  1 if is_active   == "Yes" else 0,
        "EstimatedSalary": float(salary),
    }

    with st.spinner("Contacting prediction API…"):
        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()

            prediction   = data["prediction"]
            probability  = data["churn_probability"]
            label        = data["result"]

            st.divider()
            st.subheader("📊 Prediction Result")

            if prediction == 1:
                st.error(f"### ⚠️  {label}  —  Customer is likely to CHURN")
            else:
                st.success(f"### ✅  {label}  —  Customer is likely to STAY")

            col_a, col_b = st.columns(2)
            col_a.metric("Prediction", label)
            col_b.metric("Churn Probability", f"{probability * 100:.1f}%")

            # Visual probability gauge
            st.progress(probability, text=f"Churn risk: {probability * 100:.1f}%")

            with st.expander("🔍 View Payload Sent to API"):
                st.json(payload)

            with st.expander("📬 Full API Response"):
                st.json(data)

        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Cannot connect to the backend API.\n\n"
                "Make sure the FastAPI server is running:\n"
                "```\ncd backend\nuvicorn app.main:app --reload\n```"
            )
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ API error: {e.response.status_code} — {e.response.text}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

# ─────────────────────────────────────────────
# Sidebar info
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        "This app uses a **Random Forest** model trained on the "
        "Bank Churn Modelling dataset to predict whether a customer "
        "will leave the bank.\n\n"
        "**API:** FastAPI running on `http://127.0.0.1:8000`\n\n"
        "**Docs:** [/docs](http://127.0.0.1:8000/docs)"
    )
    st.divider()
    st.markdown("**Model Features Used:**")
    st.markdown(
        "- CreditScore\n- Geography\n- Gender\n- Age\n- Tenure\n"
        "- Balance\n- NumOfProducts\n- HasCrCard\n- IsActiveMember\n"
        "- EstimatedSalary"
    )
