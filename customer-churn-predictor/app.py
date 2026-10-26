import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ChurnAI | Customer Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------------------------

MODEL_PATH = Path(__file__).resolve().parent / "best_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"Unable to load model: {e}")


# --------------------------------------------------
# 3. DARK GREEN AND BEIGE THEME
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #F5F0E5;
        color: #183D32;
    }

    [data-testid="stSidebar"] {
        background-color: #123C30;
    }

    [data-testid="stSidebar"] * {
        color: #F5F0E5 !important;
    }

    h1, h2, h3 {
        color: #174C3A !important;
    }

    p, label, .stMarkdown {
        color: #294B3D;
    }

    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #D9CDB7;
        padding: 18px;
        border-radius: 12px;
    }

    div.stButton > button {
        background-color: #174C3A;
        color: white;
        border: none;
        border-radius: 9px;
        padding: 12px 24px;
        font-weight: bold;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #24664B;
        color: white;
        border: 1px solid #C9B895;
    }

    div[data-testid="stForm"] {
        background-color: #FFFFFF;
        border: 1px solid #E0D5C1;
        border-radius: 14px;
        padding: 20px;
    }

    .hero {
        background: linear-gradient(120deg, #123C30, #286348);
        padding: 28px;
        border-radius: 16px;
        margin-bottom: 22px;
    }

    .hero h1 {
        color: #F5F0E5 !important;
        margin-bottom: 8px;
    }

    .hero p {
        color: #E7E1D2 !important;
        margin-bottom: 0;
    }

    .result-card {
        background-color: #FFFFFF;
        border: 1px solid #D9CDB7;
        border-radius: 14px;
        padding: 22px;
        margin-top: 10px;
    }

    .footer {
        text-align: center;
        color: #64796C;
        padding: 25px;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 4. SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("# 🌿 ChurnAI")
    st.caption("CUSTOMER INTELLIGENCE PLATFORM")
    st.divider()

    st.subheader("Model Information")

    if MODEL_LOADED:
        st.success("Model loaded successfully")
        st.write("*Algorithm:* Logistic Regression")
        st.write("*Input features:* 21")
    else:
        st.error("Model unavailable")

    st.divider()

    st.subheader("How to Use")
    st.markdown("""
    1. Enter customer details.
    2. Review account information.
    3. Click *Predict Churn*.
    4. Review the prediction and probability.
    """)

    st.divider()
    st.caption("ChurnAI • ML Customer Analytics")


# --------------------------------------------------
# 5. MAIN HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>🌿 Customer Churn Prediction</h1>
    <p>Understand customer behaviour and identify potential churn risks.</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 6. CUSTOMER INPUT FORM
# --------------------------------------------------

with st.form("customer_form"):

    st.subheader("👤 Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen", ["No", "Yes"]
        )

        partner = st.selectbox(
            "Has Partner", ["No", "Yes"]
        )

    with col2:
        dependents = st.selectbox(
            "Has Dependents", ["No", "Yes"]
        )

        tenure = st.slider(
            "Tenure (months)", 0, 72, 12
        )

        phone_service = st.selectbox(
            "Phone Service", ["No", "Yes"]
        )

    with col3:
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No phone service", "No", "Yes"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

    st.divider()
    st.subheader("🛡️ Services and Protection")

    col1, col2, col3 = st.columns(3)

    with col1:
        online_security = st.selectbox(
            "Online Security",
            ["No", "No internet service", "Yes"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["No", "No internet service", "Yes"]
        )

    with col2:
        device_protection = st.selectbox(
            "Device Protection",
            ["No", "No internet service", "Yes"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["No", "No internet service", "Yes"]
        )

    with col3:
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "No internet service", "Yes"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "No internet service", "Yes"]
        )

    st.divider()
    st.subheader("💳 Account Information")

    col1, col2 = st.columns(2)

    with col1:
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing", ["No", "Yes"]
        )

    with col2:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=70.0,
            step=5.0
        )

    # Feature engineering based on the displayed notebook formula
    total_charges = monthly_charges * tenure

    # The notebook uses tenure_group values 0, 1, 2, 3
    if tenure <= 12:
        tenure_group = 0
    elif tenure <= 24:
        tenure_group = 1
    elif tenure <= 48:
        tenure_group = 2
    else:
        tenure_group = 3

    # The notebook shows avg_monthly_per_tenure = TotalCharges / (tenure + 1)
    avg_monthly_per_tenure = (
        total_charges / (tenure + 1)
        if tenure >= 0 else 0.0
    )

    submitted = st.form_submit_button(
        "🌿 Predict Churn",
        use_container_width=True
    )


# --------------------------------------------------
# 7. PREPARE INPUT DATA
# --------------------------------------------------

if submitted and MODEL_LOADED:

    # NOTE:
    # These numeric category mappings must match the exact
    # LabelEncoder mappings used during model training.

    binary_mapping = {
        "No": 0,
        "Yes": 1,
        "Female": 0,
        "Male": 1
    }

    service_mapping = {
        "No": 0,
        "Yes": 1,
        "No phone service": 2,
        "No internet service": 2
    }

    contract_mapping = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
    }

    payment_mapping = {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3
    }

    input_data = {
        "gender": binary_mapping[gender],
        "SeniorCitizen": binary_mapping[senior_citizen],
        "Partner": binary_mapping[partner],
        "Dependents": binary_mapping[dependents],
        "tenure": tenure,
        "PhoneService": binary_mapping[phone_service],
        "MultipleLines": service_mapping[multiple_lines],
        "InternetService": {
            "DSL": 0,
            "Fiber optic": 1,
            "No": 2
        }[internet_service],
        "OnlineSecurity": service_mapping[online_security],
        "OnlineBackup": service_mapping[online_backup],
        "DeviceProtection": service_mapping[device_protection],
        "TechSupport": service_mapping[tech_support],
        "StreamingTV": service_mapping[streaming_tv],
        "StreamingMovies": service_mapping[streaming_movies],
        "Contract": contract_mapping[contract],
        "PaperlessBilling": binary_mapping[paperless_billing],
        "PaymentMethod": payment_mapping[payment_method],
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "tenure_group": tenure_group,
        "avg_monthly_per_tenure": avg_monthly_per_tenure
    }

    # Match the feature names and order used by the model
    feature_order = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "tenure_group",
        "avg_monthly_per_tenure"
    ]

    X_new = pd.DataFrame(
        [[input_data[col] for col in feature_order]],
        columns=feature_order
    )

    # --------------------------------------------------
    # 8. PREDICTION
    # --------------------------------------------------

    st.divider()
    st.subheader("🎯 Prediction Results")

    try:
        prediction = model.predict(X_new)[0]

        probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_new)[0]
            classes = list(model.classes_)

            if 1 in classes:
                probability = float(
                    probabilities[classes.index(1)]
                )

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if int(prediction) == 1:
                st.error("⚠️ Potential Churn")
                st.write(
                    "The model predicts that this customer "
                    "may leave the service."
                )
            else:
                st.success("✅ Likely to Stay")
                st.write(
                    "The model predicts that this customer "
                    "may continue using the service."
                )

        with result_col2:
            if probability is not None:
                st.metric(
                    "Predicted Churn Probability",
                    f"{probability:.1%}"
                )
                st.progress(probability)
            else:
                st.info(
                    "This model does not provide probability estimates."
                )

        st.divider()
        st.subheader("📊 Customer Snapshot")

        m1, m2, m3 = st.columns(3)

        m1.metric("Tenure", f"{tenure} months")
        m2.metric("Monthly Charges", f"${monthly_charges:.2f}")
        m3.metric("Total Charges", f"${total_charges:.2f}")

        st.caption(
            "Predictions are estimates from the trained model, "
            "not guarantees of customer behaviour."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write(
            "Check the category encodings, feature order, "
            "and scikit-learn version used to train the model."
        )


# --------------------------------------------------
# 9. FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    ChurnAI | Customer Intelligence Platform<br>
    Built with Python, Streamlit and Machine Learning
</div>
""", unsafe_allow_html=True)