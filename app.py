
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

model = joblib.load("models/churn_model.pkl")

df = pd.read_csv(
    "data/Telco-Customer-Churn.csv"
)

st.markdown("""
<style>

.main {
    background-color: #F5F7FB;
}

.block-container {
    padding-top: 3rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 38px;
    font-weight: 700;
    color: #17365D;
    line-height: 1.4;
}

.subtitle {
    font-size: 16px;
    color: #64748B;
}

div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #E2E8F0;
    padding: 18px;
    border-radius: 12px;
}

div.stButton > button {
    background-color: #2563EB;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 22px;
    font-weight: 600;
}

div.stButton > button:hover {
    background-color: #1D4ED8;
    color: white;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered customer retention and churn analysis</div>',
    unsafe_allow_html=True
)

st.divider()

st.subheader("Business Overview")

total_customers = len(df)

churn_count = (df["Churn"] == "Yes").sum()

retained_count = (df["Churn"] == "No").sum()

churn_rate = (churn_count / total_customers) * 100

retention_rate = (retained_count / total_customers) * 100

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

with col2:
    st.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )

with col3:
    st.metric(
        "Retention Rate",
        f"{retention_rate:.2f}%"
    )

st.divider()

st.subheader("Customer Churn Analytics")

col1, col2 = st.columns(2)

with col1:

    st.write("**Customer Churn Distribution**")

    churn_counts = df["Churn"].value_counts()

    st.bar_chart(churn_counts)


with col2:

    st.write("**Churn Rate by Contract Type (%)**")

    contract_churn = pd.crosstab(
        df["Contract"],
        df["Churn"],
        normalize="index"
    ) * 100

    st.bar_chart(contract_churn["Yes"])

st.divider()

st.subheader("Customer Information")

with st.form("customer_form"):

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Has Partner?",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Has Dependents?",
            ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

    with col2:

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

    with col3:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=70.0
        )

        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=840.0
        )

    st.write("")

    predict_button = st.form_submit_button(
        "Predict Customer Churn",
        use_container_width=True
    )


if predict_button:

    input_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    input_df = pd.DataFrame([input_data])

   
    prediction = model.predict(input_df)[0]

  
    probability = model.predict_proba(input_df)[0][1]


    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("⚠️ This customer is likely to leave!")

    else:

        st.success("✅ This customer is likely to stay!")

    st.write(
        f"**Churn Probability: {probability:.2%}**"
    )

    st.progress(float(probability))


    st.subheader("Customer Risk Analysis")

    if probability < 0.30:

        risk_level = "Low Risk"

        recommendation = (
            "Customer is likely to stay. "
            "Continue providing good service "
            "and maintain customer satisfaction."
        )

    elif probability < 0.60:

        risk_level = "Medium Risk"

        recommendation = (
            "Monitor customer activity and "
            "offer personalized discounts "
            "or suitable service upgrades."
        )

    else:

        risk_level = "High Risk"

        recommendation = (
            "Contact the customer and offer "
            "retention benefits, personalized offers "
            "or a suitable contract upgrade."
        )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Risk Level",
            risk_level
        )

    with col2:

        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

    st.info(
        f"💡 **Recommendation:** {recommendation}"
    )


st.divider()

st.caption(
    "Customer Churn Prediction | Machine Learning Project"
)
