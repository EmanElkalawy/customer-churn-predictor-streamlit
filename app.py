import streamlit as st
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="ChurnGuard", page_icon="📉", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.9rem; font-weight: 700; color: #FF4B4B; text-align: center; margin-bottom: 0.4rem;}
    .sub-header {font-size: 1.35rem; text-align: center; color: #b0b0b0; margin-bottom: 2rem;}
    .high-risk {background-color: #3c1a1a; padding: 18px; border-radius: 12px; border-left: 7px solid #FF4B4B; color: white;}
    .low-risk  {background-color: #1a3c34; padding: 18px; border-radius: 12px; border-left: 7px solid #00C853; color: white;}
    .roi-box {background-color: #1e3a5f; padding: 20px; border-radius: 12px; border-left: 7px solid #00BFFF;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📉 ChurnGuard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Customer Churn Prediction with Business ROI Calculator</p>', unsafe_allow_html=True)

# Load artifacts
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load('models/preprocessor.pkl')
    model = joblib.load('models/churn_model.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return preprocessor, model, feature_names

preprocessor, model, feature_names = load_artifacts()

# Sidebar
st.sidebar.header("Customer Details")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 118.0, 70.0)
total_charges = st.sidebar.slider("Total Charges ($)", 18.0, 8684.0, 800.0)

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox("Payment Method", 
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
internet_service = st.sidebar.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])

senior = st.sidebar.checkbox("Senior Citizen")
partner = st.sidebar.checkbox("Has Partner")
dependents = st.sidebar.checkbox("Has Dependents")
paperless = st.sidebar.checkbox("Paperless Billing")

# Input Data
data = {
    'gender': 'Male',
    'SeniorCitizen': int(senior),
    'Partner': 'Yes' if partner else 'No',
    'Dependents': 'Yes' if dependents else 'No',
    'tenure': tenure,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': internet_service,
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': contract,
    'PaperlessBilling': 'Yes' if paperless else 'No',
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

input_df = pd.DataFrame([data])

# Prediction
X_processed = preprocessor.transform(input_df)
prediction = model.predict(X_processed)[0]
probability = model.predict_proba(X_processed)[0][1]

# Main Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Prediction Result")
    if prediction == 1:
        st.markdown('<div class="high-risk">🔴 HIGH CHURN RISK — This customer is likely to leave</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="low-risk">🟢 LOW CHURN RISK — This customer is likely to stay</div>', unsafe_allow_html=True)

    # Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability", 'font': {'size': 26}},
        number={'font': {'size': 70}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#FF4B4B" if probability > 0.5 else "#00C853"},
            'steps': [
                {'range': [0, 40], 'color': "#00C853"},
                {'range': [40, 70], 'color': "#FFB300"},
                {'range': [70, 100], 'color': "#FF4B4B"}
            ]
        }
    ))
    st.plotly_chart(fig, width='stretch')

with col2:
    st.metric("Churn Probability", f"{probability:.1%}")
    st.metric("Risk Level", "High Risk" if prediction == 1 else "Low Risk")

# Retention Strategies
if probability > 0.5:
    st.subheader("💡 Recommended Retention Strategies")
    st.info("• Offer discount or loyalty package  \n• Upgrade to Fiber optic + Tech Support  \n• Personalized retention call/email")

# ==================== NEW: BUSINESS ROI CALCULATOR ====================
st.subheader("💰 Business ROI / Savings Calculator")

with st.expander("Calculate Potential Savings from Retention Campaign", expanded=True):
    col_a, col_b = st.columns(2)
    
    with col_a:
        num_customers = st.number_input("Number of Similar Customers", min_value=10, value=500, step=50)
        avg_customer_value = st.number_input("Average Monthly Revenue per Customer ($)", min_value=10.0, value=70.0, step=5.0)
    
    with col_b:
        retention_rate = st.slider("Expected Retention Success Rate (%)", 0, 100, 25)
        campaign_cost = st.number_input("Retention Campaign Cost per Customer ($)", min_value=0.0, value=15.0, step=5.0)

    # Calculations
    churn_prob = probability
    expected_churners = num_customers * churn_prob
    retained_customers = expected_churners * (retention_rate / 100)
    revenue_saved = retained_customers * avg_customer_value * 12   # Annual revenue
    total_campaign_cost = num_customers * campaign_cost
    net_savings = revenue_saved - total_campaign_cost

    # Display Results
    st.markdown("### Estimated Business Impact")
    
    col_res1, col_res2, col_res3 = st.columns(3)
    with col_res1:
        st.metric("Expected Churners", f"{expected_churners:.0f}")
    with col_res2:
        st.metric("Customers You Can Retain", f"{retained_customers:.0f}", f"+{retention_rate}% success")
    with col_res3:
        st.metric("Potential Annual Revenue Saved", f"${revenue_saved:,.0f}")

    st.metric("Net Savings After Campaign Cost", f"${net_savings:,.0f}", 
              delta=f"${net_savings:,.0f}" if net_savings > 0 else f"${net_savings:,.0f}")

    if net_savings > 0:
        st.success(f"✅ This retention campaign could save your business **${net_savings:,.0f}** annually!")
    else:
        st.warning("⚠️ Campaign cost is higher than expected savings. Consider optimizing cost or targeting higher-risk customers.")

# SHAP Section
with st.expander("🔍 Explain This Prediction (SHAP Analysis)", expanded=False):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("**Feature Contributions**")
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=feature_names
        )
        shap.plots.waterfall(explanation)
        fig = plt.gcf()
        st.pyplot(fig, clear_figure=True)

    with col_b:
        st.write("**Top 10 Feature Importance**")
        importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(10)
        fig_imp = px.bar(importance, orientation='h', title="Top Important Features")
        st.plotly_chart(fig_imp, width='stretch')

st.caption("ChurnGuard • XGBoost + Pipeline + SHAP + Business ROI Calculator ")