import streamlit as st
import numpy as np
import joblib

model = joblib.load("pd_model_rf.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Ghana Bank Loan PD Scoring")

age = st.slider("Age", 21, 65)
income = st.number_input("Monthly Income (GHS)", min_value=100.0, value=3000.0)
loan = st.number_input("Loan Amount (GHS)", min_value=1000.0, value=20000.0)
tenure = st.slider("Loan Tenure (months)", 6, 60)
interest = st.slider("Interest Rate (%)", 5.0, 35.0, 15.0)
liquidity = st.slider("Bank Liquidity Ratio (%)", 10.0, 80.0)
npl = st.slider("Bank NPL Ratio (%)", 5.0, 25.0)
gdp = st.slider("GDP Growth (%)", -5.0, 8.0, 3.0)
inflation = st.slider("Inflation Rate (%)", 5.0, 20.0)

sector = st.selectbox("Sector", ["Agriculture", "Manufacturing", "Salaried", "SME", "Trade"])
sector_enc = [0, 0, 0, 0]
sector_map = {"Manufacturing": 0, "Salaried": 1, "SME": 2, "Trade": 3}
if sector != "Agriculture":
    sector_enc[sector_map[sector]] = 1

features = np.array([[age, income, loan, tenure, interest, liquidity, npl, gdp, inflation] + sector_enc])
features_scaled = scaler.transform(features)

if st.button("Predict Default Probability"):
    pred_prob = model.predict_proba(features_scaled)[0][1]
    st.success(f"Predicted Probability of Default: {pred_prob:.2%}")