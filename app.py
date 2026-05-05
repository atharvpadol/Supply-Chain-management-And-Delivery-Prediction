import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Supply Chain Intelligence", layout="wide")

# ---------- DARK UI CSS ----------
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0F172A, #1E293B);
    color: #E2E8F0;
}

/* Text */
h1, h2, h3, h4, h5, h6, p, span, label {
    color: #E2E8F0 !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #1E293B;
    padding: 15px;
    border-radius: 12px;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-size: 18px;
    color: #CBD5F5;
}
.stTabs [aria-selected="true"] {
    color: #60A5FA !important;
    border-bottom: 2px solid #60A5FA;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #3B82F6, #2563EB);
    color: white;
    border-radius: 10px;
    padding: 10px;
    border: none;
}
.stButton>button:hover {
    transform: scale(1.05);
}

/* Plotly */
.js-plotly-plot {
    background-color: transparent !important;
}

/* Footer hide */
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# ---------- LOAD DATA ----------
df = pd.read_csv("DataCoSupplyChainDataset.csv")

# ---------- CLEAN DATA ----------
df = df[df['Order Status'] != 'CANCELED']

df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'])
df['shipping date (DateOrders)'] = pd.to_datetime(df['shipping date (DateOrders)'])

df['processing'] = (
    df['shipping date (DateOrders)'] - df['order date (DateOrders)']
).dt.days

df['delay'] = df['processing'] - df['Days for shipment (scheduled)']
df['is_late'] = (df['delay'] > 0).astype(int)

# ---------- KPIs ----------
late_pct = df['is_late'].mean() * 100
avg_profit = df['Order Profit Per Order'].mean()
top_region = df.groupby('Order Region')['is_late'].mean().idxmax()

# ---------- HEADER ----------
st.markdown("""
<h1 style='text-align:center;'>📦 Supply Chain Intelligence</h1>
<p style='text-align:center;'>Analytics + Machine Learning Dashboard</p>
""", unsafe_allow_html=True)

# ---------- TABS ----------
tab1, tab2 = st.tabs(["📊 Dashboard", "🔮 Prediction"])

# ================= DASHBOARD =================
with tab1:

    col1, col2, col3 = st.columns(3)

    col1.metric("🚨 Late Delivery %", f"{late_pct:.2f}%")
    col2.metric("💰 Avg Profit", f"${avg_profit:.2f}")
    col3.metric("🌍 Risk Region", top_region)

    st.markdown("### 📈 Delay Distribution")
    fig1 = px.histogram(df, x="delay", nbins=40, template="plotly_dark")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### 🌍 Regional Risk")
    region = df.groupby('Order Region')['is_late'].mean().sort_values(ascending=False).head(10)
    fig2 = px.bar(region, x=region.values*100, y=region.index,
                  orientation='h', template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 💰 Profit vs Delay")
    dp = df.groupby('delay')['Order Profit Per Order'].mean().reset_index()
    fig3 = px.line(dp, x="delay", y="Order Profit Per Order",
                   template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)

# ================= PREDICTION =================
with tab2:

    st.markdown("### 🔮 Predict Delivery Status")

    col1, col2 = st.columns(2)

    with col1:
        shipping = st.selectbox("🚚 Shipping Mode", list(encoders['Shipping Mode'].keys()))
        segment = st.selectbox("👤 Customer Segment", list(encoders['Customer Segment'].keys()))
        dept = st.selectbox("🏢 Department", list(encoders['Department Name'].keys()))

    with col2:
        region = st.selectbox("🌍 Region", list(encoders['Order Region'].keys()))
        days = st.slider("📅 Scheduled Days", 1, 10, 3)
        month = st.slider("📆 Month", 1, 12, 6)
        hour = st.slider("⏰ Hour", 0, 23, 12)

    def encode(col, val):
        return encoders[col].get(val, 0)

    input_data = np.array([[days,
                            encode('Shipping Mode', shipping),
                            encode('Customer Segment', segment),
                            encode('Department Name', dept),
                            encode('Order Region', region),
                            month,
                            hour]])

    if st.button("🚀 Predict Delivery"):

        pred = model.predict(input_data)[0]

        if pred == 1:
            st.error("⚠️ High Risk of Delay")
            st.info("💡 Recommendation: Use faster shipping / optimize processing")
        else:
            st.success("✅ Likely On-Time")
            st.info("📈 Delivery conditions look optimal")

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("<p style='text-align:center;'>Built by Atharv Padol 🚀</p>", unsafe_allow_html=True)
