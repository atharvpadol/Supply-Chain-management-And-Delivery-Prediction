import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

st.set_page_config(page_title="Supply Chain Intelligence", layout="wide")

# ---------- CSS (Premium UI) ----------
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #F8FAFC, #EEF2F7);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E293B, #334155);
    color: white;
}

/* Cards */
.card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.08);
}

/* KPI */
.kpi {
    text-align: center;
}

/* Hover animation */
.card:hover {
    transform: translateY(-5px);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #2563EB, #3B82F6);
    color: white;
    border-radius: 10px;
    font-weight: bold;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-size: 18px;
    padding: 10px;
}

/* Smooth */
* {
    transition: all 0.2s ease-in-out;
}

</style>
""", unsafe_allow_html=True)

# ---------- LOAD ----------
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

df = pd.read_csv("DataCoSupplyChainDataset.csv")

# -------- SAFE PREPROCESSING --------

# Handle Order Status (if exists)
if 'Order Status' in df.columns:
    df = df[df['Order Status'] != 'CANCELED']

# Handle date columns safely
if 'order date (DateOrders)' in df.columns and 'shipping date (DateOrders)' in df.columns:
    
    df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'])
    df['shipping date (DateOrders)'] = pd.to_datetime(df['shipping date (DateOrders)'])

    df['processing'] = (
        df['shipping date (DateOrders)'] - df['order date (DateOrders)']
    ).dt.days
else:
    df['processing'] = 0

# Handle delay calculation safely
if 'Days for shipment (scheduled)' in df.columns:
    df['delay'] = df['processing'] - df['Days for shipment (scheduled)']
else:
    df['delay'] = 0

# Target column
df['is_late'] = (df['delay'] > 0).astype(int)

# ---------- KPIs ----------

# Late %
late_pct = df['is_late'].mean() * 100

# Avg Profit (safe)
if 'Order Profit Per Order' in df.columns:
    avg_profit = df['Order Profit Per Order'].mean()
else:
    avg_profit = 0

# Top region (safe)
if 'Order Region' in df.columns:
    top_region = df.groupby('Order Region')['is_late'].mean().idxmax()
else:
    top_region = "N/A"

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

    col1.markdown(f"<div class='card kpi'><h3>🚨 Late Delivery %</h3><h2>{late_pct:.2f}%</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card kpi'><h3>💰 Avg Profit</h3><h2>${avg_profit:.2f}</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card kpi'><h3>🌍 Risk Region</h3><h2>{top_region}</h2></div>", unsafe_allow_html=True)

    st.markdown("### 📈 Delay Distribution")
    fig1 = px.histogram(df, x="delay", nbins=40, title="Delay Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### 🌍 Regional Risk")
    region = df.groupby('Order Region')['is_late'].mean().sort_values(ascending=False).head(10)
    fig2 = px.bar(region, x=region.values*100, y=region.index, orientation='h',
                  title="Top Risk Regions")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 💰 Profit Impact")
    dp = df.groupby('delay')['Order Profit Per Order'].mean().reset_index()
    fig3 = px.line(dp, x="delay", y="Order Profit Per Order", title="Profit vs Delay")
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
            st.info("💡 Recommendation: Use faster shipping / check processing")
        else:
            st.success("✅ Likely On-Time")
            st.info("📈 Delivery conditions look optimal")

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("<p style='text-align:center;'>Built by Atharv Padol 🚀</p>", unsafe_allow_html=True)
