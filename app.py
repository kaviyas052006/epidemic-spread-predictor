import streamlit as st
import pandas as pd
import plotly.express as px

# Page config
st.set_page_config(page_title="Epidemic Dashboard", layout="wide")

# Title
st.title("🌍 Epidemic Spread Prediction Dashboard")
st.markdown("### 🔍 Predicting and Visualizing Global Health Risks")

# Sidebar
st.sidebar.header("Options")

country = st.sidebar.selectbox(
    "Select Country",
    ["India", "US", "Brazil"]
)

# Load data
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/" \
      "csse_covid_19_data/csse_covid_19_time_series/" \
      "time_series_covid19_confirmed_global.csv"

df = pd.read_csv(url)

# Melt data
df = df.melt(
    id_vars=["Province/State", "Country/Region", "Lat", "Long"],
    var_name="Date",
    value_name="Cases"
)

df["Date"] = pd.to_datetime(df["Date"])

# Latest data for map
latest_df = df[df["Date"] == df["Date"].max()]

# 🌍 MAP
st.subheader("🌍 Global Hotspots Map")

fig_map = px.scatter_geo(
    latest_df,
    lat="Lat",
    lon="Long",
    size="Cases",
    color="Cases",
    hover_name="Country/Region",
    title="Global Spread",
)

st.plotly_chart(fig_map)

# Country filtering
country_df = df[df["Country/Region"] == country]
country_df = country_df.groupby("Date")["Cases"].sum().reset_index()

# Features
country_df["Daily Cases"] = country_df["Cases"].diff().fillna(0)
country_df["7-day Avg"] = country_df["Daily Cases"].rolling(window=7).mean()

# 🔴 HOTSPOT DETECTION
latest_growth = country_df["Daily Cases"].iloc[-1]

if latest_growth > 10000:
    risk = "🔴 High Risk"
elif latest_growth > 5000:
    risk = "🟠 Medium Risk"
else:
    risk = "🟢 Low Risk"

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Total Cases")
    fig1 = px.line(country_df, x="Date", y="Cases")
    st.plotly_chart(fig1)

with col2:
    st.subheader("📊 Daily Cases vs Avg")
    fig2 = px.line(country_df, x="Date", y=["Daily Cases", "7-day Avg"])
    st.plotly_chart(fig2)

# Metrics
st.subheader("📌 Key Stats")

col3, col4, col5 = st.columns(3)

with col3:
    st.metric("Total Cases", int(country_df["Cases"].iloc[-1]))

with col4:
    st.metric("Latest Daily Cases", int(latest_growth))

with col5:
    st.metric("Risk Level", risk)

# 🧠 INSIGHTS SECTION
st.subheader("🧠 Insights")

if risk == "🔴 High Risk":
    st.error("Cases are rising rapidly. Immediate action required.")
elif risk == "🟠 Medium Risk":
    st.warning("Cases are increasing. Monitor closely.")
else:
    st.success("Situation is under control.")

st.markdown("""
### 📌 What This Means:
- This dashboard helps identify potential outbreak zones.
- Authorities can prepare healthcare resources.
- Citizens can avoid high-risk areas.
""")