import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Epidemic Dashboard", layout="wide")

# Title
st.title("📊 Epidemic Prediction Dashboard")

# Sidebar
st.sidebar.header("Options")

country = st.sidebar.selectbox(
    "Select Country",
    ["India", "US", "Brazil"]
)

st.write(f"Showing data for: **{country}**")

# Load data
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/" \
      "csse_covid_19_data/csse_covid_19_time_series/" \
      "time_series_covid19_confirmed_global.csv"

df = pd.read_csv(url)

# Convert format
df = df.melt(
    id_vars=["Province/State", "Country/Region", "Lat", "Long"],
    var_name="Date",
    value_name="Cases"
)

df["Date"] = pd.to_datetime(df["Date"])

# Filter country
country_df = df[df["Country/Region"] == country]
country_df = country_df.groupby("Date")["Cases"].sum().reset_index()

# 🔥 NEW: Daily Cases
country_df["Daily Cases"] = country_df["Cases"].diff().fillna(0)

# 🔥 NEW: 7-day rolling average
country_df["7-day Avg"] = country_df["Daily Cases"].rolling(window=7).mean()

# Layout (columns)
col1, col2 = st.columns(2)

# Graph 1: Total Cases
with col1:
    st.subheader("📈 Total Cases")
    fig1, ax1 = plt.subplots()
    ax1.plot(country_df["Date"], country_df["Cases"])
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Cases")
    st.pyplot(fig1)

# Graph 2: Daily Cases + Average
with col2:
    st.subheader("📊 Daily Cases & 7-Day Average")
    fig2, ax2 = plt.subplots()
    ax2.plot(country_df["Date"], country_df["Daily Cases"], label="Daily Cases")
    ax2.plot(country_df["Date"], country_df["7-day Avg"], label="7-Day Avg")
    ax2.legend()
    st.pyplot(fig2)

# Metrics
st.subheader("📌 Key Stats")

col3, col4 = st.columns(2)

with col3:
    st.metric("Total Cases", int(country_df["Cases"].iloc[-1]))

with col4:
    st.metric("Latest Daily Cases", int(country_df["Daily Cases"].iloc[-1]))