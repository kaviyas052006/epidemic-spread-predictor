import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page settings
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

# Load data from GitHub
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/" \
      "csse_covid_19_data/csse_covid_19_time_series/" \
      "time_series_covid19_confirmed_global.csv"

df = pd.read_csv(url)

# Convert wide → long format
df = df.melt(
    id_vars=["Province/State", "Country/Region", "Lat", "Long"],
    var_name="Date",
    value_name="Cases"
)

# Convert date column
df["Date"] = pd.to_datetime(df["Date"])

# Filter selected country
country_df = df[df["Country/Region"] == country]

# Group by date
country_df = country_df.groupby("Date")["Cases"].sum().reset_index()

# Plot graph
st.subheader("📈 Total Cases Over Time")

fig, ax = plt.subplots()
ax.plot(country_df["Date"], country_df["Cases"])
ax.set_xlabel("Date")
ax.set_ylabel("Cases")

st.pyplot(fig)

# Show latest total cases
st.metric("Total Cases", int(country_df["Cases"].iloc[-1]))