import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Weather ML Dashboard",
    page_icon="🌦️",
    layout="centered"
)

# -----------------------------
# Title & Description
# -----------------------------
st.title("🌦️ Weather Temperature Prediction Dashboard")
st.markdown(
    """
    This dashboard uses **Machine Learning (Linear Regression)**  
    to predict **Temperature** based on **Humidity** and **Wind Speed**.
    """
)

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("weather_data.csv")

data = load_data()

# -----------------------------
# Train ML Model
# -----------------------------
X = data[["humidity", "wind"]]
y = data["temp"]

model = LinearRegression()
model.fit(X, y)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("🔧 Input Parameters")

humidity = st.sidebar.slider(
    "Humidity (%)",
    min_value=20,
    max_value=100,
    value=60
)

wind = st.sidebar.slider(
    "Wind Speed (m/s)",
    min_value=0.5,
    max_value=10.0,
    value=4.0,
    step=0.1
)

# -----------------------------
# Prediction
# -----------------------------
prediction = model.predict([[humidity, wind]])

# -----------------------------
# Display Prediction
# -----------------------------
st.subheader("📊 Predicted Result")

st.metric(
    label="🌡 Predicted Temperature (°C)",
    value=f"{prediction[0]:.2f}"
)

# -----------------------------
# Model Accuracy
# -----------------------------
accuracy = model.score(X, y)
st.info(f"📈 Model Accuracy: {accuracy:.2f}")

# -----------------------------
# Dataset Display
# -----------------------------
with st.expander("📁 View Training Dataset"):
    st.dataframe(data)

# -----------------------------
# Visualization
# -----------------------------
st.subheader("📉 Data Visualization")

fig, ax = plt.subplots()
ax.scatter(data["humidity"], data["temp"])
ax.set_xlabel("Humidity (%)")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Humidity vs Temperature")

st.pyplot(fig)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("👨‍💻 **Developed using Python, Streamlit & Machine Learning**")
