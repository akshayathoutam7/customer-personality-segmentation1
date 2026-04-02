import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Personality Segmentation", layout="wide")

st.title("🧠 Customer Personality Segmentation App")
st.write("This app segments customers based on their personality and spending behavior.")

# Sidebar
st.sidebar.header("Enter Customer Details")

age = st.sidebar.slider("Age", 18, 60, 25)
income = st.sidebar.slider("Income", 10000, 100000, 30000)
spending = st.sidebar.slider("Spending Score", 1, 100, 50)
savings = st.sidebar.slider("Savings Score", 1, 100, 40)

# Sample dataset
data = {
    'Age': [22, 25, 47, 52, 46, 23, 30, 40],
    'Income': [15000, 29000, 48000, 52000, 50000, 16000, 30000, 40000],
    'Spending': [39, 81, 6, 77, 40, 76, 50, 60],
    'Savings': [20, 30, 80, 70, 60, 25, 40, 50]
}

df = pd.DataFrame(data)

# Model
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(df)

# Prediction
new_customer = [[age, income, spending, savings]]
cluster = kmeans.predict(new_customer)

st.subheader("📌 Customer Details")
st.write("Age:", age)
st.write("Income:", income)
st.write("Spending Score:", spending)
st.write("Savings Score:", savings)

st.subheader("🎯 Predicted Customer Personality")

if cluster[0] == 0:
    st.success("Low Spender Customer")
elif cluster[0] == 1:
    st.success("High Value Customer")
else:
    st.success("Average Customer")

# Show dataset
st.subheader("📂 Training Dataset")
st.dataframe(df)

st.subheader("📊 Customer Segments Info")

st.info("""
1. High Value Customer – High income and high spending.
2. Average Customer – Medium income and spending.
3. Low Spender – Low spending and savings behavior.
""")