import streamlit as st
import pickle
import numpy as np

# ---------------- LOAD FILES ---------------- #
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("gmm.pkl", "rb") as f:
    gmm = pickle.load(f)

with open("birch.pkl", "rb") as f:
    birch = pickle.load(f)

# ---------------- UI ---------------- #
st.set_page_config(page_title="World Development Clustering", layout="centered")
st.title("🌍 World Development Clustering App")

st.write("Assign development clusters based on country indicators")

# ---------------- MODEL SELECTION ---------------- #
model_choice = st.selectbox(
    "Choose Clustering Model",
    ["K-Means", "Gaussian Mixture Model", "BIRCH"]
)

# ---------------- INPUTS ---------------- #
gdp = st.number_input("GDP per capita", min_value=0.0)
life = st.number_input("Life Expectancy", min_value=0.0)
population = st.number_input("Population", min_value=0.0)
literacy = st.number_input("Literacy Rate", min_value=0.0)
inflation = st.number_input("Inflation Rate", min_value=0.0)

input_data = np.array([[gdp, life, population, literacy, inflation]])
input_scaled = scaler.transform(input_data)

# ---------------- PREDICTION ---------------- #
if st.button("Assign Cluster"):
    if model_choice == "K-Means":
        cluster = kmeans.predict(input_scaled)[0]

    elif model_choice == "Gaussian Mixture Model":
        cluster = gmm.predict(input_scaled)[0]

    elif model_choice == "BIRCH":
        cluster = birch.predict(input_scaled)[0]

    st.success(f"📊 Assigned Cluster: {cluster}")

# ---------------- INFO ---------------- #
st.markdown("---")
st.subheader("Models Used in This Project")
st.markdown("""
✔ K-Means (Deployed)  
✔ Gaussian Mixture Model (Deployed)  
✔ BIRCH (Deployed)  
✔ Hierarchical Clustering (Analysis Only)  
✔ DBSCAN (Analysis Only)  
❌ Spectral Clustering (Excluded due to poor silhouette score)
""")