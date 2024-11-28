import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("rf_model.pkl", "rb") as file:
    rf_model = pickle.load(file)

# Function to predict water potability
def predict_potability(params):
    prediction = rf_model.predict([params])[0]
    return "💧 Safe to Drink!" if prediction == 1 else "🚫 Not Safe to Drink"

# Information about each parameter
def parameter_info():
    st.write("""
    ### 🌊 Water Quality Parameters Explained:

    **pH** - A vital measure! This tells us if the water’s acidic or alkaline. Ideal range? Between 6.5 and 8.5 to keep it safe and refreshing. Too low or too high? It could corrode or scale your pipes - and your stomach isn’t a fan either!

    **Hardness** - Ever noticed soap not lathering easily? High hardness in water is why! While hard water isn’t unsafe, softer water means happier soap and scale-free appliances.

    **Solids (TDS)** - Too many dissolved solids affect taste and clarity. Think of it as the “purity meter” - ideally, we keep it under 500 ppm for safe drinking.

    **Chloramines** - Essential for disinfection but a double-edged sword. It’s safe in controlled amounts, but high levels may pose risks.

    **Sulfate** - Adds bitterness if too high. While safe in small amounts, excessive sulfate could lead to gastrointestinal issues.

    **Conductivity** - Indicates the water’s ability to conduct electricity, directly tied to dissolved salts. The more it conducts, the “saltier” it is!

    **Organic Carbon** - Represents the presence of organic materials. Think of it as a contamination indicator; high organic content isn’t good news.

    **Trihalomethanes (THMs)** - By-products from water disinfection. Safe in low amounts, but prolonged exposure to high levels can be harmful.

    **Turbidity** - Measures cloudiness. Higher values may mean contamination, as clear water is usually safer to drink.
    """)


# # Streamlit App
# st.title("Water Potability Prediction")
# st.write("Enter water quality parameters to check if the water is safe to drink:")

# Streamlit App
st.title("🚰 Water Potability Prediction App 🚰")
st.write("### Is your water safe to drink? Let's find out!")

# Explanation Section
st.markdown("""
    ## Why This Matters 🌍
    Access to clean, safe drinking water is a **human right**. Waterborne illnesses are still a global concern, and ensuring water safety is crucial for health and development. Predicting potability helps monitor water quality, reduce health risks, and support economic growth by cutting healthcare costs.

    **This app** leverages data and machine learning to analyze various water characteristics, helping predict if water is safe for consumption. Each parameter tells us a unique story about water quality, offering valuable insights into water's suitability for drinking.
""")

st.write("\n---\n")
parameter_info()

# Set default values to likely safe ranges
pH = st.number_input("pH (6.5-8.5 is ideal)", min_value=0.0, max_value=14.0, value=7.5, step=0.1)
hardness = st.number_input("Hardness (mg/L)", min_value=0.0, max_value=500.0, value=120.0, step=1.0)
solids = st.number_input("Total Dissolved Solids (ppm)", min_value=0.0, max_value=5000.0, value=250.0, step=1.0)
chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, max_value=500.0, value=180.0, step=1.0)
conductivity = st.number_input("Conductivity (μS/cm)", min_value=0.0, max_value=1000.0, value=400.0, step=1.0)
organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
trihalomethanes = st.number_input("Trihalomethanes (μg/L)", min_value=0.0, max_value=200.0, value=40.0, step=1.0)
turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)

# Predict potability when user clicks the button
if st.button("Check Potability"):
    params = [pH, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]
    result = predict_potability(params)
    st.subheader(f"Prediction: {result}")

# Display additional information

st.write("\n---\n")
