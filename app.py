import streamlit as st
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# Load the trained model
with open("rf_model.pkl", "rb") as file:
    rf_model = pickle.load(file)

# Load the cleaned dataset
df = pd.read_csv("water_potability.csv")  # Pastikan file ini adalah dataset yang sudah dibersihkan

# Function to predict water potability
def predict_potability(params):
    prediction = rf_model.predict([params])[0]
    return "💧 Safe to Drink!" if prediction == 1 else "🚫 Not Safe to Drink"

# Navbar
selected = option_menu(
    menu_title="Water Quality App",
    options=["Prediction", "Visualization", "Dataset"],
    icons=["droplet", "bar-chart", "table"],
    menu_icon="menu",
    default_index=0,
    orientation="horizontal"
)

# Page: Prediction
if selected == "Prediction":
    st.title("🚰 Water Potability Prediction App 🚰")
    st.write("### Is your water safe to drink? Let's find out!")

    # Input parameters
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

# Page: Visualization
elif selected == "Visualization":
    st.title("📊 Water Quality Data Visualization 📊")

    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Scatterplot: User selects variables to compare
    st.write("### Relationship Between Variables")
    var_x = st.selectbox("Select X-axis Variable", df.columns[:-1])  # Exclude 'Potability'
    var_y = st.selectbox("Select Y-axis Variable", df.columns[:-1])
    hue_var = st.selectbox("Select Hue (Optional)", ["None"] + list(df.columns))

    fig, ax = plt.subplots()
    if hue_var == "None":
        sns.scatterplot(x=df[var_x], y=df[var_y], ax=ax)
    else:
        sns.scatterplot(x=df[var_x], y=df[var_y], hue=df[hue_var], ax=ax)

    ax.set_title(f"{var_x} vs {var_y}")
    st.pyplot(fig)

# Page: Dataset
elif selected == "Dataset":
    st.title("📄 Water Quality Dataset 📄")
    st.write("### Full Dataset")
    st.dataframe(df)

    st.write("### Basic Statistics")
    st.write(df.describe())

