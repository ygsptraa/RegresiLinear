import pandas as pd
import numpy as np
import streamlit as st
import pickle
import plotly.express as px

# Memuat Model dari pickle
with open('linear_regression_model.pkl', 'rb') as f:
    models = pickle.load(f)

# Memuat data asli
ump_data = pd.read_csv("Indonesian Salary by Region (1997-2022).csv")

# Menyusun data dengan prediksi (tahun mendatang 2023 - 2032)
future_years = np.array(range(ump_data['YEAR'].max() + 1, ump_data['YEAR'].max() + 11))  # Tahun 2023-2032
future_ump = []

# Melakukan prediksi untuk setiap region
for provinsi in ump_data['REGION'].unique():
    model = models[provinsi]
    future_ump_provinsi = model.predict(future_years.reshape(-1, 1))
    future_ump.extend(future_ump_provinsi)

# Membuat DataFrame untuk prediksi
future_df = pd.DataFrame({
    'REGION': np.repeat(ump_data['REGION'].unique(), 10),
    'YEAR': np.tile(range(ump_data['YEAR'].max() + 1, ump_data['YEAR'].max() + 11), ump_data['REGION'].nunique()),
    'SALARY': future_ump
})

# Gabungkan data asli dengan data prediksi
combined_df = pd.concat([ump_data, future_df], axis=0, ignore_index=True)

# Membuat Grafik dengan Plotly (termasuk prediksi)
fig = px.bar(combined_df, x='REGION', y="SALARY", color="REGION",
             animation_frame="YEAR", range_y=[0, 7000000])

# Menampilkan grafik di Streamlit
st.plotly_chart(fig)
