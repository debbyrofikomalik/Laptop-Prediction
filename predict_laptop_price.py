import streamlit as st
import numpy as np
import joblib

# Load the model
model = joblib.load("laptop_pred.pkl")

st.title("Laptop Price Prediction App")
st.divider()

st.write(
    "This application provides an estimated price for a laptop based on the input values of processor speed, RAM size, and storage capacity. "
    "After entering the values, click the calculation button to obtain the price estimation."
)

st.divider()

# Input fields
processor_speed = st.number_input("Processor Speed (GHz)", value=2.50, step=0.50)
ram_size = st.number_input("RAM Size (GB)", value=16, step=8)
storage_capacity = st.number_input("Storage Capacity (GB)", value=512, step=256)

X = [processor_speed, ram_size, storage_capacity]

st.divider()

# Prediction button
if st.button("Calculate Price Estimation"):
    
    x1 = np.array(X).reshape(1, -1)  # ensure 2D shape for prediction
    prediction = model.predict(x1)[0]
    
    st.write(f"The estimated price of the laptop is ${prediction:,.2f}.")
else:
    st.write("Press the calculation button to obtain a price estimation.")