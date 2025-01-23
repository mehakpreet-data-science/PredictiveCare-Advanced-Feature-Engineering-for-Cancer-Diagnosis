import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Assuming you have a trained Random Forest model saved

# Load the trained model
# Replace 'grid_search_model.pkl' with the path to your trained model file
model = joblib.load("grid_search_model.pkl")

# App title
st.title("Breast Cancer Diagnosis Predictor")
st.markdown("Enter the required features below to predict whether the diagnosis is **Malignant** or **Benign**.")

# User input
st.sidebar.header("Input Features")

# Create input fields for the user
area_worst = st.sidebar.number_input("Enter area_worst:", value=0.0)
perimeter_worst = st.sidebar.number_input("Enter perimeter_worst:", value=0.0)
concave_points_mean = st.sidebar.number_input("Enter concave_points_mean:", value=0.0)
concavity_mean = st.sidebar.number_input("Enter concavity_mean:", value=0.0)
perimeter_mean = st.sidebar.number_input("Enter perimeter_mean:", value=0.0)
area_se = st.sidebar.number_input("Enter area_se:", value=0.0)
area_mean = st.sidebar.number_input("Enter area_mean:", value=0.0)
radius_worst = st.sidebar.number_input("Enter radius_worst:", value=0.0)
concave_points_worst = st.sidebar.number_input("Enter concave_points_worst:", value=0.0)
smoothness_worst = st.sidebar.number_input("Enter smoothness_worst:", value=0.0)

# Button for prediction
if st.sidebar.button("Predict"):
    # Convert inputs into a 2D array
    input_entries = np.array([[
        area_worst, perimeter_worst, concave_points_mean, concavity_mean,
        perimeter_mean, area_mean, area_se, radius_worst,
        concave_points_worst, smoothness_worst
    ]])

    # Create a DataFrame for display
    input_df = pd.DataFrame({
        'Area Worst': [area_worst],
        'Perimeter Worst': [perimeter_worst],
        'Concave Points Mean': [concave_points_mean],
        'Concavity Mean': [concavity_mean],
        'Perimeter Mean': [perimeter_mean],
        'Area Mean': [area_mean],
        'Area SE': [area_se],
        'Radius Worst': [radius_worst],
        'Concave Points Worst': [concave_points_worst],
        'Smoothness Worst': [smoothness_worst]
    })

    # Display the input values
    st.subheader("Input Values")
    st.write(input_df)

    # Make a prediction
    output = model.predict(input_entries)

    # Display the prediction result
    st.subheader("Prediction")
    if output[0] == 0:
        st.error("The diagnosis is **Malignant**. Consult a doctor immediately.")
    elif output[0] == 1:
        st.success("The diagnosis is **Benign**. It is safe.")
    else:
        st.warning("Invalid inputs. Please check the values.")
