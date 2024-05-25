# ml_app_cp.py
import streamlit as st
import numpy as np
import joblib
import xgboost as xgb

# Load model
def load_model(model_file):
    loaded_model = joblib.load(open(model_file, "rb"))
    return loaded_model

def get_value(val, my_dict):
    return my_dict.get(val, 0)

# Main function for the app
def run_ml_app():
    # Load dictionaries
    df_manufacturer = {
        "SKODA": 46101.0,
        "SSANGYONG": 29997.600490196077,
        "JAGUAR": 27949.64285714286,
        "TOYOTA": 30000.0,
        "BMW": 45000.0
    }

    df_fuel_type = {
        "Diesel": 23474.326286398085,
        "Plug-in Hybrid": 20708.416666666668,
        "Petrol": 25000.0,
        "Electric": 30000.0
    }

    df_gear_box_type = {
        "Tiptronic": 21097.235555555555,
        "Automatic": 17249.66915153158,
        "Manual": 15000.0,
        "CVT": 20000.0
    }

    # Attribute information
    attribute_info = """
                    - Manufacturer
                    - Fuel Type
                    - Mileage
                    - Gear Box Type
                    - Airbags
                    - Car Age
                    """

    st.subheader("ML Section")

    # Display attribute information
    with st.expander("Attribute Info"):
        st.markdown(attribute_info)

    # User input data
    st.subheader("Input Your Data")
    manufacturer = st.selectbox("Manufacturer", list(df_manufacturer.keys()))
    fuel_type = st.selectbox("Fuel Type", list(df_fuel_type.keys()))
    mileage = st.number_input("Mileage", min_value=0)
    gear_box_type = st.selectbox("Gear Box Type", list(df_gear_box_type.keys()))
    airbags = st.number_input("Airbags", 0, 16)
    car_age = st.number_input("Car Age", 0, 24)

    # Convert user input to feature vector
    encoded_result = [
        get_value(manufacturer, df_manufacturer),
        get_value(fuel_type, df_fuel_type),
        mileage,
        get_value(gear_box_type, df_gear_box_type),
        airbags,
        car_age,
    ]

    # Load model
    model = load_model("best_xgb_model.pkl")

    if model:
        # Prediction
        prediction = model.predict(np.array(encoded_result).reshape(1, -1))

        # Display prediction result
        st.subheader("Car Prediction Result")
        st.write("Predicted Car Price:", round(prediction[0], 2))
    else:
        st.write("Model not found. Please make sure the model file is available.")
