import pickle
import numpy as np
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from streamlit_extras.stylable_container import stylable_container

# Load car dataset
car_df = pd.read_csv("cardata.csv")

# Load the model and scaler
model = joblib.load("best_xgb_model.pkl")
scaler = joblib.load('scaler.pkl')

# Initialize LabelEncoders for categorical features
categorical_columns = [
    'bodytype', 'fueltype', 'transmission', 'DriveType', 'Insurance', 'oem', 'city'
]
label_encoders = {col: LabelEncoder().fit(car_df[col]) for col in categorical_columns}

# Function to predict the resale price
def predict_resale_price():
    # Prepare numerical features
    num_features = np.array([
        int(m_seats),
        int(m_km),
        int(m_Registration_year),
        int(m_ownerNo),
        int(m_Engine),
        int(m_modelYear),
        int(m_Year_of_Manufacture),
        int(m_gear),
        float(m_mileage)
    ]).reshape(1, -1)

    # Scale the numerical features using the MinMaxScaler
    scaled_num_features = scaler.transform(num_features)

    # Prepare and encode categorical features
    cat_features = np.array([
        label_encoders['bodytype'].transform([m_bodytype])[0],
        label_encoders['fueltype'].transform([m_fuel_type])[0],
        label_encoders['transmission'].transform([m_transmission])[0],
        label_encoders['DriveType'].transform([m_drivetype])[0],
        label_encoders['Insurance'].transform([m_Insurance])[0],
        label_encoders['oem'].transform([m_oem])[0],
        label_encoders['city'].transform([m_city])[0]
    ]).reshape(1, -1)

    # Concatenate scaled numerical features and encoded categorical features
    final_input = np.hstack((scaled_num_features, cat_features))

    # Make prediction using the model
    prediction = model.predict(final_input)

    # Return formatted price prediction
    return prediction[0]


# Streamlit Page Configuration

# Title

st.set_page_config(layout="wide",page_icon=":material/directions_bus:",page_title="CarPrediction Project",initial_sidebar_state="expanded")
st.title(":red[Car Dekho Used Car Price Prediction]")
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://wallpaper.forfun.com/fetch/fb/fb7772b40cf1d5ab756c9cd9b626603b.jpeg?w=1200&r=0.5625");
        background-size: cover; /* Ensures the image covers the entire container */
        background-position: center; /* Centers the image */
        background-repeat: no-repeat; /* Prevents the image from repeating */
        background-attachment: fixed; /* Fixes the image in place when scrolling */
        height: 100vh; /* Sets the height to 100% of the viewport height */
        width: 100vw; /* Sets the width to 100% of the viewport width */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"] {{
        background-color: #60191900; /* Replace with your desired color */
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    
    """,
    unsafe_allow_html=True
)


# Sidebar for user inputs
with st.sidebar:
    st.title(":red[Features]")
    m_transmission = st.selectbox(label="Transmission", options=car_df['transmission'].unique())
    m_oem = st.selectbox(label="Car Brand", options=car_df['oem'].unique())
    m_km = st.selectbox(label="Select KMs Driven", options=sorted(car_df['kms driven'].unique().astype(int)))
    m_gear = st.selectbox(label="Number of Gears", options=sorted(car_df['Gearbox'].unique().astype(int)))
    m_fuel_type = st.selectbox(label="Fuel Type", options=car_df['fueltype'].unique())
    m_bodytype = st.selectbox(label="Body Type", options=car_df['bodytype'].unique())
    m_mileage = st.selectbox(label="Mileage", options=sorted(car_df['Mileage'].unique().astype(float)))
    m_drivetype = st.selectbox(label="Drive Type", options=car_df['DriveType'].unique())
    m_Registration_year = st.selectbox(label="Registered Year", options=sorted(car_df['Registration Year'].unique().astype(int)))
    m_modelYear = st.selectbox(label="Model Year", options=sorted(car_df['modelYear'].unique().astype(int)))
    m_Year_of_Manufacture = st.selectbox(label="Year of Manufacture", options=sorted(car_df['Year of Manufacture'].unique()))
    
    m_seats = st.selectbox(label="Number of Seats", options=sorted(car_df['seats'].unique().astype(int)))
    m_ownerNo = st.selectbox(label="Number of Owners", options=sorted(car_df['owner'].unique().astype(int)))
    m_Engine = st.selectbox(label="Engine Displacement", options=sorted(car_df['Engine Displacement'].unique().astype(int)))
    
    m_Insurance = st.selectbox(label="Insurance", options=car_df['Insurance'].unique())
    m_city = st.selectbox(label="City", options=car_df['city'].unique())

    with stylable_container(
        key="red_button",
        css_styles="""
            button {
                background-color: green;
                color: white;
                border-radius: 20px;
                background-image: linear-gradient(90deg, #0575e6 0%, #021b79 100%);
            }
        """
    ):
        pred_price_button = st.button("Estimate Used Car Price")
        
if pred_price_button:
    prediction_value = predict_resale_price()
    st.subtitle(f":green[The estimated used car price is: ₹ {prediction_value / 100000:,.2f} Lakhs]")
