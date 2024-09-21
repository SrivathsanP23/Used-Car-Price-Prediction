import pickle
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_extras.stylable_container import stylable_container

# Streamlit Page Configuration
st.set_page_config(
    page_title="Used Car Price Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Title
st.title(":red[Car Dheko Used Car Price Prediction]")



# Options for user input
Gear = ['4',  '5',  '6',  '7',  '8',  '9', '10']
City_dict = {'Delhi': 2, 'Kolkata': 5, 'Chennai': 1, 'Hyderabad': 3, 'Jaipur': 4, 'Bengaluru': 0}
bt_dict = {'Hatchback': 0, 'SUV': 7, 'Sedan': 8, 'MUV': 4, 'Coupe': 1, 'Minivans': 5,
           'Pickup Trucks': 6, 'Convertibles': 0, 'Hybrids': 3, 'Wagon': 9}
fuel_type_dict = {'Petrol': 4, 'Diesel': 1, 'LPG': 0, 'CNG': 2, 'Electric': 3}
ownerNo = ['0', '1', '2', '3', '4', '5']
Inusrance_validity_dict = {'Third Party insurance': 5, 'Comprehensive': 2, 'Third Party': 4,
                           'Zero Dep': 6, 'Not Available': 3}
Year_of_Manufacture_dict = {'2018': 16, '2017': 15, '2016': 14, '2019': 17,
                            '2021': 19, '2020': 18, '2015': 13, '2014': 12,
                            '2022': 20, '2013': 11, '2012': 10, '2011': 9, '2010': 8,
                            '2009': 7, '2023': 21, '2008': 6, '2007': 5, '2006': 4, '2004': 3,
                            '2005': 2, '2003': 1, '2002': 0}

modelYear_dict = {'2018': 23, '2017': 22, '2016': 21, '2019': 24, '2021': 26, '2020': 25,
                  '2015': 20, '2014': 19, '2022': 27, '2023': 28, '2013': 18, '2012': 17,
                  '2011': 16, '2010': 15, '2009': 14, '2008': 13, '2007': 12, '2006': 11,
                  '2004': 9, '2005': 10, '2003': 8, '2002': 7, '2001': 6, '1998': 3,
                  '1995': 1, '1985': 0, '1999': 4, '2000': 5, '1997': 2}

transmission_dict = {'Automatic': 0, 'Manual': 1}

# Sidebar for user inputs
with st.sidebar:
    st.header("Car Details :Cardheko:")
    m_transmission = st.selectbox(label="Transmission", options=list(transmission_dict.keys()), index=0)
    m_Year_of_Manufacture = st.selectbox(label="Year of Manufacture", options=list(Year_of_Manufacture_dict.keys()), index=0)
    m_modelYear = st.selectbox(label="Model Year", options=list(modelYear_dict.keys()), index=0)
    m_gear = st.selectbox(label="Number of gears", options=Gear, index=0)
    m_city = st.selectbox(label='City Name', options=list(City_dict.keys()), index=0)
    m_Inusrance_validity = st.selectbox(label="Insurance Validity", options=list(Inusrance_validity_dict.keys()), index=0)
    m_ownerNo = st.selectbox(label="Number of Owners", options=ownerNo, index=0)
    m_fuel_type = st.selectbox(label="Fuel Type", options=list(fuel_type_dict.keys()), index=0)
    m_km = st.number_input(label="Kilometers Driven", step=1000, value=0)
    m_bt = st.selectbox(label="Body Type", options=list(bt_dict.keys()), index=0)
    m_mileage = st.number_input(label="Mileage", step=5)

    with stylable_container(
        key="red_button",
        css_styles="""
            button {
                background-color: green;
                color: white;
                border-radius: 20px;
                background-image: linear-gradient(90deg, #0575e6 0%, #021b79 100%);
            }
            """,
    ):  
        pred_price_button = st.button("Estimate Used Car Price")

# Function to predict the resale price
def predict_resale_price():
    # Load pre-trained model
    model = pickle.load(open("E:/Guvi DS/Project3CarDekho_UsedcarPrediction/randomforest.pkl", "rb"))

    # Combine user inputs to an array
    user_data = np.array([[
        int(m_gear),
        int(m_km),
        int(m_mileage),
        int(City_dict.get(m_city)),
        int(Inusrance_validity_dict.get(m_Inusrance_validity)),
        int(m_ownerNo),
        int(fuel_type_dict.get(m_fuel_type)),
        int(bt_dict.get(m_bt)),
        int(Year_of_Manufacture_dict.get(m_Year_of_Manufacture)),
        int(modelYear_dict.get(m_modelYear)),
        int(transmission_dict.get(m_transmission))
    ]])

    prediction = model.predict(user_data)

    return f'The estimated used car price is: ₹ {prediction[0]:,.2f} Lakhs'

if pred_price_button:
    st.write(predict_resale_price())
