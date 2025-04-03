import streamlit as st
import pandas as pd
import pickle
#import gspread
#from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# Load the trained models
electricity_model = pickle.load(open('electricity_pkl.sav', 'rb'))
steam_model = pickle.load(open('steam_pkl.sav', 'rb'))
water_model = pickle.load(open('water_pkl.sav', 'rb'))

# Google Sheets setup
def save_to_google_sheet(data):
    # Define the scope and authenticate
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name('consumption-prediction-a1384af74e5d.json', scope)
    client = gspread.authorize(creds)

    # Open the spreadsheet
    sheet = client.open("Predictions Data").sheet1

    # Get the current date and time
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")  # Format: YYYY-MM-DD HH:MM:SS

    # Append the data with the timestamp
    sheet.append_row([current_time] + data)

# Streamlit UI - Adjusting Container Width
st.set_page_config(page_title="Utility Prediction Model", layout="wide")


st.markdown(
    "<div style='text-align: center; font-size: 46px; font-weight: bold; color: #FF4B4B;'>Utility Prediction Model</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='margin-top: 40px; font-size: 18px;'>"
    "</div>",
    unsafe_allow_html=True,
)

st.write("")

# Layout for Day and Night Input Counts
st.subheader("Enter Machine Day and Night Counts")
st.write("")
st.write("")

# Function for side-by-side input with labels above the fields
def side_by_side_input(label, key_day, key_night):
    st.write(f"**{label} -**")
    col_day, col_night = st.columns([1, 1])
    with col_day:
        st.markdown("<div style='margin-bottom: -10px;'>Day</div>", unsafe_allow_html=True)
        day_value = st.number_input("", min_value=0, step=1, key=key_day, label_visibility='collapsed', value=None)
    with col_night:
        st.markdown("<div style='margin-bottom: -10px;'>Night</div>", unsafe_allow_html=True)
        night_value = st.number_input("", min_value=0, step=1, key=key_night, label_visibility='collapsed', value=None)
    return day_value, night_value

# Input fields for machines
col1, spacer, col2, spacer, col3, spacer, col4 = st.columns([2, 0.5, 2, 0.5, 0.1, 0.5, 2.5])  # Add spacing between the two columns
with col1:
    knitting_day, knitting_night = side_by_side_input("Knitting Machines", 'knit_day', 'knit_night')
    st.write("")
    st.write("")
    bulk_dye_day, bulk_dye_night = side_by_side_input("Bulk Dye Machines", 'bulk_day', 'bulk_night')
    st.write("")
    st.write("")
    sample_dye_day, sample_dye_night = side_by_side_input("Sample Dye Machines", 'sample_day', 'sample_night')
    st.write("")
    st.write("")
    dryers_day, dryers_night = side_by_side_input("Dryers", 'dryers_day', 'dryers_night')
    st.write("")
    st.write("")
    presetting_day, presetting_night = side_by_side_input("Presetting Machines", 'presetting_day', 'presetting_night')
with col2:
    chillers_day, chillers_night = side_by_side_input("Chillers", 'chill_day', 'chill_night')
    st.write("")
    st.write("")
    ahu_day, ahu_night = side_by_side_input("AHU", 'ahu_day', 'ahu_night')
    st.write("")
    st.write("")
    compressor_day, compressor_night = side_by_side_input("Compressors", 'comp_day', 'comp_night')
    st.write("")
    st.write("")
    luwa_day, luwa_night = side_by_side_input("Luwa", 'luwa_day', 'luwa_night')

# Add vertical line by using the following HTML + CSS (between col2 and col3)
# Add vertical line between col2 and col3
with col3:
    st.markdown("<div style='border-left: 2px solid #2f3336; height: 770px;'></div>", unsafe_allow_html=True)

st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("---")
with col4:

    #st.write("---")

    # Prediction Button
    if st.button("Predict Consumption"):
        # Replace None values with 0 for missing inputs
        inputs = [
            knitting_day if knitting_day is not None else 0,
            knitting_night if knitting_night is not None else 0,
            bulk_dye_day if bulk_dye_day is not None else 0,
            bulk_dye_night if bulk_dye_night is not None else 0,
            sample_dye_day if sample_dye_day is not None else 0,
            sample_dye_night if sample_dye_night is not None else 0,
            dryers_day if dryers_day is not None else 0,
            dryers_night if dryers_night is not None else 0,
            presetting_day if presetting_day is not None else 0,
            presetting_night if presetting_night is not None else 0,
            chillers_day if chillers_day is not None else 0,
            chillers_night if chillers_night is not None else 0,
            ahu_day if ahu_day is not None else 0,
            ahu_night if ahu_night is not None else 0,
            compressor_day if compressor_day is not None else 0,
            compressor_night if compressor_night is not None else 0,
            luwa_day if luwa_day is not None else 0,
            luwa_night if luwa_night is not None else 0
        ]

        # Check if all inputs are 0
        if all(value == 0 for value in inputs):
            st.error("Please provide at least one non-zero input to make a prediction.")
        else:
            # Create DataFrame for input
            input_data = {
                'Knitting - D': [inputs[0]], 
                'Knitting - N': [inputs[1]], 
                'Bulk Dye - D': [inputs[2]], 
                'Bulk Dye - N': [inputs[3]], 
                'Sample Dye - D': [inputs[4]], 
                'Sample Dye - N': [inputs[5]], 
                'Dryers - D': [inputs[6]], 
                'Dryers - N': [inputs[7]], 
                'Presetting - D': [inputs[8]], 
                'Presetting - N': [inputs[9]], 
                'Chillers - D': [inputs[10]], 
                'Chillers - N': [inputs[11]], 
                'AHU - D': [inputs[12]], 
                'AHU - N': [inputs[13]], 
                'Compressor - D': [inputs[14]], 
                'Compressor - N': [inputs[15]], 
                'Luwa - D': [inputs[16]], 
                'Luwa - N': [inputs[17]]
            }
            input_df = pd.DataFrame(input_data)

            # Predict using the models
            electricity_pred = electricity_model.predict(input_df)
            steam_pred = steam_model.predict(input_df)
            water_pred = water_model.predict(input_df)

            # Save to Google Sheets
            #save_to_google_sheet(inputs + [electricity_pred[0], steam_pred[0], water_pred[0]])

            # Display Predictions
            st.subheader("Predicted Consumption Results")
            st.write(f"**Electricity Consumption (kWh):** {electricity_pred[0]:.2f}")
            st.write(f"**Steam Consumption (kg):** {steam_pred[0]:.2f}")
            st.write(f"**Water Consumption (Cu.m.):** {water_pred[0]:.2f}")