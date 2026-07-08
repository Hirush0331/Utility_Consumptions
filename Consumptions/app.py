import streamlit as st
import pandas as pd
import pickle
#import gspread
#from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# Load the trained models
electricity_model = pickle.load(open('Consumptions/electricity_pkl.sav', 'rb'))
steam_model = pickle.load(open('Consumptions/steam_pkl.sav', 'rb'))
water_model = pickle.load(open('Consumptions/water_pkl.sav', 'rb'))

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


st.title("🏭 Utility Consumption Prediction")
st.caption("Predict electricity, steam and water consumption based on machine day and night operating counts.")
st.divider()
st.subheader("Machine Operating Details")

# Function for side-by-side input with labels above the fields
def side_by_side_input(label, key_day, key_night):
    st.markdown(f"**{label}**")
    c1,c2=st.columns(2)
    with c1:
        day_value=st.number_input("Day",min_value=0,step=1,key=key_day,value=0)
    with c2:
        night_value=st.number_input("Night",min_value=0,step=1,key=key_night,value=0)
    st.markdown("<br>",unsafe_allow_html=True)
    return day_value,night_value

# Input fields for machines
col1, col2 = st.columns(2, gap='large')  # Add spacing between the two columns
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


st.divider()

predict_col=st.container()
with predict_col:

    #st.write("---")

    # Prediction Button
    if st.button("🔮 Predict Consumption", use_container_width=True):
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
            st.divider()
            st.subheader("Prediction Results")
            c1,c2,c3=st.columns(3)
            c1.metric("⚡ Electricity",f"{electricity_pred[0]:,.2f} kWh")
            c2.metric("♨️ Steam",f"{steam_pred[0]:,.2f} kg")
            c3.metric("💧 Water",f"{water_pred[0]:,.2f} Cu.m.")