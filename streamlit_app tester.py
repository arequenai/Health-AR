import streamlit as st
import pandas as pd
from etl import init_garmin, get_garmin_data, init_whoop, get_sleep_recovery_data, init_mfp, get_meal_data, get_meal_daily, config
import os

def run_garmin_etl():
    """
    Execute Garmin ETL to update data.
    """
    try:
        email = os.getenv("USERNAME_G")
        password = os.getenv("PASSWORD_G")
        
        garmin_client = init_garmin(email, password)
        df = get_garmin_data(garmin_client)
        
        if df is not None:
            df.to_csv(config.GARMIN_DAILY_FILE, index=False)
            st.success("Garmin ETL executed successfully")
        else:
            st.error("Garmin ETL failed to execute")
    except Exception as e:
        st.error(f"Garmin ETL failed: {str(e)}")

def run_whoop_etl():
    """
    Execute Whoop ETL to update data.
    """
    try:
        un = os.getenv("USERNAME_W")
        pw = os.getenv("PASSWORD_W")
        
        whoop_client = init_whoop(un, pw)
        df = get_sleep_recovery_data(whoop_client)
        
        if df is not None:
            df.to_csv(config.WHOOP_SLEEP_RECOVERY_FILE, index=False)
            st.success("Whoop ETL executed successfully")
        else:
            st.error("Whoop ETL failed to execute")
    except Exception as e:
        st.error(f"Whoop ETL failed: {str(e)}")

def run_mfp_etl():
    """
    Execute MyFitnessPal ETL to update data.
    """
    try:
        client = init_mfp()
        get_meal_data(client, config.MFP_MEALS_FILE)
        get_meal_daily(client, config.MFP_DAILY_FILE)
        st.success("MyFitnessPal ETL executed successfully")
    except Exception as e:
        st.error(f"MyFitnessPal ETL failed: {str(e)}")

def get_last_metrics():
    """
    Read CSV files and return metrics from the last day.
    """
    try:
        # Garmin data
        df_garmin = pd.read_csv(config.GARMIN_DAILY_FILE)
        if df_garmin.empty:
            st.warning("Garmin file is empty")
            return None, None, None
        last_garmin = df_garmin.iloc[-1]
        distance = last_garmin.get("totalDistanceMeters", None)
        
        # Whoop data
        df_whoop = pd.read_csv(config.WHOOP_SLEEP_RECOVERY_FILE)
        if df_whoop.empty:
            st.warning("Whoop file is empty")
            return distance, None, None
        last_whoop = df_whoop.iloc[-1]
        recovery = last_whoop.get("recovery_score", None)
        
        # MFP data
        df_mfp = pd.read_csv(config.MFP_DAILY_FILE)
        if df_mfp.empty:
            st.warning("MyFitnessPal file is empty")
            return distance, recovery, None
        last_mfp = df_mfp.iloc[-1]
        caloric_deficit = last_mfp.get("calories_net", None)
        
        return distance, recovery, caloric_deficit
    except Exception as e:
        st.error(f"Error reading CSV files: {str(e)}")
        return None, None, None

st.title("Fitness Dashboard")

# Update data buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Update Garmin Data"):
        run_garmin_etl()
with col2:
    if st.button("Update Whoop Data"):
        run_whoop_etl()
with col3:
    if st.button("Update MFP Data"):
        run_mfp_etl()

# Show last day metrics
distance, recovery, caloric_deficit = get_last_metrics()
if distance is not None:
    st.write(f"Last day's running distance: {distance:.2f} meters")
if recovery is not None:
    st.write(f"Last day's recovery score: {recovery}%")
if caloric_deficit is not None:
    st.write(f"Last day's caloric balance: {caloric_deficit:.0f} calories")
