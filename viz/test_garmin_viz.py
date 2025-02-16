import streamlit as st
import pandas as pd
from etl import init_garmin, get_garmin_data
import os

def run_etl():
    """
    Ejecuta el script ETL para actualizar los datos de Garmin.
    """
    try:
        # Get credentials from environment
        email = os.getenv("USERNAME_G")
        password = os.getenv("PASSWORD_G")
        
        # Initialize Garmin client and get data
        garmin_client = init_garmin(email, password)
        df = get_garmin_data(garmin_client)
        
        if df is not None:
            df.to_csv('data/garmin_daily.csv', index=False)
            st.success("ETL ejecutado correctamente!")
        else:
            st.error("No se obtuvieron datos del ETL.")
    except Exception as e:
        st.error(f"Error al ejecutar el ETL: {str(e)}")

def get_last_distance():
    """
    Lee el archivo CSV generado por el ETL y retorna la distancia corrida del último día.
    """
    try:
        df = pd.read_csv("data/garmin_daily.csv")
        if df.empty:
            st.warning("El archivo CSV está vacío.")
            return None
        last_entry = df.iloc[-1]
        distance = last_entry.get("totalDistanceMeters", None)
        return distance
    except Exception as e:
        st.error("Error al leer el archivo CSV: " + str(e))
        return None

st.title("Dashboard de Garmin")

# Botón para actualizar datos
if st.button("Actualizar datos Garmin"):
    run_etl()

# Mostrar la distancia del último día
distance = get_last_distance()
if distance is not None:
    st.write(f"La distancia corrida en el último día es: {distance}")
else:
    st.write("No se pudo obtener la distancia.")
