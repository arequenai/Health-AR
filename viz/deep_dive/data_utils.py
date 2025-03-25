import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from etl import config
from etl.garmin_etl import init_garmin, get_body_battery_data_for_day, get_stress_data_for_day

# Initialize the Garmin client globally to avoid re-initialization for each data fetch
garmin_client = None
try:
    email = os.getenv("USERNAME_G")
    password = os.getenv("PASSWORD_G")
    if email and password:
        garmin_client = init_garmin(email, password)
except Exception as e:
    print(f"Error initializing Garmin client: {e}")

def get_day_data(selected_date):
    """
    Load and aggregate data for a specific day from multiple sources.
    
    Args:
        selected_date (datetime): The date to get data for
        
    Returns:
        dict: Dictionary with all data sources for the selected day
    """
    data = {}
    
    # Load Garmin daily data
    try:
        df_garmin = pd.read_csv(config.GARMIN_DAILY_FILE)
        df_garmin['date'] = pd.to_datetime(df_garmin['date'])
        day_garmin = df_garmin[df_garmin['date'] == selected_date].iloc[0] if not df_garmin[df_garmin['date'] == selected_date].empty else None
        data['garmin'] = day_garmin
    except Exception as e:
        print(f"Error loading Garmin data: {e}")
        data['garmin'] = None
    
    # Load Glucose data
    try:
        df_glucose_daily = pd.read_csv(config.GLUCOSE_DAILY_FILE)
        df_glucose_daily['date'] = pd.to_datetime(df_glucose_daily['date'])
        day_glucose_daily = df_glucose_daily[df_glucose_daily['date'] == selected_date].iloc[0] if not df_glucose_daily[df_glucose_daily['date'] == selected_date].empty else None
        data['glucose_daily'] = day_glucose_daily
        
        # Get raw glucose data for the day
        df_glucose_raw = pd.read_csv(config.GLUCOSE_RAW_FILE)
        df_glucose_raw['datetime'] = pd.to_datetime(df_glucose_raw['date'])
        df_glucose_raw['date'] = df_glucose_raw['datetime'].dt.date
        # Filter for the specific day
        selected_date_only = pd.Timestamp(selected_date).date()
        glucose_day_data = df_glucose_raw[df_glucose_raw['date'] == selected_date_only].copy()
        
        # Ensure we have a 'glucose' column (might be sgv in some files)
        if 'sgv' in glucose_day_data.columns and 'glucose' not in glucose_day_data.columns:
            glucose_day_data['glucose'] = glucose_day_data['sgv']
            
        data['glucose_raw'] = glucose_day_data
    except Exception as e:
        print(f"Error loading glucose data: {e}")
        data['glucose_daily'] = None
        data['glucose_raw'] = None
    
    # Load meals data
    try:
        df_meals = pd.read_csv(config.MFP_MEALS_FILE)
        df_meals['date'] = pd.to_datetime(df_meals['date'])
        day_meals = df_meals[df_meals['date'] == selected_date]
        data['meals'] = day_meals
    except Exception as e:
        print(f"Error loading meal data: {e}")
        data['meals'] = pd.DataFrame()
    
    # Load activities data
    try:
        df_activities = pd.read_csv(config.GARMIN_ACTIVITIES_FILE)
        df_activities['date'] = pd.to_datetime(df_activities['date'])
        day_activities = df_activities[df_activities['date'] == selected_date]
        data['activities'] = day_activities
    except Exception as e:
        print(f"Error loading activities data: {e}")
        data['activities'] = pd.DataFrame()
        
    return data

def get_body_battery_data(selected_date):
    """
    Get body battery data for a specific date.
    First tries to get data directly from Garmin API,
    falls back to reading from raw files if API fails.
    
    Args:
        selected_date (datetime): The date to get data for
        
    Returns:
        DataFrame: Body battery data with datetime and value columns
    """
    global garmin_client
    
    # First try to get data from Garmin API
    if garmin_client:
        try:
            # Convert to datetime.date if it's a Timestamp
            if hasattr(selected_date, 'date'):
                date_obj = selected_date.date()
            else:
                date_obj = selected_date
                
            df = get_body_battery_data_for_day(garmin_client, date_obj)
            if not df.empty:
                return df
        except Exception as e:
            print(f"Error getting body battery data from API: {e}")
            # Fall back to file-based approach
    
    # Fall back to the file-based approach if API fails or returns no data
    try:
        date_str = selected_date.strftime('%Y-%m-%d')
        file_path = os.path.join('data', 'raw', 'garmin', 'daily', f'{date_str}_body_battery.json')
        
        if not os.path.exists(file_path):
            return None
            
        import json
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Convert to DataFrame
        records = []
        for point in data:
            if 'timestamp' in point and 'bodyBatteryValue' in point:
                timestamp = datetime.fromtimestamp(point['timestamp'] / 1000)
                records.append({
                    'datetime': timestamp,
                    'body_battery': point['bodyBatteryValue']
                })
                
        df = pd.DataFrame(records)
        df = df.sort_values('datetime')
        return df
    except Exception as e:
        print(f"Error getting body battery data from file: {e}")
        return None

def get_stress_data(selected_date):
    """
    Get stress data for a specific date.
    First tries to get data directly from Garmin API,
    falls back to reading from raw files if API fails.
    
    Args:
        selected_date (datetime): The date to get data for
        
    Returns:
        DataFrame: Stress data with datetime and value columns
    """
    global garmin_client
    
    # First try to get data from Garmin API
    if garmin_client:
        try:
            # Convert to datetime.date if it's a Timestamp
            if hasattr(selected_date, 'date'):
                date_obj = selected_date.date()
            else:
                date_obj = selected_date
                
            df = get_stress_data_for_day(garmin_client, date_obj)
            if not df.empty:
                return df
        except Exception as e:
            print(f"Error getting stress data from API: {e}")
            # Fall back to file-based approach
    
    # Fall back to the file-based approach if API fails or returns no data
    try:
        date_str = selected_date.strftime('%Y-%m-%d')
        file_path = os.path.join('data', 'raw', 'garmin', 'daily', f'{date_str}_stress.json')
        
        if not os.path.exists(file_path):
            return None
            
        import json
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Convert to DataFrame
        records = []
        for point in data:
            if 'timestamp' in point and 'stressLevel' in point:
                timestamp = datetime.fromtimestamp(point['timestamp'] / 1000)
                records.append({
                    'datetime': timestamp,
                    'stress': point['stressLevel']
                })
                
        df = pd.DataFrame(records)
        df = df.sort_values('datetime')
        return df
    except Exception as e:
        print(f"Error getting stress data from file: {e}")
        return None

def get_sleep_data(selected_date):
    """
    Get sleep data for a specific date.
    Uses the Garmin daily data to extract sleep information.
    
    Args:
        selected_date (datetime): The date to get data for
        
    Returns:
        dict: Dictionary with sleep start, end, and duration
    """
    try:
        df_garmin = pd.read_csv(config.GARMIN_DAILY_FILE)
        df_garmin['date'] = pd.to_datetime(df_garmin['date'])
        
        row = df_garmin[df_garmin['date'] == selected_date]
        if row.empty:
            return None
            
        row = row.iloc[0]
        
        # Calculate sleep time based on garmin data
        if 'sleepingSeconds' not in row or pd.isna(row['sleepingSeconds']):
            return None
            
        # We only have the duration, not the actual start and end times
        # For the purpose of visualization, assume sleep ends at 8 AM
        sleep_duration_hours = row['sleepingSeconds'] / 3600
        sleep_end = datetime.combine(selected_date.date(), datetime.strptime('08:00', '%H:%M').time())
        sleep_start = sleep_end - timedelta(hours=sleep_duration_hours)
        
        return {
            'sleep_start': sleep_start,
            'sleep_end': sleep_end,
            'sleep_duration': sleep_duration_hours
        }
    except Exception as e:
        print(f"Error getting sleep data: {e}")
        return None 