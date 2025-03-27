import pandas as pd
import os
import datetime
from . import config

def load_data():
    """Load all necessary data files."""
    data = {}
    
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Adjust paths to use absolute paths from project root
    mfp_daily_file = os.path.join(project_root, config.MFP_DAILY_FILE)
    garmin_daily_file = os.path.join(project_root, config.GARMIN_DAILY_FILE)
    glucose_daily_file = os.path.join(project_root, config.GLUCOSE_DAILY_FILE)
    garmin_activities_file = os.path.join(project_root, config.GARMIN_ACTIVITIES_FILE)
    journal_file = os.path.join(project_root, config.JOURNAL_FILE)
    
    # Load MFP data for nutrition metrics
    if os.path.exists(mfp_daily_file):
        nutrition_df = pd.read_csv(mfp_daily_file)
        nutrition_df['date'] = pd.to_datetime(nutrition_df['date'])
        data['nutrition'] = nutrition_df
    
    # Load Garmin data for recovery, sleep, and running metrics
    if os.path.exists(garmin_daily_file):
        data['garmin'] = pd.read_csv(garmin_daily_file)
        data['garmin']['date'] = pd.to_datetime(data['garmin']['date'])
        
        # Add some pre-calculated fields
        if 'recovery' in data['garmin']:
            data['garmin']['battery'] = data['garmin']['recovery']
            data['garmin']['max_battery'] = data['garmin']['battery'] * 100
        
        # Make sure max_battery column exists - add it from max_sleep_body_battery if needed
        if 'max_sleep_body_battery' in data['garmin'].columns and 'max_battery' not in data['garmin'].columns:
            data['garmin']['max_battery'] = data['garmin']['max_sleep_body_battery']
    
    # Load Glucose data
    if os.path.exists(glucose_daily_file):
        data['glucose'] = pd.read_csv(glucose_daily_file)
        data['glucose']['date'] = pd.to_datetime(data['glucose']['date'])
    
    # Load Activities data for strength metrics
    if os.path.exists(garmin_activities_file):
        data['activities'] = pd.read_csv(garmin_activities_file)
        data['activities']['date'] = pd.to_datetime(data['activities']['date'])
        
        # Calculate and add strength training time
        strength_types = ['strength_training', 'indoor_cardio', 'fitness_equipment', 'training', 'other']
        strength_df = data['activities'][data['activities']['type'].str.lower().isin(strength_types)].copy()
        
        # Aggregate by day
        if not strength_df.empty:
            daily_strength = strength_df.groupby(strength_df['date'].dt.date).agg(
                strength_minutes=('duration', lambda x: (x.sum() / 60))  # Convert seconds to minutes
            ).reset_index()
            daily_strength.columns = ['date', 'strength_minutes']
            daily_strength['date'] = pd.to_datetime(daily_strength['date'])
            
            # Create a strength dataset
            data['strength'] = daily_strength
        
        # Create a simple running dataset
        # Filter for running activities (keep it broad to catch all running types)
        running_activities = data['activities'][
            data['activities']['type'].str.lower().str.contains('run', na=False)
        ].copy()
        
        if not running_activities.empty:
            # Group by day and calculate daily totals
            daily_running = running_activities.groupby(running_activities['date'].dt.date).agg({
                'distance': 'sum',  # Total distance in meters
                'duration': 'sum',  # Total time in seconds
            }).reset_index()
            
            # Convert units to more readable values
            daily_running['km_run'] = daily_running['distance'] / 1000  # Convert to kilometers
            daily_running['minutes_run'] = daily_running['duration'] / 60  # Convert to minutes
            
            # Clean up and format
            daily_running = daily_running[['date', 'km_run', 'minutes_run']]
            daily_running['date'] = pd.to_datetime(daily_running['date'])
            
            # Add to our data dictionary
            data['running'] = daily_running
    
    # Load Journal data for sleep behavior
    if os.path.exists(journal_file):
        data['journal'] = pd.read_csv(journal_file)
        data['journal']['date'] = pd.to_datetime(data['journal']['date'])
    
    return data 