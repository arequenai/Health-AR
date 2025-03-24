from typing import Dict, Any, Optional
import pandas as pd
from etl import config
import datetime
import os

def get_metrics() -> Optional[Dict[str, Dict[str, Dict[str, Any]]]]:
    """Get all metrics from various sources."""
    try:
        # Read MFP daily data
        df_mfp = pd.read_csv(config.MFP_DAILY_FILE)
        latest_mfp = df_mfp.iloc[-1]

        # Calculate last 7 days net calories
        l7d_net_calories = int(df_mfp.tail(7)['calories_net'].sum())

        # Read Whoop data
        df_whoop = pd.read_csv(config.WHOOP_SLEEP_RECOVERY_FILE)
        latest_whoop = df_whoop.iloc[-1]

        # Read Garmin data
        df_garmin = pd.read_csv(config.GARMIN_DAILY_FILE)
        latest_garmin = df_garmin.iloc[-1]
        
        # Read Glucose data
        glucose_value = '-'
        fasting_glucose = '-'
        mean_glucose = '-'
        
        try:
            # Check if glucose data exists
            glucose_daily_file = 'data/glucose_daily.csv'
            glucose_raw_file = 'data/glucose_raw.csv'
            
            # Get latest fasting glucose and daily mean from daily file
            if os.path.exists(glucose_daily_file):
                df_glucose_daily = pd.read_csv(glucose_daily_file)
                if not df_glucose_daily.empty:
                    # Convert date to datetime for proper comparison
                    df_glucose_daily['date'] = pd.to_datetime(df_glucose_daily['date'])
                    
                    # Get latest record for fasting glucose
                    latest_glucose_day = df_glucose_daily.iloc[-1]
                    fasting_glucose = int(latest_glucose_day['fasting_glucose']) if pd.notna(latest_glucose_day['fasting_glucose']) else '-'
                    
                    # Get mean glucose from the last complete day (yesterday)
                    today = pd.Timestamp.now().date()
                    yesterday = today - datetime.timedelta(days=1)
                    yesterday_data = df_glucose_daily[df_glucose_daily['date'].dt.date == yesterday]
                    
                    if not yesterday_data.empty:
                        mean_glucose = int(yesterday_data.iloc[0]['mean_glucose'])
                    else:
                        # If no data for yesterday, use the latest available day
                        mean_glucose = int(latest_glucose_day['mean_glucose'])
            
            # Get the latest glucose reading from raw file
            if os.path.exists(glucose_raw_file):
                df_glucose_raw = pd.read_csv(glucose_raw_file)
                if not df_glucose_raw.empty:
                    # Convert date to datetime for sorting
                    df_glucose_raw['date'] = pd.to_datetime(df_glucose_raw['date'])
                    
                    # Sort by date descending and get the most recent reading
                    df_glucose_raw = df_glucose_raw.sort_values('date', ascending=False)
                    latest_reading = df_glucose_raw.iloc[0]
                    glucose_value = int(latest_reading['sgv']) if 'sgv' in df_glucose_raw.columns else '-'
        except Exception as e:
            print(f"Error loading glucose data: {e}")
            glucose_value = '-'
            fasting_glucose = '-'
            mean_glucose = '-'
        
        # Calculate last 7 days metrics from Garmin activities instead of daily summaries
        last_7d = df_garmin.tail(7)
        last_7d_altitude = last_7d['floorsAscendedInMeters'].sum()  # Already in meters

        # Calculate running distance from activities
        last_7d_distance = 0
        try:
            # Read Garmin activities data
            df_activities = pd.read_csv(config.GARMIN_ACTIVITIES_FILE)
            
            # Convert date to datetime for comparison
            df_activities['date'] = pd.to_datetime(df_activities['date'])
            
            # Get activities from the last 7 days
            today = pd.Timestamp.now().date()
            week_ago = pd.Timestamp(today - datetime.timedelta(days=7))
            recent_activities = df_activities[df_activities['date'] >= week_ago]
            
            # Filter for running activities
            running_types = ['running', 'treadmill_running', 'trail_running', 'track_running']
            running_activities = recent_activities[recent_activities['type'].str.lower().isin(running_types)]
            
            # Sum distances in kilometers
            if not running_activities.empty and 'distance' in running_activities.columns:
                # Assuming distance is in meters in the activities file
                last_7d_distance = running_activities['distance'].sum() / 1000
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            print(f"Error calculating running distance: {e}")
            last_7d_distance = 0

        # Format sleep time
        sleep_hours = latest_garmin['sleepingSeconds'] / 3600  # Convert seconds to hours
        sleep_hrs = int(sleep_hours)
        sleep_mins = int((sleep_hours - sleep_hrs) * 60)
        sleep_time = f"{sleep_hrs}:{sleep_mins:02d}"

        # Handle calories net label and value
        calories_net = int(latest_mfp['calories_net'])
        if calories_net > 0:
            cal_label = 'kcal over'
        else:
            cal_label = 'kcal rem'
            
        # Get sleep behavior score from Google Journal
        sleep_behaviour_score = '-'
        try:
            if os.path.exists(config.JOURNAL_FILE):
                df_journal = pd.read_csv(config.JOURNAL_FILE)
                if not df_journal.empty and 'sleep_behaviour_score' in df_journal.columns:
                    # Get the most recent score
                    df_journal['date'] = pd.to_datetime(df_journal['date'])
                    df_journal = df_journal.sort_values('date', ascending=False)
                    latest_score = df_journal.iloc[0]['sleep_behaviour_score']
                    if pd.notna(latest_score):
                        sleep_behaviour_score = int(latest_score)
        except Exception as e:
            print(f"Error getting sleep behavior score: {e}")
            sleep_behaviour_score = '-'
            
        # Calculate time spent on strength training from Garmin activities
        strength_minutes = 0
        try:
            # Read Garmin activities data
            df_activities = pd.read_csv(config.GARMIN_ACTIVITIES_FILE)
            
            # Convert date to datetime for comparison
            df_activities['date'] = pd.to_datetime(df_activities['date'])
            
            # Get the last 7 days instead of start of week
            today = pd.Timestamp.now().date()
            seven_days_ago = today - datetime.timedelta(days=7)
            
            # Filter for strength activities in the last 7 days
            strength_types = ['strength_training', 'indoor_cardio', 'fitness_equipment', 'training', 'other']
            last_7d_strength = df_activities[
                (df_activities['date'] >= pd.Timestamp(seven_days_ago)) & 
                (df_activities['type'].str.lower().isin(strength_types))
            ]
            
            # Calculate total duration in minutes
            if not last_7d_strength.empty and 'duration' in last_7d_strength.columns:
                # Convert duration from seconds to minutes
                strength_minutes = int(last_7d_strength['duration'].sum() / 60)
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            strength_minutes = 0
            
        # Format strength time in h:mm format
        strength_hours = strength_minutes // 60
        strength_mins = strength_minutes % 60
        strength_time = f"{strength_hours}:{strength_mins:02d}"
            
        # Read Strength (Google Sheets) data
        try:
            df_strength = pd.read_csv(config.GSHEETS_FILE)
            
            # Ensure date column is in datetime format
            df_strength['date'] = pd.to_datetime(df_strength['date'])
            
            # Find max pull-ups in the last recorded day with pull-ups
            # First, make sure the 'exercise' and 'reps' columns exist
            if 'exercise' in df_strength.columns and 'reps' in df_strength.columns:
                # Filter for pull-up exercises
                pullup_df = df_strength[df_strength['exercise'].str.contains('pull-up|pullup|pull up', case=False, na=False)]
                
                if not pullup_df.empty:
                    # Convert reps to numeric, handling any errors
                    pullup_df.loc[:, 'reps'] = pd.to_numeric(pullup_df['reps'], errors='coerce')
                    
                    # Get the most recent date with pull-ups
                    last_pullup_date = pullup_df['date'].max()
                    
                    # Get the max reps for that date
                    last_day_pullups = pullup_df[pullup_df['date'] == last_pullup_date]['reps'].max()
                    max_pullups = int(last_day_pullups) if not pd.isna(last_day_pullups) else '-'
                else:
                    max_pullups = '-'
            else:
                max_pullups = '-'
                
            # Count entries in the last 7 days
            today = pd.Timestamp.now().date()
            week_ago = pd.Timestamp(today - datetime.timedelta(days=7))
            recent_entries = df_strength[df_strength['date'] >= week_ago]
            entries_last_7d = len(recent_entries)
            
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            # If file doesn't exist or is empty, use placeholder values
            max_pullups = '-'
            entries_last_7d = 0

        metrics = {
            'nutrition': {
                'primary': {'value': abs(calories_net), 'label': cal_label, 'color_value': calories_net},
                'secondary1': {'value': l7d_net_calories, 'label': 'net above L7d'},
                'secondary2': {'value': int(latest_mfp['protein']), 'label': 'protein g'}
            },
            'glucose': {
                'primary': {'value': glucose_value, 'label': 'mg/dL'},
                'secondary1': {'value': fasting_glucose, 'label': 'fasting'}, 
                'secondary2': {'value': mean_glucose, 'label': 'mean day'}
            },
            'recovery': {
                'primary': {'value': '-' if 'max_sleep_body_battery' not in latest_garmin or pd.isna(latest_garmin['max_sleep_body_battery']) or latest_garmin['max_sleep_body_battery'] == 0 else int(latest_garmin['max_sleep_body_battery']), 'label': 'max battery'},
                'secondary1': {'value': int(latest_garmin['bodyBatteryMostRecentValue']), 'label': 'battery now'},
                'secondary2': {'value': int(latest_garmin['averageStressLevel']), 'label': 'stress'}
            },
            'sleep': {
                'primary': {'value': int(latest_garmin['sleep_score']), 'label': 'sleep'},
                'secondary1': {'value': sleep_time, 'label': 'hrs in bed'},
                'secondary2': {'value': sleep_behaviour_score, 'label': 'bed habits'}
            },
            'running': {
                'primary': {'value': int(latest_garmin['TSB']), 'label': 'TSB'},
                'secondary1': {'value': f"{last_7d_distance:.1f}", 'label': 'km run L7d'},
                'secondary2': {'value': f"{int(last_7d_altitude)}", 'label': 'm gain L7d'}
            },
            'strength': {
                'primary': {'value': strength_time, 'label': 'time L7d', 'color_value': strength_minutes},
                'secondary1': {'value': max_pullups, 'label': 'pullups'},
                'secondary2': {'value': entries_last_7d, 'label': 'sets L7d'}
            }
        }
        return metrics
    except Exception as e:
        print(f"Error getting metrics: {e}")  # For debugging
        return None 