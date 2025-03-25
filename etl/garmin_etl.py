from dotenv import load_dotenv
load_dotenv("Credentials.env")

import datetime
from datetime import datetime as dt  # Add this to fix timestamp conversion issues
import pandas as pd
import logging
import os
from getpass import getpass
from garth.exc import GarthHTTPError
from garminconnect import Garmin, GarminConnectAuthenticationError
import warnings
from etl import config
import numpy as np

warnings.filterwarnings('ignore', category=FutureWarning)

# Change logging level to INFO
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

email = os.getenv("USERNAME_G")
password = os.getenv("PASSWORD_G")

# Define essential fields to keep
ESSENTIAL_FIELDS = [
    # Currently used
    'totalDistanceMeters', 'floorsAscendedInMeters', 'sleepingSeconds',
    'bodyBatteryMostRecentValue', 'averageStressLevel',
    # Important for future use
    'restingHeartRate', 'maxHeartRate', 'intensityMinutes',
    'activeKilocalories', 'totalSteps', 'floorsAscended'
]

def init_garmin(email, password):
    tokenstore = os.getenv("GARMINTOKENS") or "~/.garminconnect"
    try:
        garmin = Garmin()
        garmin.login(tokenstore)
    except (FileNotFoundError, GarthHTTPError, GarminConnectAuthenticationError):
        garmin = Garmin(email=email, password=password)
        garmin.login()
        garmin.garth.dump(tokenstore)
    return garmin

def get_garmin_data(garmin_client, start_date=None):
    """Get Garmin data."""
    if start_date is None:
        start_date = config.DATA_START_DATE
        
    api = garmin_client
    end_date = datetime.date.today()
    data_file = config.GARMIN_DAILY_FILE
    
    existing_data = None
    if os.path.exists(data_file):
        existing_data = pd.read_csv(data_file)
        if len(existing_data) > 0:
            last_date = pd.to_datetime(existing_data['date'].max()).date()
            start_date = last_date
            existing_data = existing_data[existing_data['date'] < start_date.strftime('%Y-%m-%d')]
    
    # Check if today's data already exists with sleep score
    today_str = end_date.strftime('%Y-%m-%d')
    today_data_exists = False
    if existing_data is not None and today_str in existing_data['date'].values:
        today_row = existing_data[existing_data['date'] == today_str]
        if 'sleep_score' in today_row.columns and today_row['sleep_score'].iloc[0] > 0:
            today_data_exists = True
            logger.info(f"Today's sleep score already exists, skipping sleep data retrieval")
    
    # Ensure existing_data has max_sleep_body_battery column
    if existing_data is not None and 'max_sleep_body_battery' not in existing_data.columns:
        existing_data['max_sleep_body_battery'] = 0
        logger.info("Added max_sleep_body_battery column to existing data")
    
    data_list = []
    current_date = start_date
    while current_date <= end_date:
        try:
            stats = api.get_stats(current_date)
            race_predictor = api.get_race_predictions(startdate=current_date, enddate=current_date, _type='daily')
            
            data_dict = {'date': current_date.strftime('%Y-%m-%d')}
            # Only keep essential fields
            for field in ESSENTIAL_FIELDS:
                data_dict[field] = stats.get(field, 0)

            # Add race predictions if available
            if race_predictor and len(race_predictor) > 0:
                race_data = race_predictor[0]
                for race in ['time5K', 'time10K', 'timeHalfMarathon', 'timeMarathon']:
                    if race in race_data:
                        seconds = race_data[race]
                        hours = int(seconds // 3600)
                        minutes = int((seconds % 3600) // 60)
                        prediction = f"{hours}:{minutes:02d}"
                        data_dict[f'{race[4:]}_prediction'] = prediction

            # Only get sleep data if we need it (not today or today's data doesn't exist yet)
            if current_date != end_date or not today_data_exists:
                sleep_score, max_body_battery = get_sleep_score(api, current_date)
                data_dict['sleep_score'] = sleep_score
                data_dict['max_sleep_body_battery'] = max_body_battery
            else:
                # Skip sleep API call for today if we already have it
                data_dict['sleep_score'] = 0  # Will be filled from existing data later
                data_dict['max_sleep_body_battery'] = 0  # Will be filled from existing data later

            data_list.append(data_dict)
        except Exception as e:
            logger.warning(f"Failed to get data for {current_date}: {str(e)}")
        current_date += datetime.timedelta(days=1)

    df = pd.DataFrame(data_list)
    
    # Get activities and calculate daily training load
    try:
        activities = api.get_activities_by_date(start_date, end_date)
        activity_data = []
        for activity in activities:
            activity_date = pd.to_datetime(activity['startTimeLocal']).date().strftime('%Y-%m-%d')
            activity_dict = {
                'date': activity_date,
                'training_load': activity.get('activityTrainingLoad', 0)
            }
            activity_data.append(activity_dict)
            
        if activity_data:
            activities_df = pd.DataFrame(activity_data)
            daily_load = activities_df.groupby('date')['training_load'].sum().reset_index()
            
            # Merge with daily stats
            df = pd.merge(df, daily_load, on='date', how='left')
            df['training_load'] = df['training_load'].fillna(0)
    except Exception as e:
        logger.warning(f"Failed to process activities: {str(e)}")

    df.fillna(0, inplace=True)
    if existing_data is not None:
        df = pd.concat([existing_data, df], ignore_index=True)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df = df.sort_values('date').drop_duplicates(subset=['date'], keep='last')
    
    # After merging training_load data
    try:
        # Ensure training_load exists and is 0 where missing
        df['training_load'] = df['training_load'].fillna(0)
        
        # Create warmup period with zeros (42 days before our first date)
        first_date = pd.to_datetime(df['date'].min())
        warmup_dates = pd.date_range(end=first_date - pd.Timedelta(days=1), periods=42)
        warmup_df = pd.DataFrame({
            'date': warmup_dates.strftime('%Y-%m-%d'),
            'training_load': [0] * 42
        })
        
        # Combine warmup and actual data
        df_with_warmup = pd.concat([warmup_df, df]).sort_values('date')
        
        # Calculate CTL and ATL using exponential decay
        ctl_decay = np.exp(-1/42)  # 42-day time constant for CTL
        atl_decay = np.exp(-1/7)   # 7-day time constant for ATL
        
        # Initialize arrays
        loads = df_with_warmup['training_load'].values
        ctl = np.zeros(len(loads))
        atl = np.zeros(len(loads))
        
        # Calculate CTL and ATL
        for i in range(1, len(loads)):
            ctl[i] = loads[i] * (1 - ctl_decay) + ctl[i-1] * ctl_decay
            atl[i] = loads[i] * (1 - atl_decay) + atl[i-1] * atl_decay
        
        # Add to dataframe
        df_with_warmup['CTL'] = ctl
        df_with_warmup['ATL'] = atl
        df_with_warmup['TSB'] = df_with_warmup['CTL'] - df_with_warmup['ATL']
        
        # Keep only the actual data period
        df = df_with_warmup[df_with_warmup['date'].isin(df['date'])].copy()
        
        # Round the values
        df['CTL'] = df['CTL'].round(1)
        df['ATL'] = df['ATL'].round(1)
        df['TSB'] = df['TSB'].round(1)
        
    except Exception as e:
        logger.warning(f"Failed to calculate training metrics: {str(e)}")

    return df

def get_garmin_activities(garmin_client, start_date=datetime.date(2024, 3, 16)):
    """Get detailed activity data for analysis from start_date to today.
    If the existing data file is mostly empty, it will pull all data since March 16, 2024.
    Otherwise, it will pull the last week of data and merge it with existing data."""
    api = garmin_client
    end_date = datetime.date.today()
    data_file = config.GARMIN_ACTIVITIES_FILE
    
    # Check existing data
    if os.path.exists(data_file):
        existing_data = pd.read_csv(data_file)
        if len(existing_data) > 0:
            # If we have data, start from the last date minus 2 days
            last_date = pd.to_datetime(existing_data['date'].max()).date()
            start_date = last_date
            logger.info(f"Found existing data, pulling from {start_date} onwards")
            
            # Remove the last week of data from existing_data to avoid duplicates
            cutoff_date = start_date.strftime('%Y-%m-%d')
            existing_data = existing_data[existing_data['date'] < cutoff_date]
        else:
            logger.info("Existing data is empty, pulling all data since March 16, 2024")
            start_date = datetime.date(2024, 3, 16)
            existing_data = None
    else:
        logger.info("No existing data found, pulling all data since March 16, 2024")
        start_date = config.DATA_START_DATE
        existing_data = None
    
    logger.info(f"Getting Garmin activities from {start_date} to {end_date}")
    
    try:
        activities = api.get_activities_by_date(start_date, end_date)
    except Exception as e:
        logger.error(f"Failed to get activities: {str(e)}")
        return None
    
    activity_data = []
    total_activities = len(activities)
    for i, activity in enumerate(activities, 1):
        activity_date = pd.to_datetime(activity['startTimeLocal']).date().strftime('%Y-%m-%d')
        
        activity_dict = {
            'date': activity_date,
            'start_time_local': activity.get('startTimeLocal', ''),
            'start_time_utc': activity.get('startTimeGMT', ''),
            'type': activity.get('activityType', {}).get('typeKey', 'unknown'),
            'duration': activity.get('duration', 0),
            'distance': activity.get('distance', 0),
            'training_load': activity.get('activityTrainingLoad', 0),
            'aerobic_te': activity.get('aerobicTrainingEffect', 0),
            'anaerobic_te': activity.get('anaerobicTrainingEffect', 0),
            'avg_hr': activity.get('averageHR', 0),
            'max_hr': activity.get('maxHR', 0),
            'avg_power': activity.get('avgPower', 0),
            'norm_power': activity.get('normPower', 0),
            'intensity_factor': activity.get('intensityFactor', 0),
            'tss': activity.get('trainingStressScore', 0)
        }
        activity_data.append(activity_dict)
        
        if i % 10 == 0:  # Log every 10 activities
            logger.info(f"Processed {i}/{total_activities} activities ({(i/total_activities)*100:.1f}%)")
    
    if not activity_data:
        logger.warning("No activities found for the specified date range")
        return None
    
    df = pd.DataFrame(activity_data)
    
    # If we have existing data, append the new data
    if existing_data is not None:
        df = pd.concat([existing_data, df], ignore_index=True)
    
    # Convert date to datetime for proper sorting and deduplication
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates(subset=['date', 'type', 'duration', 'start_time_local'], keep='last')
    # Convert back to string format
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    
    return df

def get_sleep_score(api, date):
    """Get sleep score and max body battery for a given date.
    
    Args:
        api: Garmin API client
        date: Date to get sleep score for (datetime.date object)
        
    Returns:
        tuple: (sleep_score, max_sleep_body_battery) - both 0 if not available
    """
    try:
        formatted_date = date.strftime('%Y-%m-%d')
        sleep_data = api.get_sleep_data(formatted_date)
        
        # Default values
        sleep_score = 0
        max_body_battery = 0
        
        # Extract sleep score
        if sleep_data:
            sleep_dto = sleep_data.get('dailySleepDTO', {})
            scores = sleep_dto.get('sleepScores', {})
            overall = scores.get('overall', {})
            sleep_score = overall.get('value', 0)
            
            # Extract max body battery during sleep if available
            if "sleepBodyBattery" in sleep_data:
                battery_values = [point.get("value", 0) for point in sleep_data.get("sleepBodyBattery", [])]
                if battery_values:
                    max_body_battery = max(battery_values)
                    
        return sleep_score, max_body_battery
    except Exception as e:
        logger.warning(f"Failed to get sleep data for {date}: {str(e)}")
        return 0, 0

def get_body_battery_data_for_day(garmin_client, date):
    """
    Get detailed body battery data for a specific day.
    
    Args:
        garmin_client: Initialized Garmin API client
        date: Date to get body battery data for (datetime.date object)
        
    Returns:
        DataFrame with timestamps and body battery values for the day
    """
    try:
        # Format the date for API call
        formatted_date = date.strftime('%Y-%m-%d')
        
        # Get the stress data for the day, which also contains body battery data
        stress_data = garmin_client.get_stress_data(formatted_date)
        
        # Extract body battery points
        body_battery_data = []
        
        if stress_data and 'bodyBatteryValuesArray' in stress_data:
            for point in stress_data['bodyBatteryValuesArray']:
                try:
                    # Each point is [timestamp, "MEASURED", bodyBatteryLevel, version]
                    if len(point) >= 3:
                        timestamp = dt.fromtimestamp(point[0] / 1000.0)
                        body_battery = point[2]  # The battery level is the third element
                        
                        body_battery_data.append({
                            'datetime': timestamp,
                            'body_battery': body_battery
                        })
                except Exception as e:
                    logger.warning(f"Error processing body battery point: {e}")
                    continue
        
        # Create DataFrame
        if body_battery_data:
            df = pd.DataFrame(body_battery_data)
            df.sort_values('datetime', inplace=True)
            return df
        else:
            return pd.DataFrame(columns=['datetime', 'body_battery'])
            
    except Exception as e:
        logger.warning(f"Failed to get body battery data for {date}: {str(e)}")
        return pd.DataFrame(columns=['datetime', 'body_battery'])

def get_stress_data_for_day(garmin_client, date):
    """
    Get detailed stress data for a specific day.
    
    Args:
        garmin_client: Initialized Garmin API client
        date: Date to get stress data for (datetime.date object)
        
    Returns:
        DataFrame with timestamps and stress values for the day
    """
    try:
        # Format the date for API call
        formatted_date = date.strftime('%Y-%m-%d')
        
        # Get the stress data for the day
        stress_data_response = garmin_client.get_stress_data(formatted_date)
        
        # Extract stress points
        stress_points = []
        
        if stress_data_response and 'stressValuesArray' in stress_data_response:
            for point in stress_data_response['stressValuesArray']:
                try:
                    # Each point is [timestamp, stressLevel]
                    if len(point) >= 2:
                        timestamp = dt.fromtimestamp(point[0] / 1000.0)
                        stress_level = point[1]
                        
                        # Include points with -1 as these will be processed later
                        stress_points.append({
                            'datetime': timestamp,
                            'stress': stress_level
                        })
                except Exception as e:
                    logger.warning(f"Error processing stress point: {e}")
                    continue
        
        # Create DataFrame
        if not stress_points:
            return pd.DataFrame(columns=['datetime', 'stress'])
            
        df = pd.DataFrame(stress_points)
        df.sort_values('datetime', inplace=True)
        
        # Process -1 and -2 values
        # First, replace -2 with NaN (these are unusable values)
        df['stress'] = df['stress'].replace(-2, np.nan)
        
        # Replace -1 with NaN for processing
        df['stress'] = df['stress'].replace(-1, np.nan)
        
        # Find gaps and interpolate if they're shorter than 15 minutes
        if df['stress'].isna().any():
            # Create a helper column to identify the groups of non-NaN values
            df['valid'] = ~df['stress'].isna()
            df['group'] = (df['valid'] != df['valid'].shift()).cumsum()
            
            # For each gap, check if it's short enough to interpolate
            for gap_group in df[~df['valid']]['group'].unique():
                gap_start = df[df['group'] == gap_group].index.min()
                gap_end = df[df['group'] == gap_group].index.max()
                
                # Get timestamps at the beginning and end of the gap
                if gap_start > 0 and gap_end < len(df) - 1:
                    gap_start_time = df.loc[gap_start, 'datetime']
                    gap_end_time = df.loc[gap_end, 'datetime']
                    
                    # Calculate the gap duration in minutes
                    gap_duration = (gap_end_time - gap_start_time).total_seconds() / 60
                    
                    if gap_duration <= 15:
                        # Short gap, interpolate
                        df.loc[gap_start:gap_end, 'stress'] = np.nan  # Make sure it's NaN for interpolation
                    # For longer gaps, leave as NaN
            
            # Interpolate for short gaps
            df['stress'] = df['stress'].interpolate(method='linear')
            
            # Drop helper columns
            df = df.drop(columns=['valid', 'group'])
        
        # Apply moving average smoothing (centered)
        window_size = 10
        df['stress_smooth'] = df['stress'].rolling(window=window_size, center=True, min_periods=1).mean()
        
        # Replace NaN in the smoothed column with original values for the edges
        mask = df['stress_smooth'].isna()
        df.loc[mask, 'stress_smooth'] = df.loc[mask, 'stress']
        
        # Replace the stress column with the smoothed version
        df['stress'] = df['stress_smooth'].round(1)
        
        # Drop the smoothing column
        df = df.drop(columns=['stress_smooth'])
        
        return df
            
    except Exception as e:
        logger.warning(f"Failed to get stress data for {date}: {str(e)}")
        return pd.DataFrame(columns=['datetime', 'stress'])

def get_sleep_data_for_day(garmin_client, date):
    """
    Get detailed sleep data for a specific day.
    
    Args:
        garmin_client: Initialized Garmin API client
        date: Date to get sleep data for (datetime.date object)
        
    Returns:
        Dictionary with sleep information including:
        - start_time: datetime object for sleep start
        - end_time: datetime object for sleep end
        - duration_seconds: total sleep duration in seconds
        - deep_sleep_seconds: deep sleep duration in seconds
        - light_sleep_seconds: light sleep duration in seconds
        - rem_sleep_seconds: REM sleep duration in seconds
        - awake_seconds: time awake during sleep in seconds
        - sleep_score: sleep score value (0-100)
        - sleep_levels: detailed sleep stages
    """
    try:
        # Format the date for API call
        formatted_date = date.strftime('%Y-%m-%d')
        
        # Get the sleep data for the day
        sleep_data = garmin_client.get_sleep_data(formatted_date)
        
        # Create a default response with empty values
        result = {
            'start_time': None,
            'end_time': None,
            'duration_seconds': 0,
            'deep_sleep_seconds': 0,
            'light_sleep_seconds': 0,
            'rem_sleep_seconds': 0,
            'awake_seconds': 0,
            'sleep_score': 0,
            'sleep_levels': []
        }
        
        if not sleep_data or 'dailySleepDTO' not in sleep_data:
            logger.warning(f"No sleep data found for {date}")
            return result
        
        daily_sleep = sleep_data['dailySleepDTO']
        
        # Extract sleep start and end times
        if 'sleepStartTimestampLocal' in daily_sleep and daily_sleep['sleepStartTimestampLocal']:
            start_time_ms = daily_sleep['sleepStartTimestampLocal']
            result['start_time'] = dt.fromtimestamp(start_time_ms / 1000.0)
        
        if 'sleepEndTimestampLocal' in daily_sleep and daily_sleep['sleepEndTimestampLocal']:
            end_time_ms = daily_sleep['sleepEndTimestampLocal']
            result['end_time'] = dt.fromtimestamp(end_time_ms / 1000.0)
        
        # Extract sleep durations
        result['duration_seconds'] = daily_sleep.get('sleepTimeSeconds', 0)
        result['deep_sleep_seconds'] = daily_sleep.get('deepSleepSeconds', 0)
        result['light_sleep_seconds'] = daily_sleep.get('lightSleepSeconds', 0)
        result['rem_sleep_seconds'] = daily_sleep.get('remSleepSeconds', 0)
        result['awake_seconds'] = daily_sleep.get('awakeSleepSeconds', 0)
        
        # Extract sleep score
        if 'sleepScores' in daily_sleep and 'overall' in daily_sleep['sleepScores']:
            result['sleep_score'] = daily_sleep['sleepScores']['overall'].get('value', 0)
        
        # Extract detailed sleep levels if available
        if 'sleepLevels' in sleep_data:
            result['sleep_levels'] = sleep_data['sleepLevels']
        
        return result
    
    except Exception as e:
        logger.warning(f"Failed to get sleep data for {date}: {str(e)}")
        return {
            'start_time': None,
            'end_time': None,
            'duration_seconds': 0,
            'deep_sleep_seconds': 0,
            'light_sleep_seconds': 0,
            'rem_sleep_seconds': 0,
            'awake_seconds': 0,
            'sleep_score': 0,
            'sleep_levels': []
        }

def run_garmin_etl():
    """Execute Garmin ETL process."""
    load_dotenv("Credentials.env")
    email = os.getenv("USERNAME_G")
    password = os.getenv("PASSWORD_G")
    client = init_garmin(email, password)
    df = get_garmin_data(client)

    df_activities = get_garmin_activities(client)
    if df_activities is not None:
        # Save activities data
        df_activities.to_csv(config.GARMIN_ACTIVITIES_FILE, index=False)
        logger.info(f'Garmin activities saved to {config.GARMIN_ACTIVITIES_FILE}')

        df.to_csv(config.GARMIN_DAILY_FILE, index=False)
        logger.info(f'Garmin data saved to {config.GARMIN_DAILY_FILE}')

if __name__ == "__main__":
    run_garmin_etl()
