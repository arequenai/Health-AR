from dotenv import load_dotenv
load_dotenv("Credentials.env")

import datetime
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

            # Only get sleep score if we need it (not today or today's data doesn't exist yet)
            if current_date != end_date or not today_data_exists:
                data_dict['sleep_score'] = get_sleep_score(api, current_date)
            else:
                # Skip sleep API call for today if we already have it
                data_dict['sleep_score'] = 0  # Will be filled from existing data later

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
            'type': activity.get('activityType', {}).get('typeKey', 'unknown'),
            'duration': activity.get('duration', 0),
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
    df = df.sort_values('date').drop_duplicates(subset=['date', 'type', 'duration'], keep='last')
    # Convert back to string format
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    
    return df

def get_sleep_score(api, date):
    """Get only the sleep score for a given date.
    
    Args:
        api: Garmin API client
        date: Date to get sleep score for (datetime.date object)
        
    Returns:
        int: Sleep score (0 if not available)
    """
    try:
        formatted_date = date.strftime('%Y-%m-%d')
        sleep_data = api.get_sleep_data(formatted_date)
        
        # Direct path navigation with defaults at each step
        if not sleep_data:
            return 0
            
        sleep_dto = sleep_data.get('dailySleepDTO', {})
        scores = sleep_dto.get('sleepScores', {})
        overall = scores.get('overall', {})
        return overall.get('value', 0)
    except Exception as e:
        logger.warning(f"Failed to get sleep score for {date}: {str(e)}")
        return 0

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
