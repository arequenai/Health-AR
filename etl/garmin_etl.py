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
from .garmin_tss_calculation import get_tss_data

warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

email = os.getenv("USERNAME_G")
password = os.getenv("PASSWORD_G")

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
    """Get Garmin data.
    
    Args:
        garmin_client: Garmin client
        start_date: Optional start date, defaults to config.DATA_START_DATE
    """
    if start_date is None:
        start_date = config.DATA_START_DATE
        
    api = garmin_client
    end_date = datetime.date.today()
    data_file = config.GARMIN_DAILY_FILE  # Use config instead of hardcoded path
    
    existing_data = None
    if os.path.exists(data_file):
        existing_data = pd.read_csv(data_file)
        if len(existing_data) > 0:
            last_date = pd.to_datetime(existing_data['date'].max()).date()
            start_date = last_date - datetime.timedelta(days=7)
            existing_data = existing_data[existing_data['date'] < start_date.strftime('%Y-%m-%d')]
    
    data_list = []
    current_date = start_date
    while current_date <= end_date:
        try:
            stats = api.get_stats(current_date)
            data_dict = {'date': current_date.strftime('%Y-%m-%d')}
            data_dict.update(stats)
            data_list.append(data_dict)
        except Exception as e:
            logger.warning(f"Failed to get data for {current_date}: {str(e)}")
        current_date += datetime.timedelta(days=1)
    
    df = pd.DataFrame(data_list)
    df.fillna(0, inplace=True)
    if existing_data is not None:
        df = pd.concat([existing_data, df], ignore_index=True)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df = df.sort_values('date').drop_duplicates(subset=['date'], keep='last')
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
            start_date = last_date - datetime.timedelta(days=2)
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
        start_date = datetime.date(2024, 3, 16)
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

        # Calculate TSS metrics
        tss_data = get_tss_data(df_activities)

    if df is not None:

        # join tss data with daily data
        df = pd.merge(df, tss_data, on='date', how='left')
        df.to_csv(config.GARMIN_DAILY_FILE, index=False)
        logger.info(f'Garmin data saved to {config.GARMIN_DAILY_FILE}')

if __name__ == "__main__":
    run_garmin_etl()
