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

if __name__ == "__main__":
    garmin_client = init_garmin(email, password)
    df = get_garmin_data(garmin_client)
    if df is not None:
        df.to_csv(config.GARMIN_DAILY_FILE, index=False)  # Use config
        logger.info(f'Garmin data saved to {config.GARMIN_DAILY_FILE}')
