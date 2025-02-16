import os
import pandas as pd
from dotenv import load_dotenv
from whoop import WhoopClient
import csv
from datetime import datetime, timedelta
import logging
from etl import config

logger = logging.getLogger(__name__)

def init_whoop(un, pw):
    """Initialize Whoop client with credentials."""
    client = WhoopClient(un, pw)
    profile = client.get_profile()
    return client

def get_journal_data(input_file, output_file):
    
    df = pd.read_csv(input_file)

    def set_date(row):
        date = datetime.strptime(row['Cycle start time'], '%Y-%m-%d %H:%M:%S')
        if date.hour < 12:
            date -= timedelta(days=1)
        return date.date()

    df['date'] = df.apply(set_date, axis=1)
    df = df[['date', 'Question text', 'Answered yes']]
    pd.set_option('future.no_silent_downcasting', True)
    df['Answered yes'] = df['Answered yes'].replace({True: "Yes", False: "No"}).infer_objects(copy=False)
    df_u = df.pivot_table(index='date', columns='Question text', values='Answered yes', aggfunc='sum')
    df_u.reset_index(inplace=True)
    df_u = df_u[['date', 'Avoid consuming processed foods?', 'Eat any food close to bedtime?',
                 'Feeling sick or ill?', 'Have an injury or wound', 'Have any alcoholic drinks?',
                 'Read (non-screened device) while in bed?', 'Spend time stretching?',
                 'Viewed a screen device in bed?']]
    df_u.rename(columns={'Avoid consuming processed foods?': 'avoid_processed_foods',
                         'Eat any food close to bedtime?': 'bed_full',
                         'Feeling sick or ill?': 'sick_or_ill',
                         'Have an injury or wound': 'injury',
                         'Have any alcoholic drinks?': 'alcohol',
                         'Read (non-screened device) while in bed?': 'read_bed',
                         'Spend time stretching?': 'stretch',
                         'Viewed a screen device in bed?': 'screen_bed'}, inplace=True)
    df_u.columns.name = None
    df_u.to_csv(output_file, index=False)
    print(f"{output_file}: Journal data obtained and rewritten'")

def get_sleep_recovery_data(client, start_date=None):
    """Get sleep and recovery data from Whoop.
    
    Args:
        client: Whoop client
        start_date: Optional start date, defaults to config.DATA_START_DATE
    """
    if start_date is None:
        start_date = config.DATA_START_DATE
    
    logger.info(f"Getting Whoop sleep and recovery data from {start_date}")
    
    # Check existing data
    if os.path.exists(config.WHOOP_SLEEP_RECOVERY_FILE):
        existing_data = pd.read_csv(config.WHOOP_SLEEP_RECOVERY_FILE)
        if len(existing_data) > 0:
            # If we have data, start from the last date minus 7 days
            last_date = pd.to_datetime(existing_data['date'].max()).date()
            start_date = last_date - timedelta(days=7)
            logger.info(f"Found existing data, pulling from {start_date} onwards")
            
            # Remove the last week of data from existing_data to avoid duplicates
            cutoff_date = start_date.strftime('%Y-%m-%d')
            existing_data = existing_data[existing_data['date'] < cutoff_date]
        else:
            logger.info(f"Existing data is empty, pulling all data since {config.DATA_START_DATE}")
            start_date = config.DATA_START_DATE
            existing_data = None
    else:
        logger.info(f"No existing data found, pulling all data since {config.DATA_START_DATE}")
        start_date = config.DATA_START_DATE
        existing_data = None
    
    # Get sleep data
    sleep = client.get_sleep_collection(start_date.strftime('%Y-%m-%d'))
    df_s = pd.json_normalize(sleep)
    
    # Get recovery data
    recovery = client.get_recovery_collection(start_date.strftime('%Y-%m-%d'))
    df_r = pd.json_normalize(recovery)

    def apply_timezone_offset(row):
        start_time = pd.to_datetime(row['start'])
        offset = row['timezone_offset']

        # If it's just "Z", treat as zero offset
        if offset == "Z":
            hours, minutes = (0, 0)
        else:
            # Otherwise split normally
            hours, minutes = map(int, offset.split(':'))

        timezone_offset = pd.Timedelta(hours=hours, minutes=minutes)
        return start_time + timezone_offset

    # Process sleep data
    df_s['start'] = df_s.apply(apply_timezone_offset, axis=1)

    for col in df_s.columns:
        if col.endswith('_milli'):
            df_s[col[:-6]] = df_s[col] / 1000 / 60 / 60
            df_s.drop(columns=[col], inplace=True)

    df_s['start'] = pd.to_datetime(df_s['start']).dt.floor('s')
    df_s['day'], df_s['time'] = df_s['start'].dt.date, df_s['start'].dt.time
    df_s = df_s[['day', 'time'] + [col for col in df_s.columns if col not in ['day', 'time']]]
    mask = (df_s['time'] > pd.to_datetime('20:00').time()).astype(int)
    df_s['day'] = pd.to_datetime(df_s['day']) + pd.to_timedelta(mask, unit='D')
    df_s = df_s[df_s['score_state'] == 'SCORED'].drop(columns=['score_state'])
    df_s.columns = df_s.columns.str.replace('score.', '')
    df_s.columns = df_s.columns.str.replace('stage_summary.total_', '')
    columns_map = {'day': 'date', 'time': 'sleep_time', 'id': 'sleep_id', 'nap': 'nap', 'score': 'sleep_score',
                   'sleep_performance_percentage': 'sleep_score_performance',
                   'sleep_consistency_percentage': 'sleep_score_consistency',
                   'sleep_efficiency_percentage': 'sleep_score_efficiency',
                   'no_data_time': 'sleep_unspecified', 'awake_time': 'sleep_awake',
                   'light_sleep_time': 'sleep_light', 'slow_wave_sleep_time': 'sleep_deep',
                   'rem_sleep_time': 'sleep_rem', 'in_bed_time': 'sleep_duration'}
    df_s.rename(columns=columns_map, inplace=True)
    df_s = df_s[['date', 'sleep_time', 'sleep_id', 'nap', 'sleep_score_performance',
                 'sleep_score_consistency', 'sleep_score_efficiency', 'sleep_duration',
                 'sleep_rem', 'sleep_deep', 'sleep_light', 'sleep_awake']]
    df_s = df_s[df_s['nap'] == False]
    df_s.drop(columns=['nap'], inplace=True)

    # Process recovery data
    df_r.columns = df_r.columns.str.replace('score.', '')
    df_r.drop(columns=['cycle_id', 'user_id', 'created_at', 'updated_at', 'score_state', 'user_calibrating'], inplace=True)
    columns_map = {'resting_heart_rate': 'resting_hr', 'hrv_rmssd_milli':'hrv', 'spo2_percentage': 'spo2','skin_temp_celsius': 'skin_temp'}
    df_r.rename(columns=columns_map, inplace=True)
    
    # Merge sleep and recovery data
    df = pd.merge(df_s, df_r, on='sleep_id', how='left')
    
    # If we have existing data, append the new data
    if existing_data is not None:
        df = pd.concat([existing_data, df], ignore_index=True)
    
    # Convert date to datetime for proper sorting and deduplication
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates(subset=['date'], keep='last')
    # Convert back to string format
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    
    logger.info("Completed Whoop data retrieval and processing")
    return df

def main():
   
    # Load environment variables 
    load_dotenv("Credentials.env")
    un = os.getenv("USERNAME_W")
    pw = os.getenv("PASSWORD_W")
   
    print("Logging in to Whoop")
    client = init_whoop(un, pw)

    whoop_file = config.WHOOP_SLEEP_RECOVERY_FILE
    start_date = config.DATA_START_DATE
    df = get_sleep_recovery_data(client, start_date)
    df.to_csv(whoop_file, index=False)
    print('Sleep and recovery data cleaned and saved')
   
    journal_file_raw = config.WHOOP_JOURNAL_RAW_FILE
    journal_file = config.WHOOP_JOURNAL_CLEAN_FILE
 #   get_journal_data(journal_file_raw, journal_file)

if __name__ == "__main__":
    main()
