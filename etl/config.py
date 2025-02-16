# Common configuration settings for ETL processes

from datetime import datetime, date

# Start date for all data collection
DATA_START_DATE = date(2024, 3, 16)

# File paths
DATA_DIR = 'data'
DATA_DIR_RAW = 'data/raw'

# File names
GARMIN_DAILY_FILE = f'{DATA_DIR}/garmin_daily.csv'

# Whoop specific config
WHOOP_SLEEP_RECOVERY_FILE = f'{DATA_DIR}/whoop.csv'
WHOOP_JOURNAL_RAW_FILE = f'{DATA_DIR_RAW}/journal_entries.csv'
WHOOP_JOURNAL_CLEAN_FILE = f'{DATA_DIR}/journal.csv'

# Add if not already there
GARMIN_DATA_START_DATE = date(2024, 3, 16)  # If you want a different start date for Garmin
