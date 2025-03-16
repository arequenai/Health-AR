from dotenv import load_dotenv
load_dotenv("Credentials.env")

import requests
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_nightscout_data(base_url, token, start_date=None):
    """
    Fetch glucose data from Nightscout using the token in the URL
    If start_date is provided, fetch data from that date onwards
    """
    # Format the API URL with token in query string
    if start_date:
        # Convert start_date to timestamp in milliseconds
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        
        # Make sure we get the start of the day
        start_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
        entries_url = f"{base_url}/api/v1/entries.json?token={token}&count=100000&find[date][$gte]={start_timestamp}"
        logger.info(f"Fetching data from: {base_url}/api/v1/entries.json")
        logger.info(f"Time range: Since {start_date}")
    else:
        # Fetch all available data
        entries_url = f"{base_url}/api/v1/entries.json?token={token}&count=100000"
        logger.info(f"Fetching all available data from: {base_url}/api/v1/entries.json")
    
    # Make the request
    try:
        response = requests.get(entries_url, timeout=30)
        
        if response.status_code == 200:
            entries = response.json()
            logger.info(f"Successfully retrieved {len(entries)} glucose readings")
            return entries
        else:
            logger.error(f"Error fetching data: Status code {response.status_code}")
            logger.error(f"Response: {response.text[:200]}")  # Print first 200 chars of response
            return []
    except Exception as e:
        logger.error(f"Exception during request: {str(e)}")
        return []

def get_last_available_date():
    """
    Get the last available date from the glucose_daily.csv file
    Returns None if the file doesn't exist
    """
    file_path = os.path.join('data', 'glucose_daily.csv')
    if not os.path.exists(file_path):
        logger.info("No existing glucose_daily.csv file found. Will fetch all data.")
        return None
    
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            logger.info("Existing glucose_daily.csv file is empty. Will fetch all data.")
            return None
        
        # Convert date column to datetime if it's not already
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date
            last_date = df['date'].max()
            logger.info(f"Last available date in glucose_daily.csv: {last_date}")
            return last_date
        else:
            logger.warning("No 'date' column found in glucose_daily.csv. Will fetch all data.")
            return None
    except Exception as e:
        logger.error(f"Error reading glucose_daily.csv: {str(e)}. Will fetch all data.")
        return None

def process_entries(entries):
    """
    Process entries into a pandas DataFrame
    """
    if not entries:
        logger.warning("No entries to process")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(entries)
    
    # Convert timestamp to datetime
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    
    # Add date column (without time)
    df['day'] = df['date'].dt.date
    
    # Add hour column for time-of-day analysis
    df['hour'] = df['date'].dt.hour
    
    logger.info(f"Processed data spanning from {df['date'].min()} to {df['date'].max()}")
    
    return df

def create_daily_summary(df):
    """
    Create daily summary with min, max, mean values, and fasting glucose at 6am
    """
    if df is None or df.empty:
        logger.warning("No data available for daily summary")
        return None
    
    # Ensure we have the 'sgv' column (sensor glucose value)
    if 'sgv' not in df.columns:
        logger.warning("Warning: 'sgv' column not found in data")
        return None
    
    # Group by day and calculate statistics
    daily = df.groupby('day').agg({
        'sgv': ['count', 'mean', 'min', 'max']
    }).reset_index()
    
    # Flatten MultiIndex columns
    daily.columns = ['date', 'readings_count', 'mean_glucose', 'min_glucose', 'max_glucose']
    
    # Get morning reading for fasting glucose approximation (6 AM specifically)
    fasting_glucose_list = []
    
    for day in daily['date']:
        day_data = df[df['day'] == day]
        
        # Look for readings at exactly 6 AM
        morning_6am = day_data[day_data['date'].dt.hour == 6]
        
        if not morning_6am.empty:
            # Use the first reading at 6 AM
            fasting_glucose = morning_6am.iloc[0]['sgv']
        else:
            # Fall back to the first reading of the day if no 6 AM reading
            morning_readings = day_data.sort_values('date').iloc[0:1]
            fasting_glucose = morning_readings['sgv'].iloc[0] if not morning_readings.empty else None
            
        fasting_glucose_list.append(fasting_glucose)
        
    daily['fasting_glucose'] = fasting_glucose_list
    
    return daily

def merge_with_existing_data(new_daily_df):
    """
    Merge new daily summary with existing data in glucose_daily.csv
    """
    if new_daily_df is None or new_daily_df.empty:
        logger.warning("No new data to merge")
        return new_daily_df
    
    file_path = os.path.join('data', 'glucose_daily.csv')
    if not os.path.exists(file_path):
        logger.info("No existing glucose_daily.csv file found. Using only new data.")
        return new_daily_df
    
    try:
        existing_df = pd.read_csv(file_path)
        if existing_df.empty:
            logger.info("Existing glucose_daily.csv file is empty. Using only new data.")
            return new_daily_df
        
        # Convert date columns to datetime
        existing_df['date'] = pd.to_datetime(existing_df['date']).dt.date
        new_daily_df['date'] = pd.to_datetime(new_daily_df['date']).dt.date
        
        # Remove days from existing data that are in new data (to handle updates for the last day)
        existing_df = existing_df[~existing_df['date'].isin(new_daily_df['date'])]
        
        # Concatenate and sort by date
        merged_df = pd.concat([existing_df, new_daily_df], ignore_index=True)
        merged_df = merged_df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Merged data now spans from {merged_df['date'].min()} to {merged_df['date'].max()}")
        return merged_df
    except Exception as e:
        logger.error(f"Error merging with existing data: {str(e)}. Using only new data.")
        return new_daily_df

def save_to_csv(df, filename='glucose_data.csv'):
    """
    Save DataFrame to CSV in the data directory
    """
    os.makedirs('data', exist_ok=True)
    file_path = os.path.join('data', filename)
    df.to_csv(file_path, index=False)
    logger.info(f"Data saved to {file_path}")
    return file_path

def generate_visualizations(daily_df):
    """
    Generate basic visualizations from the data
    """
    try:
        os.makedirs('data/visualizations', exist_ok=True)
        
        if daily_df is not None and not daily_df.empty:
            # Daily glucose plot
            plt.figure(figsize=(12, 6))
            plt.plot(daily_df['date'], daily_df['mean_glucose'], 'b-', label='Mean')
            plt.plot(daily_df['date'], daily_df['min_glucose'], 'g--', label='Min')
            plt.plot(daily_df['date'], daily_df['max_glucose'], 'r--', label='Max')
            plt.plot(daily_df['date'], daily_df['fasting_glucose'], 'y-', marker='o', label='Fasting (6am)')
            plt.fill_between(daily_df['date'], daily_df['min_glucose'], daily_df['max_glucose'], alpha=0.2)
            plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
            plt.axhline(y=180, color='r', linestyle='-', alpha=0.3)
            plt.title('Daily Glucose Trends')
            plt.xlabel('Date')
            plt.ylabel('Glucose (mg/dL)')
            plt.legend()
            plt.tight_layout()
            plt.savefig('data/visualizations/daily_glucose_trends.png')
            
            logger.info("Visualizations saved to data/visualizations/ directory")
            return 'data/visualizations/daily_glucose_trends.png'
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        return None

def run_glucose_etl():
    """
    Main ETL function to run the glucose data extraction, transformation, and loading process
    Efficiently fetches only new data since the last available date
    """
    # Get credentials from environment variables
    nightscout_url = os.getenv('NIGHTSCOUT_URL')
    token = os.getenv('NIGHTSCOUT_TOKEN')
    
    if not nightscout_url or not token:
        logger.error("Nightscout URL or token not found in environment variables")
        return None
    
    # Get the last available date from the existing data
    last_date = get_last_available_date()
    
    # Fetch data since the last available date
    entries = fetch_nightscout_data(nightscout_url, token, start_date=last_date)
    
    if not entries:
        logger.info("No new data found since the last update.")
        return None
    
    # Process entries
    df = process_entries(entries)
    if df is None:
        logger.error("Error processing entries")
        return None
    
    # Create daily summary for new data
    new_daily_df = create_daily_summary(df)
    if new_daily_df is None:
        logger.error("Could not create daily summary. Check if 'sgv' column exists in data.")
        return None
    
    # Merge with existing data
    merged_daily_df = merge_with_existing_data(new_daily_df)
    
    # Save merged data
    daily_file = save_to_csv(merged_daily_df, 'glucose_daily.csv')
    
    # Also save raw data for the new entries
    raw_file = save_to_csv(df, 'glucose_raw.csv')
    
    # Calculate overall statistics
    overall_mean = merged_daily_df['mean_glucose'].mean()
    overall_min = merged_daily_df['min_glucose'].min()
    overall_max = merged_daily_df['max_glucose'].max()
    
    logger.info("\nOverall Statistics:")
    logger.info(f"Mean Glucose: {overall_mean:.1f} mg/dL")
    logger.info(f"Min Glucose: {overall_min:.1f} mg/dL")
    logger.info(f"Max Glucose: {overall_max:.1f} mg/dL")
    
    # Generate visualizations
    try:
        vis_file = generate_visualizations(merged_daily_df)
    except ImportError:
        logger.warning("Matplotlib not installed. Skipping visualizations.")
    
    return daily_file

if __name__ == "__main__":
    # Run the ETL process efficiently, fetching only new data
    output_file = run_glucose_etl()
    
    if output_file:
        logger.info(f"ETL process completed successfully. Output file: {output_file}")
    else:
        logger.info("No new data added or ETL process failed.") 