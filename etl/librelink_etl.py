import logging
from LibreView import LibreView
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NOTE: The current LibreView Python client (imported above) only provides access to the 
# most recent glucose reading from the current connection. It does not support fetching
# historical data through the available API endpoints. The client would need to be enhanced
# to support the historical data endpoints documented in the unofficial LibreView API documentation.
#
# For more comprehensive data access, consider:
# 1. Using the lvconnect tool (https://github.com/skobkars/lvconnect) which can fetch historical data 
#    and upload to Nightscout
# 2. Directly implementing API calls using the unofficial documentation at 
#    https://libreview-unofficial.stoplight.io/
#
# This ETL script currently only captures the most recent reading and maintains a history
# by appending to an existing file, but it can't retrieve multiple past readings in a single run.

def fetch_historical_data(client, connection, days_back=90):
    """Attempt to fetch historical glucose data from LibreView.
    
    Note: With the current LibreView client, this will only return the most recent reading.
    Future enhancements might include direct API calls to get historical data.
    
    Args:
        client: Authenticated LibreView client
        connection: Connection object from get_connections()
        days_back: Number of days of historical data to fetch (ignored with current client)
        
    Returns:
        List of glucose readings
    """
    logger.info(f"Attempting to fetch glucose data...")
    glucose_data = []
    
    try:
        # Get connection ID and patient ID
        connection_id = connection.id
        patient_id = connection.patient_id
        
        logger.info(f"Connection ID: {connection_id}, Patient ID: {patient_id}")
        
        # Check if connection has the current glucose measurement
        if hasattr(connection, 'glucose_measurement') and connection.glucose_measurement:
            logger.info("Found current glucose measurement")
            
            # Process current measurement
            measurement = connection.glucose_measurement
            
            # Convert to dict if necessary
            if hasattr(measurement, 'to_dict'):
                current_measurement = measurement.to_dict()
            else:
                current_measurement = {attr: getattr(measurement, attr) 
                                     for attr in dir(measurement) 
                                     if not attr.startswith('_') and not callable(getattr(measurement, attr))}
            
            # Add this to our glucose data list
            glucose_data.append(current_measurement)
            
        # Warning about limitations
        if len(glucose_data) <= 1:
            logger.warning("NOTE: Current LibreView client only provides latest reading.")
            logger.warning("Historical data access requires enhancing the client with additional API endpoints.")
            
        logger.info(f"Returning {len(glucose_data)} readings")
        
        if hasattr(client, 'session'):
            session = client.session
            
            # Try to access the daily log endpoint for this patient
            daily_log_url = f"https://api-eu.libreview.io/llu/connections/{patient_id}/logbook"
            headers = {
                'accept-encoding': 'gzip',
                'connection': 'Keep-Alive',
                'content-type': 'application/json',
                'product': 'llu.android',
                'version': '4.2.1',
                # We would need to add authentication headers from the client
                'authorization': f'Bearer {client.token}' if hasattr(client, 'token') else None
            }
            
            try:
                response = session.get(daily_log_url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data and data['data']:
                        return data['data']  # This should contain multiple readings
            except Exception as e:
                logger.error(f"Error accessing daily log: {str(e)}")
        
        return glucose_data
        
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        return []

def process_glucose_data(glucose_data):
    """Process glucose data into daily metrics.
    
    Args:
        glucose_data: List of glucose readings
        
    Returns:
        DataFrame with daily metrics
    """
    if not glucose_data or len(glucose_data) == 0:
        logger.warning("No glucose data to process!")
        return None
        
    logger.info(f"Processing {len(glucose_data)} glucose readings")
    
    # Log the first item to help understand the data structure
    logger.info(f"Sample data item: {str(glucose_data[0])[:200]}...")
    
    # Convert to DataFrame
    readings_df = pd.DataFrame(glucose_data)
    
    # Print column names for debugging
    logger.info(f"DataFrame columns: {readings_df.columns.tolist()}")
    
    # Try to find timestamp fields
    timestamp_fields = []
    for col in readings_df.columns:
        if isinstance(col, str) and any(term in col.lower() for term in ['time', 'date', 'timestamp', 'recorded']):
            timestamp_fields.append(col)
            
    logger.info(f"Found potential timestamp fields: {timestamp_fields}")
    
    # Try to convert each to datetime and use the first valid one
    timestamp_found = False
    for field in timestamp_fields:
        try:
            readings_df['timestamp'] = pd.to_datetime(readings_df[field])
            logger.info(f"Successfully converted {field} to timestamp")
            timestamp_found = True
            break
        except Exception as e:
            logger.warning(f"Could not convert {field} to timestamp: {str(e)}")
    
    # If no timestamp field found, try using a timestamp from a nested structure
    if not timestamp_found:
        logger.warning("No direct timestamp field found, checking for nested timestamp")
        if 'rt' in readings_df.columns:  # 'rt' might be a common field in LibreView for record timestamp
            try:
                readings_df['timestamp'] = pd.to_datetime(readings_df['rt'], unit='ms')
                logger.info("Successfully converted 'rt' field to timestamp")
                timestamp_found = True
            except Exception as e:
                logger.warning(f"Could not convert 'rt' to timestamp: {str(e)}")
        
        # Try other special cases for LibreView API
        if not timestamp_found and 'timestamp' in readings_df.columns:
            try:
                # Check if timestamp is in milliseconds (epoch)
                readings_df['timestamp'] = pd.to_datetime(readings_df['timestamp'], unit='ms')
                logger.info("Successfully converted timestamp field from milliseconds")
                timestamp_found = True
            except Exception as e:
                logger.warning(f"Could not convert timestamp from milliseconds: {str(e)}")
                
    if not timestamp_found:
        logger.error("Could not create a timestamp column!")
        logger.error(f"Available columns: {readings_df.columns.tolist()}")
        # Add a sample of the first row to help debugging
        logger.error(f"Sample row data: {readings_df.iloc[0].to_dict() if not readings_df.empty else 'No data'}")
        return None
        
    # Add date column
    readings_df['timestamp'] = pd.to_datetime(readings_df['timestamp'])
    readings_df['date'] = readings_df['timestamp'].dt.date
    
    # Find glucose value field
    glucose_col = None
    # Extended list of potential glucose column names
    for col in ['value', 'valueInMgPerDl', 'glucose', 'glucoseValue', 'glucose_value', 'bg', 'sgv', 'mg_dl']:
        if col in readings_df.columns:
            glucose_col = col
            break
            
    # If standard columns not found, look for any column with 'glucose' in the name
    if not glucose_col:
        for col in readings_df.columns:
            if 'glucose' in str(col).lower():
                glucose_col = col
                break
    
    if not glucose_col:
        logger.error("No glucose value column found! Available columns:")
        logger.error(readings_df.columns.tolist())
        return None
        
    logger.info(f"Using {glucose_col} as glucose value column")
    
    # Ensure glucose values are numeric
    try:
        readings_df[glucose_col] = pd.to_numeric(readings_df[glucose_col], errors='coerce')
    except Exception as e:
        logger.warning(f"Error converting glucose values to numeric: {str(e)}")
    
    # Filter out invalid glucose values
    valid_mask = readings_df[glucose_col].notna() & (readings_df[glucose_col] > 0) & (readings_df[glucose_col] < 500)
    readings_df = readings_df[valid_mask]
    
    if readings_df.empty:
        logger.error("No valid glucose readings after filtering!")
        return None
    
    # Define ranges - could be customized based on connection.target_low and connection.target_high
    in_range_min = 70  # mg/dL
    in_range_max = 180  # mg/dL
    
    # Process each day's data
    daily_metrics = []
    
    for day, group in readings_df.groupby('date'):
        # Calculate metrics
        readings_count = len(group)
        mean_glucose = group[glucose_col].mean()
        min_glucose = group[glucose_col].min()
        max_glucose = group[glucose_col].max()
        std_glucose = group[glucose_col].std() if readings_count > 1 else 0
        
        # Calculate time in ranges
        in_range = group[(group[glucose_col] >= in_range_min) & (group[glucose_col] <= in_range_max)]
        above_range = group[group[glucose_col] > in_range_max]
        below_range = group[group[glucose_col] < in_range_min]
        
        time_in_range_pct = len(in_range) / readings_count * 100 if readings_count > 0 else 0
        time_above_range_pct = len(above_range) / readings_count * 100 if readings_count > 0 else 0
        time_below_range_pct = len(below_range) / readings_count * 100 if readings_count > 0 else 0
        
        # Get morning reading for fasting glucose approximation
        morning_readings = group[(group['timestamp'].dt.hour >= 6) & (group['timestamp'].dt.hour <= 9)]
        if morning_readings.empty:
            morning_readings = group  # Fallback to using all readings if no morning readings
        fasting_glucose = morning_readings[glucose_col].mean() if not morning_readings.empty else np.nan
        
        daily_metrics.append({
            'date': day,
            'mean_glucose': mean_glucose,
            'min_glucose': min_glucose,
            'max_glucose': max_glucose,
            'std_glucose': std_glucose,
            'readings_count': readings_count,
            'time_in_range_pct': time_in_range_pct,
            'time_above_range_pct': time_above_range_pct,
            'time_below_range_pct': time_below_range_pct,
            'fasting_glucose': fasting_glucose
        })
        
    # Create DataFrame from metrics
    metrics_df = pd.DataFrame(daily_metrics)
    
    # Sort by date
    metrics_df = metrics_df.sort_values('date')
    
    logger.info(f"Processed data for {len(metrics_df)} days")
    
    return metrics_df

def run_librelink_etl():
    """Run the ETL process for LibreView glucose data.
    
    Fetches current glucose data from LibreView, transforms it into daily metrics,
    and saves to a CSV file. If the output file exists, it will append new data.
    
    Note: Due to current client limitations, only the most recent reading is retrieved.
    
    Returns:
        DataFrame with daily metrics if successful, None otherwise.
    """
    try:
        # Define output file path
        csv_path = 'data/glucose_daily.csv'
        
        # Check if output file exists and get last date if it does
        last_date = None
        existing_data = None
        if os.path.exists(csv_path):
            try:
                existing_data = pd.read_csv(csv_path)
                if not existing_data.empty and 'date' in existing_data.columns:
                    # Convert date column to datetime
                    existing_data['date'] = pd.to_datetime(existing_data['date']).dt.date
                    last_date = existing_data['date'].max()
                    logger.info(f"Found existing data up to {last_date}")
            except Exception as e:
                logger.error(f"Error reading existing file: {str(e)}")
                logger.info("Will fetch all data")
        
        # Load credentials
        load_dotenv('Credentials.env')
        username = os.getenv("LIBRELINK_USERNAME")
        password = os.getenv("LIBRELINK_PASSWORD")
        
        if not username or not password:
            logger.error("Missing LibreView credentials in Credentials.env")
            return None
        
        logger.info(f"Using LibreView account: {username}")
        
        # Initialize LibreView client
        logger.info("Initializing LibreView client...")
        client = LibreView(username, password)
        
        # Fetch connections (patients)
        logger.info("Fetching connections...")
        connections = client.get_connections()
        
        if not connections:
            logger.warning("No connections found!")
            return None
            
        logger.info(f"Found {len(connections)} connections")
        
        # Choose the first connection (typically the user's own data)
        connection = connections[0]
        
        # Fetch data - with the current library, this only gives us the most recent reading
        logger.info("Fetching glucose data...")
        glucose_data = fetch_historical_data(client, connection)
            
        # Process the glucose data
        new_metrics_df = process_glucose_data(glucose_data)
        
        if new_metrics_df is None or new_metrics_df.empty:
            logger.warning("No new data to add")
            return existing_data
        
        # Combine with existing data if we have it
        final_metrics_df = new_metrics_df
        if existing_data is not None and not existing_data.empty:
            # Convert existing_data date to datetime for proper merging
            if not pd.api.types.is_datetime64_dtype(existing_data['date']):
                existing_data['date'] = pd.to_datetime(existing_data['date']).dt.date
                
            # Combine and remove duplicates, keeping new data where dates overlap
            combined_df = pd.concat([existing_data, new_metrics_df])
            final_metrics_df = combined_df.drop_duplicates(subset=['date'], keep='last')
            
            # Sort by date
            final_metrics_df = final_metrics_df.sort_values('date')
            
            logger.info(f"Combined {len(existing_data)} existing and {len(new_metrics_df)} new records")
            logger.info(f"Final dataset has {len(final_metrics_df)} records")
            
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save to CSV
        final_metrics_df.to_csv(csv_path, index=False)
        
        logger.info(f"Successfully saved data to {csv_path}")
        logger.info(f"Dataset covers from {final_metrics_df['date'].min()} to {final_metrics_df['date'].max()}")
        logger.info(f"Total of {len(final_metrics_df)} days of data")
        
        return final_metrics_df
    
    except Exception as e:
        logger.error(f"Error in LibreLink ETL: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    run_librelink_etl() 