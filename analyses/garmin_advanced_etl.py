import os
import pandas as pd
import numpy as np
import datetime
import json
import sys
from dotenv import load_dotenv
sys.path.append('..')  # Add parent directory to path

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
GARMIN_DAILY_FILE = os.path.join(DATA_DIR, 'garmin_daily.csv')
GARMIN_ADVANCED_FILE = os.path.join(DATA_DIR, 'garmin_advanced_metrics.csv')
GARMIN_ENHANCED_FILE = os.path.join(DATA_DIR, 'garmin_enhanced.csv')

# Make sure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

try:
    from garminconnect import Garmin, GarminConnectAuthenticationError
    from garth.exc import GarthHTTPError
    HAS_GARMINCONNECT = True
except ImportError:
    HAS_GARMINCONNECT = False
    print("Warning: python-garminconnect package is not installed. Run 'pip install garminconnect' to install it.")

def initialize_garmin_client():
    """
    Initialize Garmin client using the same authentication approach as in garmin_etl.py.
    
    Returns:
        Garmin: Authenticated Garmin client or None if authentication fails
    """
    if not HAS_GARMINCONNECT:
        print("Error: garminconnect package is required for this functionality.")
        return None
    
    # Load credentials from environment file
    load_dotenv(os.path.join(PROJECT_ROOT, "Credentials.env"))
    email = os.getenv("USERNAME_G")
    password = os.getenv("PASSWORD_G")
    
    if not email or not password:
        print("Error: Garmin Connect credentials not found in environment variables.")
        print("Make sure USERNAME_G and PASSWORD_G are set in Credentials.env file.")
        return None
    
    tokenstore = os.getenv("GARMINTOKENS") or "~/.garminconnect"
    tokenstore = os.path.expanduser(tokenstore)
    
    try:
        # Try to authenticate using stored tokens first
        print(f"Attempting to authenticate using tokens from {tokenstore}...")
        garmin = Garmin()
        garmin.login(tokenstore)
        print("Authentication with Garmin Connect successful using tokens!")
    except (FileNotFoundError, GarthHTTPError, GarminConnectAuthenticationError) as e:
        print(f"Token authentication failed: {e}")
        print("Trying to authenticate with email and password...")
        try:
            # Fall back to email/password authentication
            garmin = Garmin(email=email, password=password)
            garmin.login()
            # Save the tokens for future use
            garmin.garth.dump(tokenstore)
            print("Authentication successful with email/password! Tokens saved for future use.")
        except Exception as e:
            print(f"Authentication failed with email/password: {e}")
            return None
    except Exception as e:
        print(f"Unexpected error during authentication: {e}")
        return None
        
    return garmin

def get_date_range(days=30, end_date=None):
    """
    Get a range of dates ending with specified date or today.
    
    Args:
        days (int): Number of days to look back
        end_date (datetime.date, optional): End date. Defaults to today.
    
    Returns:
        tuple: (start_date, end_date) as datetime.date objects
    """
    if end_date is None:
        end_date = datetime.date.today()
    elif isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
    
    start_date = end_date - datetime.timedelta(days=days)
    return start_date, end_date

def extract_sleep_stages(sleep_data):
    """
    Extract detailed sleep stages from Garmin sleep data.
    
    Args:
        sleep_data (dict): Garmin sleep data from API
    
    Returns:
        dict: Dictionary with sleep stage durations in seconds
    """
    result = {
        "deep_sleep_seconds": 0,
        "light_sleep_seconds": 0,
        "rem_sleep_seconds": 0,
        "awake_sleep_seconds": 0
    }
    
    # Return zeros if no data or no sleep levels
    if not sleep_data or "sleepLevels" not in sleep_data:
        return result
    
    # Extract sleep stages from sleep levels
    for level in sleep_data.get("sleepLevels", []):
        if level["stage"] == "deep":
            result["deep_sleep_seconds"] += level["seconds"]
        elif level["stage"] == "light":
            result["light_sleep_seconds"] += level["seconds"]
        elif level["stage"] == "rem":
            result["rem_sleep_seconds"] += level["seconds"]
        elif level["stage"] == "awake":
            result["awake_sleep_seconds"] += level["seconds"]
    
    return result

def find_minimum_sustained_stress(stress_data, duration_mins=10):
    """
    Find minimum stress level sustained for a given duration.
    This is useful for finding the lowest stress period during sleep.
    
    Args:
        stress_data (dict): Garmin stress data from API
        duration_mins (int): Minimum duration in minutes to consider sustained
    
    Returns:
        float: Minimum sustained stress level or None if not available
    """
    if not stress_data or "stressValuesArray" not in stress_data:
        return None
    
    values = stress_data.get("stressValuesArray", [])
    if not values:
        return None
    
    # Each stress measurement is typically at regular intervals
    # We need to find consecutive low readings
    min_sustained = 100  # Start with max possible value
    current_run = []
    
    for entry in values:
        value = entry.get("value")
        if value is not None:  # Skip None entries
            if len(current_run) == 0 or current_run[-1][0] + 60 >= entry.get("dateTime", 0):
                # Add to current run if it's a consecutive reading (within ~1 minute)
                current_run.append((entry.get("dateTime", 0), value))
            else:
                # Start new run
                current_run = [(entry.get("dateTime", 0), value)]
            
            # Check if current run is long enough
            if len(current_run) >= 4:  # Typically 4 readings would span about 15-20 minutes
                avg_stress = sum(v for _, v in current_run) / len(current_run)
                min_sustained = min(min_sustained, avg_stress)
    
    return min_sustained if min_sustained < 100 else None

def extract_respiration_data(respiration_data):
    """
    Extract respiratory rate data from Garmin API response.
    
    Args:
        respiration_data (dict): Garmin respiration data
    
    Returns:
        dict: Dictionary with respiration metrics
    """
    result = {
        "avg_respiration_rate": None,
        "lowest_respiration_rate": None,
        "highest_respiration_rate": None
    }
    
    if not respiration_data:
        return result
    
    # Extract average respiration rate
    if "avgValueInBreathsPerMinute" in respiration_data:
        result["avg_respiration_rate"] = respiration_data["avgValueInBreathsPerMinute"]
    
    # Extract lowest respiration rate
    if "lowestValueInBreathsPerMinute" in respiration_data:
        result["lowest_respiration_rate"] = respiration_data["lowestValueInBreathsPerMinute"]
    
    # Extract highest respiration rate
    if "highestValueInBreathsPerMinute" in respiration_data:
        result["highest_respiration_rate"] = respiration_data["highestValueInBreathsPerMinute"]
    
    return result

def extract_hrv_data(hrv_data):
    """
    Extract Heart Rate Variability (HRV) data from Garmin API response.
    
    Args:
        hrv_data (dict): Garmin HRV data
    
    Returns:
        dict: Dictionary with HRV metrics
    """
    result = {
        "avg_hrv": None,
        "weekly_avg_hrv": None,
        "last_night_avg_hrv": None,
        "last_night_hrv_status": None
    }
    
    if not hrv_data or "hrvSummary" not in hrv_data:
        return result
    
    hrv_summary = hrv_data.get("hrvSummary", {})
    
    # Extract weekly average HRV
    if "weekly" in hrv_summary and "avgValueInMilliseconds" in hrv_summary["weekly"]:
        result["weekly_avg_hrv"] = hrv_summary["weekly"]["avgValueInMilliseconds"]
    
    # Extract last night's average HRV
    if "lastNight" in hrv_summary and "avgValueInMilliseconds" in hrv_summary["lastNight"]:
        result["last_night_avg_hrv"] = hrv_summary["lastNight"]["avgValueInMilliseconds"]
        result["avg_hrv"] = hrv_summary["lastNight"]["avgValueInMilliseconds"]  # Use last night as primary HRV
    
    # Extract status
    if "lastNight" in hrv_summary and "qualifierKey" in hrv_summary["lastNight"]:
        result["last_night_hrv_status"] = hrv_summary["lastNight"]["qualifierKey"]
    
    return result

def extract_garmin_data(days=90, end_date=None, save=True, verbose=True):
    """
    Extract and process advanced Garmin data.
    
    Args:
        days (int): Number of days to look back
        end_date (str or datetime.date, optional): End date. Defaults to today.
        save (bool): Whether to save results to CSV
        verbose (bool): Whether to print progress messages
    
    Returns:
        pandas.DataFrame: DataFrame with advanced Garmin metrics
    """
    # Initialize Garmin client
    client = initialize_garmin_client()
    if not client:
        print("Failed to initialize Garmin client. Exiting.")
        return None
    
    # Determine date range
    start_date, end_date = get_date_range(days, end_date)
    if verbose:
        print(f"Extracting Garmin data from {start_date} to {end_date}")
    
    all_data = []
    
    # Process each day
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        if verbose:
            print(f"Processing data for {date_str}...")
        
        try:
            # Get various data types for the day
            sleep_data = client.get_sleep_data(date_str)
            stress_data = client.get_stress_data(date_str)
            respiration_data = client.get_respiration_data(date_str)
            hrv_data = client.get_hrv_data(date_str)
            
            # Process sleep data
            sleep_stages = extract_sleep_stages(sleep_data)
            
            # Process stress data
            min_sustained_stress = find_minimum_sustained_stress(stress_data)
            
            # Process respiration data
            respiration_metrics = extract_respiration_data(respiration_data)
            
            # Process HRV data
            hrv_metrics = extract_hrv_data(hrv_data)
            
            # Compile daily data
            daily_data = {
                "date": date_str,
                "deep_sleep_seconds": sleep_stages["deep_sleep_seconds"],
                "light_sleep_seconds": sleep_stages["light_sleep_seconds"],
                "rem_sleep_seconds": sleep_stages["rem_sleep_seconds"],
                "awake_sleep_seconds": sleep_stages["awake_sleep_seconds"],
                "min_sustained_stress": min_sustained_stress,
                "avg_respiration_rate": respiration_metrics["avg_respiration_rate"],
                "lowest_respiration_rate": respiration_metrics["lowest_respiration_rate"],
                "highest_respiration_rate": respiration_metrics["highest_respiration_rate"],
                "avg_hrv": hrv_metrics["avg_hrv"],
                "weekly_avg_hrv": hrv_metrics["weekly_avg_hrv"],
                "last_night_avg_hrv": hrv_metrics["last_night_avg_hrv"],
                "last_night_hrv_status": hrv_metrics["last_night_hrv_status"]
            }
            
            all_data.append(daily_data)
            if verbose and len(all_data) % 10 == 0:  # Log progress every 10 days
                print(f"  Collected data for {len(all_data)} days...")
            
        except Exception as e:
            if verbose:
                print(f"Error processing data for {date_str}: {e}")
        
        current_date += datetime.timedelta(days=1)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    if len(df) == 0:
        print("No data was extracted. Check if your Garmin account has data for the specified date range.")
        return df
    
    if verbose:
        print(f"Successfully extracted data for {len(df)} days.")
        print(f"Sample of extracted data:")
        print(df.head(3))
    
    # Save to CSV if requested
    if save and len(df) > 0:
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(GARMIN_ADVANCED_FILE), exist_ok=True)
            
            # Save the data
            df.to_csv(GARMIN_ADVANCED_FILE, index=False)
            
            # Verify the file was created
            if os.path.exists(GARMIN_ADVANCED_FILE):
                file_size = os.path.getsize(GARMIN_ADVANCED_FILE)
                if verbose:
                    print(f"Advanced Garmin metrics saved to {GARMIN_ADVANCED_FILE} ({file_size/1024:.1f} KB)")
            else:
                print(f"Warning: File {GARMIN_ADVANCED_FILE} was not created successfully!")
        except Exception as e:
            print(f"Error saving data to {GARMIN_ADVANCED_FILE}: {e}")
    
    return df

def merge_with_daily_data(advanced_df=None, save=True, verbose=True):
    """
    Merge advanced Garmin metrics with daily data.
    
    Args:
        advanced_df (pandas.DataFrame, optional): Advanced metrics DataFrame.
                                                 If None, will try to load from file.
        save (bool): Whether to save results to CSV
        verbose (bool): Whether to print progress messages
    
    Returns:
        pandas.DataFrame: Enhanced Garmin data with all metrics
    """
    # Load advanced metrics if not provided
    if advanced_df is None:
        try:
            if not os.path.exists(GARMIN_ADVANCED_FILE):
                print(f"Error: Advanced metrics file {GARMIN_ADVANCED_FILE} does not exist.")
                return None
                
            advanced_df = pd.read_csv(GARMIN_ADVANCED_FILE)
            if verbose:
                print(f"Loaded advanced Garmin metrics from {GARMIN_ADVANCED_FILE}")
                print(f"Advanced metrics shape: {advanced_df.shape}")
        except Exception as e:
            if verbose:
                print(f"Error loading advanced Garmin metrics: {e}")
            return None
    
    # Load daily data
    try:
        if not os.path.exists(GARMIN_DAILY_FILE):
            print(f"Error: Daily data file {GARMIN_DAILY_FILE} does not exist.")
            print(f"Make sure to run the regular ETL process first to create {GARMIN_DAILY_FILE}")
            return None
            
        daily_df = pd.read_csv(GARMIN_DAILY_FILE)
        if verbose:
            print(f"Loaded daily Garmin data from {GARMIN_DAILY_FILE}")
            print(f"Daily data shape: {daily_df.shape}")
    except Exception as e:
        if verbose:
            print(f"Error loading daily Garmin data: {e}")
        return None
    
    # Convert date columns to datetime for proper merging
    advanced_df['date'] = pd.to_datetime(advanced_df['date'])
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    # Merge datasets
    enhanced_df = pd.merge(daily_df, advanced_df, on='date', how='left')
    
    if verbose:
        print(f"Merged data: {len(daily_df)} daily records, {len(advanced_df)} advanced records")
        print(f"Result: {len(enhanced_df)} records")
        
        # Check for missing values after merge
        missing_counts = enhanced_df[advanced_df.columns[1:]].isnull().sum()
        if missing_counts.sum() > 0:
            print("\nMissing values in merged data:")
            for col, count in missing_counts.items():
                if count > 0:
                    pct = (count / len(enhanced_df)) * 100
                    print(f"  {col}: {count} missing values ({pct:.1f}%)")
    
    # Save to CSV if requested
    if save and len(enhanced_df) > 0:
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(GARMIN_ENHANCED_FILE), exist_ok=True)
            
            # Save the data
            enhanced_df.to_csv(GARMIN_ENHANCED_FILE, index=False)
            
            # Verify the file was created
            if os.path.exists(GARMIN_ENHANCED_FILE):
                file_size = os.path.getsize(GARMIN_ENHANCED_FILE)
                if verbose:
                    print(f"Enhanced Garmin data saved to {GARMIN_ENHANCED_FILE} ({file_size/1024:.1f} KB)")
            else:
                print(f"Warning: File {GARMIN_ENHANCED_FILE} was not created successfully!")
        except Exception as e:
            print(f"Error saving enhanced data to {GARMIN_ENHANCED_FILE}: {e}")
    
    return enhanced_df

def main():
    """Main function to run the ETL process"""
    print("Garmin Advanced ETL Tool")
    print("------------------------")
    
    if not HAS_GARMINCONNECT:
        print("Error: garminconnect package is required. Run 'pip install garminconnect' to install it.")
        return
    
    # Extract advanced Garmin data
    print("\nExtracting advanced Garmin metrics...")
    advanced_df = extract_garmin_data(days=90, verbose=True)
    
    if advanced_df is not None and len(advanced_df) > 0:
        # Merge with daily data
        print("\nMerging with daily Garmin data...")
        enhanced_df = merge_with_daily_data(advanced_df, verbose=True)
        
        if enhanced_df is not None:
            print("\nETL process completed successfully!")
            print(f"New columns available for analysis: {list(advanced_df.columns[1:])}")
            
            # Print data sample
            print("\nSample of enhanced data:")
            pd.set_option('display.max_columns', None)
            print(enhanced_df.head())
    else:
        print("Failed to extract advanced Garmin metrics.")

if __name__ == "__main__":
    main() 