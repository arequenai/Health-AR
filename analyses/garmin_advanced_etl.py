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
    
    # Return zeros if no data
    if not sleep_data:
        print("No sleep data found")
        return result
    
    # Print a small sample of the sleep data for debugging
    print("Sleep data keys:", list(sleep_data.keys()) if isinstance(sleep_data, dict) else "Not a dict")
    
    # Process activity level data directly
    try:
        # Handle the case where we have sleepLevels directly
        sleep_levels = []
        if isinstance(sleep_data, dict) and "sleepLevels" in sleep_data:
            sleep_levels = sleep_data["sleepLevels"]
        # Handle case where sleepLevels is under another key
        elif isinstance(sleep_data, dict) and "dailySleepDTO" in sleep_data:
            if "sleepLevels" in sleep_data:
                sleep_levels = sleep_data["sleepLevels"]
        
        # Process each level based on activity level
        for level in sleep_levels:
            if not isinstance(level, dict):
                continue
                
            # Extract times and activity level
            start_time = level.get("startGMT", None)
            end_time = level.get("endGMT", None)
            activity_level = level.get("activityLevel", None)
            
            if not (start_time and end_time and activity_level is not None):
                continue
                
            # Calculate duration
            try:
                # Try parsing with millisecond precision
                start_dt = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S.%f")
                end_dt = datetime.datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S.%f")
            except ValueError:
                try:
                    # Try parsing with zero decimal precision
                    start_dt = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S.0")
                    end_dt = datetime.datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S.0")
                except ValueError:
                    # Last resort - try without decimal
                    try:
                        start_dt = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
                        end_dt = datetime.datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S")
                    except ValueError:
                        print(f"Could not parse times: {start_time} to {end_time}")
                        continue
            
            # Calculate duration in seconds
            duration_seconds = (end_dt - start_dt).total_seconds()
            
            # Map activity level to sleep stage
            # Using Garmin's activity levels (0 = deep sleep, 1 = light sleep, 2 = REM, 3+ = awake)
            if activity_level <= 0.25:
                result["deep_sleep_seconds"] += duration_seconds
            elif activity_level <= 1.25:  # 0.5-1.25 is light sleep
                result["light_sleep_seconds"] += duration_seconds
            elif activity_level <= 2.25:  # 1.5-2.25 is REM
                result["rem_sleep_seconds"] += duration_seconds
            else:  # 2.5+ is awake
                result["awake_sleep_seconds"] += duration_seconds
                
        # If we have data in the result, print a summary
        if sum(result.values()) > 0:
            print("Extracted sleep stages: deep={:.1f}h, light={:.1f}h, rem={:.1f}h, awake={:.1f}h".format(
                result["deep_sleep_seconds"]/3600,
                result["light_sleep_seconds"]/3600,
                result["rem_sleep_seconds"]/3600,
                result["awake_sleep_seconds"]/3600
            ))
        
    except Exception as e:
        print(f"Error processing sleep stages: {e}")
    
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
    try:
        # Debug logging
        print("Stress data keys:", list(stress_data.keys()) if isinstance(stress_data, dict) else "Not a dict")
        
        if not stress_data or not isinstance(stress_data, dict):
            print("No valid stress data found")
            return None
            
        if "stressValuesArray" not in stress_data:
            print("No stress values array found")
            return None
            
        values = stress_data["stressValuesArray"]
        if not values or not isinstance(values, list):
            print("No valid stress values found")
            return None
            
        # Check the format of the entries - in newer Garmin API, each entry is a list [timestamp, value]
        # rather than a dict with dateTime and value keys
        if len(values) > 0 and isinstance(values[0], list):
            print("Found list-format stress data")
            # Process list-format data [timestamp, value]
            valid_stress_points = []
            for entry in values:
                if len(entry) >= 2:
                    timestamp, stress_value = entry[0], entry[1]
                    # Filter out invalid stress values (-1, -2, etc. indicate no measurement)
                    if isinstance(stress_value, (int, float)) and stress_value >= 0:
                        valid_stress_points.append((timestamp, stress_value))
            
            if not valid_stress_points:
                print("No valid stress measurements found")
                return None
                
            # Sort by timestamp
            valid_stress_points.sort(key=lambda x: x[0])
            
            # Find consecutive measurements that span the required duration
            min_sustained = 100  # Start with max possible value
            current_window = []
            
            for timestamp, value in valid_stress_points:
                # Add the current point
                current_window.append((timestamp, value))
                
                # Remove points that are outside our time window
                # Typical sample rate is every 5-15 minutes, so duration_mins needs to be factored appropriately
                time_threshold = timestamp - (duration_mins * 60 * 1000)  # Convert minutes to milliseconds
                current_window = [p for p in current_window if p[0] >= time_threshold]
                
                # If we have enough points in our window, compute the average
                if len(current_window) >= 3:  # Require at least 3 measurements for a meaningful average
                    window_duration = current_window[-1][0] - current_window[0][0]
                    # Check if window spans enough time (at least 80% of requested duration)
                    if window_duration >= (duration_mins * 60 * 1000 * 0.8):
                        avg_stress = sum(v for _, v in current_window) / len(current_window)
                        min_sustained = min(min_sustained, avg_stress)
            
            return min_sustained if min_sustained < 100 else None
            
        else:
            # Process original dict-format data
            valid_stress_points = []
            for entry in values:
                if not isinstance(entry, dict):
                    continue
                    
                value = entry.get("value")
                entry_time = entry.get("dateTime")
                
                if value is None or entry_time is None:
                    continue
                
                # Only include valid stress values (non-negative)
                if isinstance(value, (int, float)) and value >= 0:
                    valid_stress_points.append((entry_time, value))
            
            if not valid_stress_points:
                print("No valid stress measurements found in dict format")
                return None
                
            # Sort by timestamp
            valid_stress_points.sort(key=lambda x: x[0])
            
            # Find consecutive measurements that span the required duration
            min_sustained = 100  # Start with max possible value
            current_run = []
            
            for entry_time, value in valid_stress_points:
                if len(current_run) == 0 or current_run[-1][0] + 60 >= entry_time:
                    # Add to current run if it's a consecutive reading (within ~1 minute)
                    current_run.append((entry_time, value))
                else:
                    # Start new run
                    current_run = [(entry_time, value)]
                
                # Check if current run is long enough
                if len(current_run) >= 4:  # Typically 4 readings would span about 15-20 minutes
                    avg_stress = sum(v for _, v in current_run) / len(current_run)
                    min_sustained = min(min_sustained, avg_stress)
            
            return min_sustained if min_sustained < 100 else None
    except Exception as e:
        print(f"Error processing stress data: {e}")
        return None

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
    
    try:
        # Debug logging
        print("Respiration data keys:", list(respiration_data.keys()) if isinstance(respiration_data, dict) else "Not a dict")
        
        if not respiration_data or not isinstance(respiration_data, dict):
            print("No valid respiration data found")
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
    except Exception as e:
        print(f"Error processing respiration data: {e}")
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
    
    try:
        # Debug logging
        print("HRV data keys:", list(hrv_data.keys()) if isinstance(hrv_data, dict) else "Not a dict")
        
        if not hrv_data or not isinstance(hrv_data, dict):
            print("No valid HRV data found")
            return result
            
        if "hrvSummary" not in hrv_data:
            print("No HRV summary found")
            return result
        
        hrv_summary = hrv_data["hrvSummary"]
        if not isinstance(hrv_summary, dict):
            print("HRV summary is not a dictionary")
            return result
        
        # Extract weekly average HRV
        if "weekly" in hrv_summary and isinstance(hrv_summary["weekly"], dict) and "avgValueInMilliseconds" in hrv_summary["weekly"]:
            result["weekly_avg_hrv"] = hrv_summary["weekly"]["avgValueInMilliseconds"]
        
        # Extract last night's average HRV
        if "lastNight" in hrv_summary and isinstance(hrv_summary["lastNight"], dict):
            last_night = hrv_summary["lastNight"]
            if "avgValueInMilliseconds" in last_night:
                result["last_night_avg_hrv"] = last_night["avgValueInMilliseconds"]
                result["avg_hrv"] = last_night["avgValueInMilliseconds"]  # Use last night as primary HRV
            
            if "qualifierKey" in last_night:
                result["last_night_hrv_status"] = last_night["qualifierKey"]
        
        return result
    except Exception as e:
        print(f"Error processing HRV data: {e}")
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
            
            # Print success for this day
            print(f"Successfully processed data for {date_str}")
            
        except Exception as e:
            if verbose:
                print(f"Error processing data for {date_str}: {e}")
                import traceback
                traceback.print_exc()
        
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
    
    # Changed from 90 to 3 days
    print("\nExtracting advanced Garmin metrics...")
    advanced_df = extract_garmin_data(days=3, verbose=True)
    
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