import os
import sys
import json
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# Import the functions directly from the module file path
from analyses.garmin_advanced_etl import initialize_garmin_client, find_minimum_sustained_stress
from etl.config import DATA_START_DATE

# Define current_dir for use in other functions
current_dir = os.path.dirname(os.path.abspath(__file__))

def find_min_sustained_stress(stress_values, consecutive_count=3):
    """
    Find the minimum stress level that was sustained for at least
    consecutive_count consecutive measurements.
    
    Args:
        stress_values: List of stress value measurements (integers/floats)
        consecutive_count: Minimum number of consecutive measurements
        
    Returns:
        float: Minimum sustained stress level or None if not found
    """
    if not stress_values or len(stress_values) < consecutive_count:
        return None
        
    # Sort values to organize by stress level
    stress_values = sorted(stress_values)
    
    # Start with the lowest value as a candidate
    min_sustained = None
    
    # Function to check if a value is sustained in the original sequence
    def is_sustained(threshold, orig_sequence, count):
        longest_run = 0
        current_run = 0
        
        for val in orig_sequence:
            if val <= threshold:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 0
                
        return longest_run >= count
    
    # Try each stress value as a potential threshold
    unique_values = sorted(set(stress_values))
    
    for threshold in unique_values:
        if is_sustained(threshold, stress_values, consecutive_count):
            min_sustained = threshold
            break
            
    return min_sustained

# This function is generic and can be used for any physiological measurement
def find_min_sustained_value(values, consecutive_count=3):
    """
    Find the minimum value that was sustained for at least
    consecutive_count consecutive measurements.
    
    Args:
        values: List of measurements (integers/floats)
        consecutive_count: Minimum number of consecutive measurements
        
    Returns:
        float: Minimum sustained value or None if not found
    """
    if not values or len(values) < consecutive_count:
        return None
        
    # Sort values to organize by level
    sorted_values = sorted(values)
    
    # Start with the lowest value as a candidate
    min_sustained = None
    
    # Function to check if a value is sustained in the original sequence
    def is_sustained(threshold, orig_sequence, count):
        longest_run = 0
        current_run = 0
        
        for val in orig_sequence:
            if val <= threshold:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 0
                
        return longest_run >= count
    
    # Try each value as a potential threshold
    unique_values = sorted(set(sorted_values))
    
    for threshold in unique_values:
        if is_sustained(threshold, values, consecutive_count):
            min_sustained = threshold
            break
            
    return min_sustained

def extract_time_series_data(data_array):
    """
    Extract time series data from Garmin API response arrays
    
    Args:
        data_array: List of [timestamp, value] entries
        
    Returns:
        tuple: (timestamps, values) lists with parsed data
    """
    timestamps = []
    values = []
    
    if not data_array or not isinstance(data_array, list):
        return timestamps, values
    
    for entry in data_array:
        if isinstance(entry, list) and len(entry) >= 2:
            timestamp = entry[0]
            value = entry[1]
            
            # Skip invalid values (often negative numbers)
            if isinstance(value, (int, float)) and value >= 0:
                # Convert timestamp to datetime
                if isinstance(timestamp, (int, float)):
                    try:
                        dt = datetime.fromtimestamp(timestamp / 1000)  # Convert milliseconds to seconds
                        timestamps.append(dt)
                        values.append(value)
                    except:
                        # Skip invalid timestamps
                        pass
    
    return timestamps, values

def test_stress_function():
    """Test the find_minimum_sustained_stress function with real data"""
    print("Testing find_minimum_sustained_stress function")
    print("---------------------------------------------")
    
    # Initialize Garmin client
    client = initialize_garmin_client()
    if not client:
        print("Failed to initialize Garmin client. Exiting.")
        return
    
    # Define date range from config start date to today
    end_date = date.today()
    start_date = DATA_START_DATE
    
    print(f"Analyzing data from {start_date} to {end_date}")
    
    # Store all valid measurements from the past week
    all_stress_values = []
    all_hrv_values = []
    all_respiration_values = []
    
    daily_stress_values = {}
    daily_hrv_values = {}
    daily_respiration_values = {}
    daily_sleep_stages = {}
    
    stress_data_all_days = {}
    hrv_data_all_days = {}
    respiration_data_all_days = {}
    sleep_data_all_days = {}
    
    min_sustained_stress_by_day = {}
    min_sustained_respiration_by_day = {}
    
    # Create a list to store daily metrics for CSV
    daily_metrics = []
    
    # Fetch data for each day
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        print(f"\nGetting data for {date_str}")
        
        try:
            # Fetch all different data types for this day
            stress_data = client.get_stress_data(date_str)
            hrv_data = client.get_hrv_data(date_str)
            respiration_data = client.get_respiration_data(date_str)
            sleep_data = client.get_sleep_data(date_str)  # Add sleep data fetch
            
            print(f"Successfully retrieved data")
            
            # Save all data to our collections
            stress_data_all_days[date_str] = stress_data
            hrv_data_all_days[date_str] = hrv_data
            respiration_data_all_days[date_str] = respiration_data
            sleep_data_all_days[date_str] = sleep_data
            
            # Initialize daily metrics for this date
            daily_metric = {
                'date': date_str,
                'min_sustained_stress': None,
                'min_sustained_respiration': None,
                'stress_measurements': 0,
                'respiration_measurements': 0,
                'deep_sleep_duration': None,
                'rem_sleep_duration': None,
                'light_sleep_duration': None,
                'awake_duration': None,
                'total_sleep_duration': None,
                'sleep_efficiency': None
            }
            
            # Process sleep data
            if isinstance(sleep_data, dict):
                print("  Processing sleep data...")
                sleep_stages = {}
                
                # Extract sleep stages
                if "sleepTimeSeconds" in sleep_data:
                    total_sleep = sleep_data["sleepTimeSeconds"] / 3600  # Convert to hours
                    daily_metric['total_sleep_duration'] = total_sleep
                    print(f"  Total sleep duration: {total_sleep:.2f} hours")
                
                if "deepSleepSeconds" in sleep_data:
                    deep_sleep = sleep_data["deepSleepSeconds"] / 3600
                    daily_metric['deep_sleep_duration'] = deep_sleep
                    sleep_stages['deep'] = deep_sleep
                    print(f"  Deep sleep duration: {deep_sleep:.2f} hours")
                
                if "remSleepInSeconds" in sleep_data:
                    rem_sleep = sleep_data["remSleepInSeconds"] / 3600
                    daily_metric['rem_sleep_duration'] = rem_sleep
                    sleep_stages['rem'] = rem_sleep
                    print(f"  REM sleep duration: {rem_sleep:.2f} hours")
                
                if "lightSleepInSeconds" in sleep_data:
                    light_sleep = sleep_data["lightSleepInSeconds"] / 3600
                    daily_metric['light_sleep_duration'] = light_sleep
                    sleep_stages['light'] = light_sleep
                    print(f"  Light sleep duration: {light_sleep:.2f} hours")
                
                if "awakeSleepSeconds" in sleep_data:
                    awake_time = sleep_data["awakeSleepSeconds"] / 3600
                    daily_metric['awake_duration'] = awake_time
                    sleep_stages['awake'] = awake_time
                    print(f"  Awake duration: {awake_time:.2f} hours")
                
                if "sleepEfficiencyInPercent" in sleep_data:
                    efficiency = sleep_data["sleepEfficiencyInPercent"]
                    daily_metric['sleep_efficiency'] = efficiency
                    print(f"  Sleep efficiency: {efficiency}%")
                
                # Extract sleep scores
                if "sleepScores" in sleep_data:
                    sleep_scores = sleep_data["sleepScores"]
                    if "overall" in sleep_scores:
                        daily_metric['sleep_score'] = sleep_scores["overall"].get("value")
                        print(f"  Sleep score: {daily_metric['sleep_score']}")
                    
                    if "totalDuration" in sleep_scores:
                        daily_metric['sleep_duration_score'] = sleep_scores["totalDuration"].get("qualifierKey")
                        print(f"  Sleep duration score: {daily_metric['sleep_duration_score']}")
                    
                    if "stress" in sleep_scores:
                        daily_metric['sleep_stress_score'] = sleep_scores["stress"].get("qualifierKey")
                        print(f"  Sleep stress score: {daily_metric['sleep_stress_score']}")
                    
                    if "awakeCount" in sleep_scores:
                        daily_metric['sleep_awake_count_score'] = sleep_scores["awakeCount"].get("qualifierKey")
                        print(f"  Sleep awake count score: {daily_metric['sleep_awake_count_score']}")
                    
                    if "remPercentage" in sleep_scores:
                        daily_metric['sleep_rem_score'] = sleep_scores["remPercentage"].get("qualifierKey")
                        print(f"  Sleep REM score: {daily_metric['sleep_rem_score']}")
                    
                    if "restlessness" in sleep_scores:
                        daily_metric['sleep_restlessness_score'] = sleep_scores["restlessness"].get("qualifierKey")
                        print(f"  Sleep restlessness score: {daily_metric['sleep_restlessness_score']}")
                    
                    if "lightPercentage" in sleep_scores:
                        daily_metric['sleep_light_score'] = sleep_scores["lightPercentage"].get("qualifierKey")
                        print(f"  Sleep light score: {daily_metric['sleep_light_score']}")
                
                # Extract sleep stress data
                if "sleepStress" in sleep_data:
                    stress_values = [point["value"] for point in sleep_data["sleepStress"]]
                    if stress_values:
                        daily_metric['avg_sleep_stress'] = sum(stress_values) / len(stress_values)
                        daily_metric['min_sleep_stress'] = min(stress_values)
                        daily_metric['max_sleep_stress'] = max(stress_values)
                        print(f"  Sleep stress - avg: {daily_metric['avg_sleep_stress']:.1f}, min: {daily_metric['min_sleep_stress']}, max: {daily_metric['max_sleep_stress']}")
                
                # Extract body battery data
                if "sleepBodyBattery" in sleep_data:
                    battery_values = [point["value"] for point in sleep_data["sleepBodyBattery"]]
                    if battery_values:
                        daily_metric['avg_sleep_body_battery'] = sum(battery_values) / len(battery_values)
                        daily_metric['min_sleep_body_battery'] = min(battery_values)
                        daily_metric['max_sleep_body_battery'] = max(battery_values)
                        print(f"  Sleep body battery - avg: {daily_metric['avg_sleep_body_battery']:.1f}, min: {daily_metric['min_sleep_body_battery']}, max: {daily_metric['max_sleep_body_battery']}")
                
                daily_sleep_stages[date_str] = sleep_stages
            
            # Process stress data
            if isinstance(stress_data, dict) and "stressValuesArray" in stress_data:
                values = stress_data["stressValuesArray"]
                daily_values = []
                
                if isinstance(values, list):
                    timestamps, stress_vals = extract_time_series_data(values)
                    all_stress_values.extend(stress_vals)
                    daily_values = stress_vals
                
                daily_stress_values[date_str] = daily_values
                daily_metric['stress_measurements'] = len(daily_values)
                print(f"  Found {len(daily_values)} valid stress measurements")
                
                # Find minimum sustained stress for this day
                if daily_values:
                    min_sustained = find_min_sustained_value(daily_values, consecutive_count=3)
                    min_sustained_stress_by_day[date_str] = min_sustained
                    daily_metric['min_sustained_stress'] = min_sustained
                    print(f"  Minimum sustained stress level (3+ consecutive readings): {min_sustained}")
            
            # Process HRV data
            if isinstance(hrv_data, dict):
                print("  HRV data keys:", list(hrv_data.keys()))
                daily_hrv_vals = []
                
                # Check for rmssd data which contains HRV values
                if "hrvValues" in hrv_data:
                    print("  Found hrvValues array")
                    hrv_values_array = hrv_data["hrvValues"]
                    timestamps, hrv_vals = extract_time_series_data(hrv_values_array)
                    all_hrv_values.extend(hrv_vals)
                    daily_hrv_vals = hrv_vals
                    print(f"  Found {len(daily_hrv_vals)} valid HRV measurements")
                    
                # Try alternative format with timeOffsetHrvSamples
                elif "timeOffsetHrvSamples" in hrv_data:
                    print("  Found timeOffsetHrvSamples array")
                    samples = hrv_data["timeOffsetHrvSamples"]
                    if isinstance(samples, list):
                        for sample in samples:
                            if isinstance(sample, dict) and "rmssd" in sample:
                                rmssd = sample.get("rmssd")
                                if isinstance(rmssd, (int, float)) and rmssd > 0:
                                    all_hrv_values.append(rmssd)
                                    daily_hrv_vals.append(rmssd)
                    print(f"  Found {len(daily_hrv_vals)} valid HRV measurements")
                else:
                    print("  No HRV time series data found")
                
                daily_hrv_values[date_str] = daily_hrv_vals
            
            # Process respiration data
            if isinstance(respiration_data, dict):
                daily_resp_vals = []
                
                # Check for respiration values array
                if "respirationValuesArray" in respiration_data:
                    print("  Found respirationValuesArray")
                    resp_values_array = respiration_data["respirationValuesArray"]
                    timestamps, resp_vals = extract_time_series_data(resp_values_array)
                    all_respiration_values.extend(resp_vals)
                    daily_resp_vals = resp_vals
                    daily_metric['respiration_measurements'] = len(daily_resp_vals)
                    print(f"  Found {len(daily_resp_vals)} valid respiration measurements")
                    
                    # Find minimum sustained respiration rate for this day
                    if daily_resp_vals:
                        min_sustained_resp = find_min_sustained_value(daily_resp_vals, consecutive_count=3)
                        min_sustained_respiration_by_day[date_str] = min_sustained_resp
                        daily_metric['min_sustained_respiration'] = min_sustained_resp
                        print(f"  Minimum sustained respiration rate (3+ consecutive readings): {min_sustained_resp}")
                else:
                    print("  No respiration time series data found")
                
                daily_respiration_values[date_str] = daily_resp_vals
            
            # Add this day's metrics to our list
            daily_metrics.append(daily_metric)
            
        except Exception as e:
            print(f"Error retrieving data for {date_str}: {e}")
            import traceback
            traceback.print_exc()
        
        current_date += timedelta(days=1)
    
    # Create DataFrame from daily metrics and save to CSV
    df_daily_metrics = pd.DataFrame(daily_metrics)
    csv_filename = os.path.join(current_dir, 'daily_metrics.csv')
    df_daily_metrics.to_csv(csv_filename, index=False)
    print(f"\nDaily metrics saved to {csv_filename}")
    
    # Print summary of minimum sustained stress by day
    print("\n===== MINIMUM SUSTAINED STRESS LEVELS =====")
    print("Day               Min Sustained Stress")
    print("----------------------------------------")
    for date_str, min_stress in min_sustained_stress_by_day.items():
        print(f"{date_str}        {min_stress if min_stress is not None else 'None'}")
    
    # Print summary of minimum sustained respiration rates by day
    print("\n===== MINIMUM SUSTAINED RESPIRATION RATES =====")
    print("Day               Min Sustained Respiration")
    print("---------------------------------------------")
    for date_str, min_resp in min_sustained_respiration_by_day.items():
        print(f"{date_str}        {min_resp if min_resp is not None else 'None'}")
    
    # Create visualizations
    create_visualizations(all_stress_values, all_respiration_values, 
                         daily_stress_values, daily_respiration_values,
                         min_sustained_stress_by_day, min_sustained_respiration_by_day,
                         stress_data_all_days, respiration_data_all_days,
                         daily_sleep_stages)

def create_visualizations(all_stress_values, all_respiration_values,
                         daily_stress_values, daily_respiration_values,
                         min_sustained_stress_by_day, min_sustained_respiration_by_day,
                         stress_data_all_days, respiration_data_all_days,
                         daily_sleep_stages):
    """Create all visualizations"""
    os.makedirs(os.path.join(current_dir, 'visualizations'), exist_ok=True)
    
    # Create histograms
    if all_stress_values:
        create_histogram(all_stress_values, 'Stress', 'skyblue', 'black')
    
    if all_respiration_values:
        create_histogram(all_respiration_values, 'Respiration Rate', 'plum', 'purple')
    
    # Create daily averages charts
    create_daily_charts(daily_stress_values, 'Stress', 'skyblue', 'darkblue')
    create_daily_charts(daily_respiration_values, 'Respiration Rate', 'plum', 'purple')
    
    # Create minimum sustained charts
    create_min_sustained_chart(min_sustained_stress_by_day, 'Stress', 'lightblue', 'navy')
    create_min_sustained_chart(min_sustained_respiration_by_day, 'Respiration Rate', 'lightblue', 'navy')
    
    # Create daily timelines
    create_daily_timelines(stress_data_all_days, respiration_data_all_days)
    
    # Create sleep stage visualizations
    create_sleep_stage_charts(daily_sleep_stages)

def create_sleep_stage_charts(daily_sleep_stages):
    """Create visualizations for sleep stages"""
    if not daily_sleep_stages:
        return
    
    # Prepare data for stacked bar chart
    dates = []
    deep_sleep = []
    rem_sleep = []
    light_sleep = []
    awake_time = []
    
    for date_str, stages in daily_sleep_stages.items():
        dates.append(date_str)
        deep_sleep.append(stages.get('deep', 0))
        rem_sleep.append(stages.get('rem', 0))
        light_sleep.append(stages.get('light', 0))
        awake_time.append(stages.get('awake', 0))
    
    # Create stacked bar chart
    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(dates))
    
    plt.bar(dates, deep_sleep, label='Deep Sleep', bottom=bottom, color='navy')
    bottom += deep_sleep
    plt.bar(dates, rem_sleep, label='REM Sleep', bottom=bottom, color='purple')
    bottom += rem_sleep
    plt.bar(dates, light_sleep, label='Light Sleep', bottom=bottom, color='lightblue')
    bottom += light_sleep
    plt.bar(dates, awake_time, label='Awake', bottom=bottom, color='red')
    
    plt.xlabel('Date')
    plt.ylabel('Duration (hours)')
    plt.title('Sleep Stages by Day')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.75, linestyle='--')
    plt.tight_layout()
    
    # Save the figure
    filename = os.path.join(current_dir, 'visualizations', 'sleep_stages.png')
    plt.savefig(filename)
    print(f"Sleep stages chart saved to {filename}")
    
    # Create pie chart for average sleep stage distribution
    plt.figure(figsize=(8, 8))
    avg_deep = np.mean(deep_sleep)
    avg_rem = np.mean(rem_sleep)
    avg_light = np.mean(light_sleep)
    avg_awake = np.mean(awake_time)
    
    stages = ['Deep Sleep', 'REM Sleep', 'Light Sleep', 'Awake']
    values = [avg_deep, avg_rem, avg_light, avg_awake]
    colors = ['navy', 'purple', 'lightblue', 'red']
    
    plt.pie(values, labels=stages, colors=colors, autopct='%1.1f%%')
    plt.title('Average Sleep Stage Distribution')
    
    # Save the figure
    filename = os.path.join(current_dir, 'visualizations', 'sleep_stages_pie.png')
    plt.savefig(filename)
    print(f"Sleep stages pie chart saved to {filename}")

def create_histogram(values, metric_name, color, edge_color):
    """Create histogram for a metric"""
    print(f"\nCreating histogram from {len(values)} {metric_name.lower()} values")
    
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=20, alpha=0.7, color=color, edgecolor=edge_color)
    plt.xlabel(metric_name)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {metric_name} Values')
    
    # Add statistics
    avg_val = np.mean(values)
    median_val = np.median(values)
    min_val = min(values)
    max_val = max(values)
    
    stats_text = f"Mean: {avg_val:.2f}\nMedian: {median_val:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}"
    plt.annotate(stats_text, xy=(0.75, 0.85), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.grid(axis='y', alpha=0.75, linestyle='--')
    plt.tight_layout()
    
    # Save the figure
    hist_filename = os.path.join(current_dir, 'visualizations', f'{metric_name.lower().replace(" ", "_")}_histogram.png')
    plt.savefig(hist_filename)
    print(f"{metric_name} histogram saved to {hist_filename}")

def create_daily_timelines(stress_data_all_days, respiration_data_all_days):
    """Create timelines for the most recent day with data"""
    if stress_data_all_days:
        for date_str in sorted(stress_data_all_days.keys(), reverse=True):
            stress_data = stress_data_all_days[date_str]
            if isinstance(stress_data, dict) and "stressValuesArray" in stress_data:
                values = stress_data["stressValuesArray"]
                timestamps, stress_vals = extract_time_series_data(values)
                
                if len(timestamps) > 0 and len(stress_vals) > 0:
                    create_daily_timeline(date_str, timestamps, stress_vals, 'Stress')
                    break
    
    if respiration_data_all_days:
        for date_str in sorted(respiration_data_all_days.keys(), reverse=True):
            resp_data = respiration_data_all_days[date_str]
            if isinstance(resp_data, dict) and "respirationValuesArray" in resp_data:
                values = resp_data["respirationValuesArray"]
                timestamps, resp_vals = extract_time_series_data(values)
                
                if len(timestamps) > 0 and len(resp_vals) > 0:
                    create_daily_timeline(date_str, timestamps, resp_vals, 'Respiration Rate')
                    break

def create_daily_charts(daily_values, metric_name, color, edge_color):
    """Create bar charts showing daily averages"""
    dates = []
    avg_values = []
    
    for date_str, values in daily_values.items():
        if values:  # Only include days with data
            dates.append(date_str)
            avg_values.append(np.mean(values))
    
    if not dates:
        return
    
    plt.figure(figsize=(10, 6))
    plt.bar(dates, avg_values, color=color, edgecolor=edge_color)
    plt.xlabel('Date')
    plt.ylabel(f'Average {metric_name}')
    plt.title(f'Daily Average {metric_name} (Last 7 Days)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.75, linestyle='--')
    plt.tight_layout()
    
    # Save the figure
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                           'visualizations', f'daily_{metric_name.lower().replace(" ", "_")}.png')
    plt.savefig(filename)
    print(f"Daily {metric_name} chart saved to {filename}")

def create_min_sustained_chart(min_sustained_values, metric_name, color, edge_color):
    """Create bar chart showing minimum sustained levels by day"""
    dates = []
    min_values = []
    
    for date_str, value in min_sustained_values.items():
        if value is not None:  # Only include days with valid data
            dates.append(date_str)
            min_values.append(value)
    
    if not dates:
        return
    
    plt.figure(figsize=(10, 6))
    plt.bar(dates, min_values, color=color, edgecolor=edge_color)
    plt.xlabel('Date')
    plt.ylabel(f'Minimum Sustained {metric_name}')
    plt.title(f'Daily Minimum Sustained {metric_name} (Last 7 Days)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.75, linestyle='--')
    plt.tight_layout()
    
    # Save the figure
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                           'visualizations', f'min_sustained_{metric_name.lower().replace(" ", "_")}.png')
    plt.savefig(filename)
    print(f"Minimum sustained {metric_name} chart saved to {filename}")

def create_daily_timeline(date_str, timestamps, values, metric_name):
    """Create a timeline of values throughout a single day"""
    if not timestamps or not values or len(timestamps) != len(values):
        return
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({'timestamp': timestamps, 'value': values})
    df = df.sort_values('timestamp')
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['value'], color='navy' if metric_name == 'Stress' else 'darkgreen')
    
    # Format the x-axis to show hours
    plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
    # Add labels and title
    plt.xlabel('Time of Day')
    plt.ylabel(f'{metric_name} Level')
    plt.title(f'{metric_name} Throughout the Day ({date_str})')
    
    # Add grid and improve visual appearance
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                           'visualizations', f'{metric_name.lower().replace(" ", "_")}_timeline.png')
    plt.savefig(filename)
    print(f"{metric_name} timeline saved to {filename}")

if __name__ == "__main__":
    test_stress_function() 