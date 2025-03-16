import requests
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def fetch_nightscout_data(base_url, token, days_back=30):
    """
    Fetch glucose data from Nightscout using the token in the URL
    """
    # Calculate start time (days_back days ago)
    start_date = datetime.now() - timedelta(days=days_back)
    start_time = int(start_date.timestamp() * 1000)  # Convert to milliseconds
    
    # Format the API URL with token in query string
    entries_url = f"{base_url}/api/v1/entries.json?token={token}&count=10000&find[date][$gt]={start_time}"
    
    print(f"Fetching data from: {base_url}/api/v1/entries.json")
    print(f"Time range: Last {days_back} days (since {start_date.strftime('%Y-%m-%d')})")
    
    # Make the request
    try:
        response = requests.get(entries_url, timeout=10)
        
        if response.status_code == 200:
            entries = response.json()
            return entries
        else:
            print(f"Error fetching data: Status code {response.status_code}")
            print(f"Response: {response.text[:200]}")  # Print first 200 chars of response
            return []
    except Exception as e:
        print(f"Exception during request: {str(e)}")
        return []

def process_entries(entries):
    """
    Process entries into a pandas DataFrame
    """
    if not entries:
        return None
    
    # Create DataFrame
    df = pd.DataFrame(entries)
    
    # Convert timestamp to datetime
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    
    # Add date column (without time)
    df['day'] = df['date'].dt.date
    
    # Add hour column for time-of-day analysis
    df['hour'] = df['date'].dt.hour
    
    return df

def save_to_csv(df, filename='glucose_data.csv'):
    """
    Save DataFrame to CSV
    """
    os.makedirs('data', exist_ok=True)
    file_path = os.path.join('data', filename)
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")
    
def create_daily_summary(df):
    """
    Create daily summary with min, max, mean values
    """
    if df is None or df.empty:
        return None
    
    # Ensure we have the 'sgv' column (sensor glucose value)
    if 'sgv' not in df.columns:
        print("Warning: 'sgv' column not found in data")
        return None
    
    # Group by day and calculate statistics
    daily = df.groupby('day').agg({
        'sgv': ['count', 'mean', 'min', 'max', 'std']
    }).reset_index()
    
    # Flatten MultiIndex columns
    daily.columns = ['date', 'readings_count', 'mean_glucose', 'min_glucose', 'max_glucose', 'std_glucose']
    
    # Calculate time in range (70-180 mg/dL)
    in_range_min = 70
    in_range_max = 180
    
    time_in_range_list = []
    time_above_range_list = []
    time_below_range_list = []
    
    for day in daily['date']:
        day_data = df[df['day'] == day]
        readings_count = len(day_data)
        
        # Calculate time in ranges
        in_range = day_data[(day_data['sgv'] >= in_range_min) & (day_data['sgv'] <= in_range_max)]
        above_range = day_data[day_data['sgv'] > in_range_max]
        below_range = day_data[day_data['sgv'] < in_range_min]
        
        time_in_range_pct = (len(in_range) / readings_count * 100) if readings_count > 0 else 0
        time_above_range_pct = (len(above_range) / readings_count * 100) if readings_count > 0 else 0
        time_below_range_pct = (len(below_range) / readings_count * 100) if readings_count > 0 else 0
        
        time_in_range_list.append(time_in_range_pct)
        time_above_range_list.append(time_above_range_pct)
        time_below_range_list.append(time_below_range_pct)
    
    # Add time in range metrics to the dataframe
    daily['time_in_range_pct'] = time_in_range_list
    daily['time_above_range_pct'] = time_above_range_list
    daily['time_below_range_pct'] = time_below_range_list
    
    # Get morning reading for fasting glucose approximation (6-9 AM)
    fasting_glucose_list = []
    
    for day in daily['date']:
        day_data = df[df['day'] == day]
        morning_readings = day_data[(day_data['date'].dt.hour >= 6) & (day_data['date'].dt.hour <= 9)]
        
        if morning_readings.empty:
            fasting_glucose = None
        else:
            fasting_glucose = morning_readings['sgv'].mean()
            
        fasting_glucose_list.append(fasting_glucose)
        
    daily['fasting_glucose'] = fasting_glucose_list
    
    # Calculate estimated A1C (estimated using mean glucose)
    # Formula: A1C = (mean glucose + 46.7) / 28.7
    daily['estimated_a1c'] = (daily['mean_glucose'] + 46.7) / 28.7
    
    return daily

def create_time_of_day_analysis(df):
    """
    Create analysis of glucose levels by time of day
    """
    if df is None or df.empty or 'sgv' not in df.columns:
        return None
    
    # Group data by hour of day
    hourly = df.groupby('hour').agg({
        'sgv': ['count', 'mean', 'min', 'max', 'std']
    }).reset_index()
    
    # Flatten MultiIndex columns
    hourly.columns = ['hour', 'readings_count', 'mean_glucose', 'min_glucose', 'max_glucose', 'std_glucose']
    
    # Save to CSV
    save_to_csv(hourly, 'glucose_hourly.csv')
    
    return hourly

def generate_visualizations(daily_df, hourly_df, raw_df):
    """
    Generate basic visualizations from the data
    """
    try:
        os.makedirs('data/visualizations', exist_ok=True)
        
        if daily_df is not None and not daily_df.empty:
            # Daily average glucose plot
            plt.figure(figsize=(12, 6))
            plt.plot(daily_df['date'], daily_df['mean_glucose'], 'b-', label='Mean')
            plt.plot(daily_df['date'], daily_df['min_glucose'], 'g--', label='Min')
            plt.plot(daily_df['date'], daily_df['max_glucose'], 'r--', label='Max')
            plt.fill_between(daily_df['date'], daily_df['min_glucose'], daily_df['max_glucose'], alpha=0.2)
            plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
            plt.axhline(y=180, color='r', linestyle='-', alpha=0.3)
            plt.title('Daily Glucose Trends')
            plt.xlabel('Date')
            plt.ylabel('Glucose (mg/dL)')
            plt.legend()
            plt.tight_layout()
            plt.savefig('data/visualizations/daily_glucose_trends.png')
            
            # Time in range plot
            plt.figure(figsize=(12, 6))
            plt.bar(daily_df['date'], daily_df['time_in_range_pct'], color='g', label='In Range (70-180 mg/dL)')
            plt.bar(daily_df['date'], daily_df['time_above_range_pct'], bottom=daily_df['time_in_range_pct'], color='r', label='Above Range (>180 mg/dL)')
            plt.bar(daily_df['date'], daily_df['time_below_range_pct'], bottom=daily_df['time_in_range_pct'] + daily_df['time_above_range_pct'], color='y', label='Below Range (<70 mg/dL)')
            plt.title('Daily Time in Range')
            plt.xlabel('Date')
            plt.ylabel('Percentage of Time')
            plt.legend()
            plt.tight_layout()
            plt.savefig('data/visualizations/time_in_range.png')
        
        if hourly_df is not None and not hourly_df.empty:
            # Hourly average glucose plot
            plt.figure(figsize=(12, 6))
            plt.plot(hourly_df['hour'], hourly_df['mean_glucose'], 'b-', marker='o')
            plt.fill_between(hourly_df['hour'], hourly_df['mean_glucose'] - hourly_df['std_glucose'], 
                            hourly_df['mean_glucose'] + hourly_df['std_glucose'], alpha=0.2)
            plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
            plt.axhline(y=180, color='r', linestyle='-', alpha=0.3)
            plt.title('Average Glucose by Hour of Day')
            plt.xlabel('Hour of Day')
            plt.ylabel('Glucose (mg/dL)')
            plt.xticks(range(0, 24))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('data/visualizations/hourly_glucose_trends.png')
        
        print("Visualizations saved to data/visualizations/ directory")
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")

def main():
    # Your Nightscout URL and token
    nightscout_url = "https://3195.ns.gluroo.com"
    token = "3195c41e-6c98-4d8d-8435-ff5d307914a4"
    
    # Number of days of historical data to fetch
    days_back = 90
    
    # Fetch the data
    entries = fetch_nightscout_data(nightscout_url, token, days_back=days_back)
    
    print(f"Retrieved {len(entries)} glucose readings")
    
    if not entries:
        print("No data found. Check your Nightscout URL and token.")
        return
    
    # Print sample data for verification
    if entries:
        print("\nSample data (first entry):")
        print(entries[0])
    
    # Process entries
    df = process_entries(entries)
    if df is not None:
        print(f"\nProcessed data spanning from {df['date'].min()} to {df['date'].max()}")
        print(f"Total of {len(df)} glucose readings")
        
        # Save raw data
        save_to_csv(df, 'glucose_raw.csv')
        
        # Create and save daily summary
        daily_df = create_daily_summary(df)
        if daily_df is not None:
            save_to_csv(daily_df, 'glucose_daily.csv')
            print(f"Created daily summary with {len(daily_df)} days of data")
            
            # Calculate overall statistics
            overall_mean = daily_df['mean_glucose'].mean()
            overall_std = daily_df['mean_glucose'].std()
            avg_tir = daily_df['time_in_range_pct'].mean()
            est_a1c = daily_df['estimated_a1c'].mean()
            
            print("\nOverall Statistics:")
            print(f"Mean Glucose: {overall_mean:.1f} mg/dL")
            print(f"Standard Deviation: {overall_std:.1f} mg/dL")
            print(f"Average Time in Range: {avg_tir:.1f}%")
            print(f"Estimated A1C: {est_a1c:.1f}%")
            
            # Print sample of daily summary
            print("\nSample daily summary (first 5 days):")
            print(daily_df.head().to_string())
        else:
            print("Could not create daily summary. Check if 'sgv' column exists in data.")
        
        # Create time-of-day analysis
        hourly_df = create_time_of_day_analysis(df)
        if hourly_df is not None:
            print(f"Created hourly analysis with {len(hourly_df)} time points")
        
        # Generate visualizations
        try:
            import matplotlib.pyplot as plt
            print("\nGenerating visualizations...")
            generate_visualizations(daily_df, hourly_df, df)
        except ImportError:
            print("Matplotlib not installed. Skipping visualizations.")
            print("To generate visualizations, install matplotlib with: pip install matplotlib")

if __name__ == "__main__":
    main() 