import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys

# Configure logging with a more visible format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def categorize_activity(types):
    """Categorize activity type for TSS calculation."""
    types = set(types)
    if any(t in types for t in ['running', 'trail_running']):
        return 'running'
    elif 'strength_training' in types:
        return 'strength'
    else:
        return 'other'

def validate_tss_calculation(daily_data):
    """Validate TSS calculations and log any potential issues.
    
    Args:
        daily_data (pd.DataFrame): DataFrame with calculated TSS values
    
    Returns:
        bool: True if validation passes, False if there are critical issues
    """
    # Create a copy for validation to avoid modifying the original data
    validation_df = daily_data.copy()
    
    # Define validation thresholds
    MAX_DAILY_TSS = 400  # Maximum reasonable TSS for a single day
    MIN_DAILY_TSS = 0    # Minimum TSS (should never be negative)
    MAX_DAILY_CTL = 150  # Maximum reasonable CTL
    MAX_DAILY_ATL = 200  # Maximum reasonable ATL
    MAX_TSB_CHANGE = 30  # Maximum reasonable day-to-day TSB change
    
    validation_passed = True
    
    # Check for negative values
    if (validation_df[['TSS', 'CTL', 'ATL']] < MIN_DAILY_TSS).any().any():
        logger.error("Found negative values in TSS/CTL/ATL calculations")
        validation_passed = False
    
    # Check for unreasonably high TSS values
    high_tss = validation_df[validation_df['TSS'] > MAX_DAILY_TSS]
    if not high_tss.empty:
        logger.warning(f"Found {len(high_tss)} days with unusually high TSS (>{MAX_DAILY_TSS})")
        for _, row in high_tss.iterrows():
            logger.warning(f"High TSS on {row['date']}: {row['TSS']}")
    
    # Check for unreasonably high CTL/ATL values
    if (validation_df['CTL'] > MAX_DAILY_CTL).any():
        logger.warning(f"Found CTL values exceeding {MAX_DAILY_CTL}")
    if (validation_df['ATL'] > MAX_DAILY_ATL).any():
        logger.warning(f"Found ATL values exceeding {MAX_DAILY_ATL}")
    
    # Check for large day-to-day TSB changes
    tsb_changes = validation_df['TSB'].diff()
    large_changes = validation_df[abs(tsb_changes) > MAX_TSB_CHANGE]
    if not large_changes.empty:
        logger.warning(f"Found {len(large_changes)} days with large TSB changes (>{MAX_TSB_CHANGE})")
        for _, row in large_changes.iterrows():
            logger.warning(f"Large TSB change on {row['date']}: {tsb_changes.loc[row.name]:.1f}")
    
    # Check for missing days
    dates = pd.to_datetime(validation_df['date'])
    date_range = pd.date_range(dates.min(), dates.max())
    actual_dates = set([d.strftime('%Y-%m-%d') for d in dates])
    expected_dates = set([d.strftime('%Y-%m-%d') for d in date_range])
    missing_dates = expected_dates - actual_dates
    if missing_dates:
        logger.warning(f"Found {len(missing_dates)} missing days in the data")
        logger.warning(f"First few missing dates: {sorted(list(missing_dates))[:5]}")
    
    return validation_passed

def calculate_tss(activity_data):
    """Calculate TSS from Garmin data using derived formulas.
    
    Args:
        activity_data (pd.DataFrame): DataFrame containing Garmin activity data
            Required columns: type, training_load, duration, aerobic_te, 
            anaerobic_te (optional), avg_hr (optional)
    
    Returns:
        pd.DataFrame: Daily TSS values with date index
    """
    logger.info("Starting TSS calculation...")
    
    # Ensure numeric columns
    numeric_cols = ['training_load', 'duration', 'aerobic_te', 'anaerobic_te', 'avg_hr']
    for col in numeric_cols:
        if col in activity_data.columns:
            activity_data[col] = pd.to_numeric(activity_data[col], errors='coerce')
    
    # Fill NaN values with 0
    activity_data = activity_data.fillna(0)
    
    # Group by date and calculate daily metrics
    daily_data = activity_data.groupby('date').agg({
        'type': list,
        'training_load': 'sum',
        'duration': 'sum',
        'aerobic_te': 'sum',
        'anaerobic_te': 'sum',
        'avg_hr': 'mean'
    }).reset_index()
    
    # Create a complete date range including today
    start_date = daily_data['date'].min()
    end_date = pd.Timestamp.today().normalize()  # Use today's date
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create a DataFrame with all dates including today
    complete_daily_data = pd.DataFrame({'date': date_range.strftime('%Y-%m-%d')})
    
    # Merge with actual data, filling missing values
    daily_data = pd.merge(
        complete_daily_data,
        daily_data,
        on='date',
        how='left'
    )
    
    # Fill missing values for calculations
    daily_data['type'] = daily_data['type'].fillna('').apply(lambda x: [] if x == '' else x)
    daily_data['training_load'] = daily_data['training_load'].fillna(0)
    daily_data['duration'] = daily_data['duration'].fillna(0)
    daily_data['aerobic_te'] = daily_data['aerobic_te'].fillna(0)
    daily_data['anaerobic_te'] = daily_data['anaerobic_te'].fillna(0)
    daily_data['avg_hr'] = daily_data['avg_hr'].fillna(0)
    
    # Add activity category and trail flag
    daily_data['activity_category'] = daily_data['type'].apply(categorize_activity)
    daily_data['is_trail'] = daily_data['type'].apply(
        lambda x: 1 if any('trail_running' in t for t in x) else 0
    )
    
    # Calculate TSS for each category using derived formulas
    def calculate_running_tss(row):
        if row['duration'] == 0:
            return 0
        return (-15.4521 +
                (0.7234 * row['training_load']) +
                (0.0012 * row['duration']) +
                (-18.3421 * row['aerobic_te']) +
                (0.2145 * row['avg_hr']) +
                (5.8932 * row['is_trail']))
    
    def calculate_strength_tss(row):
        if row['duration'] == 0:
            return 0
        return (12.8932 +
                (0.4567 * row['training_load']) +
                (0.0008 * row['duration']) +
                (-12.4532 * row['aerobic_te']) +
                (8.3421 * row['anaerobic_te']))
    
    def calculate_other_tss(row):
        if row['duration'] == 0:
            return 0
        return (8.2341 +
                (0.8605 * row['training_load']) +
                (0.0007 * row['duration']) +
                (-24.0555 * row['aerobic_te']))
    
    # Apply formulas based on activity category
    daily_data['TSS'] = daily_data.apply(
        lambda row: calculate_running_tss(row) if row['activity_category'] == 'running'
        else calculate_strength_tss(row) if row['activity_category'] == 'strength'
        else calculate_other_tss(row),
        axis=1
    )
    
    # Ensure TSS is non-negative and round to 1 decimal
    daily_data['TSS'] = np.maximum(daily_data['TSS'], 0).round(1)
    
    # Calculate CTL, ATL, and TSB
    ctl_decay = np.exp(-1/42)  # 42-day time constant for CTL
    atl_decay = np.exp(-1/7)   # 7-day time constant for ATL
    
    daily_data = daily_data.sort_values('date')
    daily_data['CTL'] = 0.0  # Initialize as float
    daily_data['ATL'] = 0.0  # Initialize as float
    
    # Initialize with reasonable starting values
    if len(daily_data) > 0:
        daily_data.iloc[0, daily_data.columns.get_loc('CTL')] = float(daily_data['TSS'].iloc[0])
        daily_data.iloc[0, daily_data.columns.get_loc('ATL')] = float(daily_data['TSS'].iloc[0])
    
    # Calculate rolling averages
    for i in range(1, len(daily_data)):
        prev_ctl = daily_data.iloc[i-1]['CTL']
        prev_atl = daily_data.iloc[i-1]['ATL']
        tss = daily_data.iloc[i]['TSS']
        
        daily_data.iloc[i, daily_data.columns.get_loc('CTL')] = (prev_ctl * ctl_decay) + (tss * (1 - ctl_decay))
        daily_data.iloc[i, daily_data.columns.get_loc('ATL')] = (prev_atl * atl_decay) + (tss * (1 - atl_decay))
    
    # Calculate Training Stress Balance (TSB)
    daily_data['TSB'] = daily_data['CTL'] - daily_data['ATL']
    
    # Return only the necessary columns
    result = daily_data[['date', 'TSS', 'CTL', 'ATL', 'TSB']].copy()
    result = result.round(1)
    
    # Validate the calculations
    if not validate_tss_calculation(result):
        logger.warning("TSS calculations completed but validation found issues")
    else:
        logger.info("TSS calculations completed and validated successfully")
    
    return result

def get_tss_data(activity_data, start_date=None):
    """Get TSS metrics from Garmin activity data.
    
    Args:
        activity_data (pd.DataFrame): DataFrame containing Garmin activity data
        start_date (datetime, optional): Start date for calculations
    
    Returns:
        pd.DataFrame: Daily TSS metrics
    """
    if start_date:
        activity_data = activity_data[activity_data['date'] >= start_date].copy()
    
    tss_data = calculate_tss(activity_data)
    return tss_data 