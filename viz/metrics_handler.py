from typing import Dict, Any, Optional
import pandas as pd
from etl import config

def get_metrics() -> Optional[Dict[str, Dict[str, Dict[str, Any]]]]:
    """Get all metrics from various sources."""
    try:
        # Read MFP daily data
        df_mfp = pd.read_csv(config.MFP_DAILY_FILE)
        latest_mfp = df_mfp.iloc[-1]

        # Calculate last 7 days net calories
        l7d_net_calories = int(df_mfp.tail(7)['calories_net'].sum())

        # Read Whoop data
        df_whoop = pd.read_csv(config.WHOOP_SLEEP_RECOVERY_FILE)
        latest_whoop = df_whoop.iloc[-1]

        # Read Garmin data
        df_garmin = pd.read_csv(config.GARMIN_DAILY_FILE)
        latest_garmin = df_garmin.iloc[-1]
        
        # Calculate last 7 days metrics
        last_7d = df_garmin.tail(7)
        last_7d_distance = last_7d['totalDistanceMeters'].sum() / 1000  # Convert to km
        last_7d_altitude = last_7d['floorsAscendedInMeters'].sum()  # Already in meters

        # Format sleep time
        sleep_hours = latest_garmin['sleepingSeconds'] / 3600  # Convert seconds to hours
        sleep_hrs = int(sleep_hours)
        sleep_mins = int((sleep_hours - sleep_hrs) * 60)
        sleep_time = f"{sleep_hrs}:{sleep_mins:02d}"

        # Handle calories net label and value
        calories_net = int(latest_mfp['calories_net'])
        if calories_net > 0:
            cal_label = 'kcal over'
        else:
            cal_label = 'kcal rem'

        metrics = {
            'nutrition': {
                'primary': {'value': abs(calories_net), 'label': cal_label, 'color_value': calories_net},
                'secondary1': {'value': l7d_net_calories, 'label': 'net above L7d'},
                'secondary2': {'value': '-', 'label': 'kg'}    # Placeholder
            },
            'recovery': {
                'primary': {'value': int(latest_whoop['recovery_score']), 'label': 'recovery'},
                'secondary1': {'value': int(latest_garmin['bodyBatteryMostRecentValue']), 'label': 'battery'},
                'secondary2': {'value': int(latest_garmin['stressPercentage']), 'label': 'stress'}
            },
            'sleep': {
                'primary': {'value': int(latest_garmin['sleep_score']), 'label': 'sleep'},
                'secondary1': {'value': sleep_time, 'label': 'hrs in bed'},
                'secondary2': {'value': '-', 'label': 'behavior'} # Still placeholder
            },
            'running': {
                'primary': {'value': int(latest_garmin['TSB']), 'label': 'TSB'},
                'secondary1': {'value': f"{last_7d_distance:.1f}", 'label': 'km L7d'},
                'secondary2': {'value': f"{int(last_7d_altitude)}", 'label': 'm gain L7d'}
            },
            'strength': {
                'primary': {'value': 3, 'label': 'since lifting'},    # Placeholder
                'secondary1': {'value': '-', 'label': 'pullups'}, # Placeholder
                'secondary2': {'value': '-', 'label': 'min'}      # Placeholder
            },
            'glucose': {
                'primary': {'value': 95, 'label': 'mg/dL'},        # Placeholder
                'secondary1': {'value': '-', 'label': 'fasting'}, # Placeholder
                'secondary2': {'value': '-', 'label': 'mean'}    # Placeholder
            }
        }
        return metrics
    except Exception as e:
        print(f"Error getting metrics: {e}")  # For debugging
        return None 