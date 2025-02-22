from typing import Dict, Any, Optional
import pandas as pd
from etl import config

def get_metrics() -> Optional[Dict[str, Dict[str, Dict[str, Any]]]]:
    """Get all metrics from various sources."""
    try:
        # Read MFP daily data
        df_mfp = pd.read_csv(config.MFP_DAILY_FILE)
        latest_mfp = df_mfp.iloc[-1]

        # Read Whoop data
        df_whoop = pd.read_csv(config.WHOOP_SLEEP_RECOVERY_FILE)
        latest_whoop = df_whoop.iloc[-1]

        # Read Garmin data
        df_garmin = pd.read_csv(config.GARMIN_DAILY_FILE)
        latest_garmin = df_garmin.iloc[-1]
        # Calculate last 7 days distance in km
        last_7d_distance = df_garmin.tail(7)['totalDistanceMeters'].sum() / 1000

        # Format sleep time
        sleep_hours = latest_garmin['sleepingSeconds'] / 3600  # Convert seconds to hours
        sleep_hrs = int(sleep_hours)
        sleep_mins = int((sleep_hours - sleep_hrs) * 60)
        sleep_time = f"{sleep_hrs}:{sleep_mins:02d}"

        # Handle calories net label and value
        calories_net = int(latest_mfp['calories_net'])
        if calories_net > 0:
            cal_label = 'kcal over'
            cal_value = calories_net
        else:
            cal_label = 'kcal rem'
            cal_value = abs(calories_net)

        metrics = {
            'nutrition': {
                'primary': {'value': cal_value, 'label': cal_label, 'color_value': calories_net},
                'secondary1': {'value': '75.5', 'label': 'kg (placeholder)'},     # Placeholder
                'secondary2': {'value': int(latest_mfp['protein']), 'label': 'g protein eaten'}
            },
            'recovery': {
                'primary': {'value': int(latest_whoop['recovery_score']), 'label': 'recovery'},
                'secondary1': {'value': int(latest_garmin['bodyBatteryMostRecentValue']), 'label': 'battery'},
                'secondary2': {'value': int(latest_garmin['stressPercentage']), 'label': 'stress'}
            },
            'sleep': {
                'primary': {'value': int(latest_garmin['bodyBatteryDuringSleep']), 'label': 'quality'},
                'secondary1': {'value': sleep_time, 'label': 'hrs in bed'},
                'secondary2': {'value': 'Good', 'label': 'behavior'} # Still placeholder
            },
            'running': {
                'primary': {'value': int(latest_garmin['TSB']), 'label': 'TSB'},
                'secondary1': {'value': int(latest_garmin['CTL']), 'label': 'CTL'}, # Not a placeholder, but CTL calculation is wrong
                'secondary2': {'value': f"{last_7d_distance:.1f}", 'label': 'km L7D'}
            },
            'strength': {
                'primary': {'value': 3, 'label': 'days since'},    # Placeholder
                'secondary1': {'value': '12', 'label': 'pullups'}, # Placeholder
                'secondary2': {'value': '45', 'label': 'min'}      # Placeholder
            },
            'glucose': {
                'primary': {'value': 95, 'label': 'mg/dL'},        # Placeholder
                'secondary1': {'value': '85', 'label': 'fasting'}, # Placeholder
                'secondary2': {'value': '105', 'label': 'mean'}    # Placeholder
            }
        }
        return metrics
    except Exception as e:
        print(f"Error getting metrics: {e}")  # For debugging
        return None 