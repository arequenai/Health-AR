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

        # Handle calories net label and value
        calories_net = int(latest_mfp['calories_net'])
        if calories_net > 0:
            cal_label = 'kcal over'
            cal_value = calories_net
        else:
            cal_label = 'kcal remaining'
            cal_value = abs(calories_net)

        metrics = {
            'nutrition': {
                'primary': {'value': cal_value, 'label': cal_label, 'color_value': calories_net},
                'secondary1': {'value': '75.5', 'label': 'kg (placeholder)'},     # Placeholder
                'secondary2': {'value': int(latest_mfp['protein']), 'label': 'g protein eaten'}
            },
            'recovery': {
                'primary': {'value': int(latest_whoop['recovery_score']), 'label': 'recovery'},
                'secondary1': {'value': '65', 'label': 'battery'},  # Still placeholder
                'secondary2': {'value': '45', 'label': 'stress'}    # Still placeholder
            },
            'sleep': {
                'primary': {'value': 88, 'label': 'quality'},       # From Whoop
                'secondary1': {'value': '7:30', 'label': 'in bed'}, # From Whoop
                'secondary2': {'value': 'Good', 'label': 'behavior'} # From Whoop Journal
            },
            'running': {
                'primary': {'value': -5, 'label': 'TSB'},          # Still placeholder
                'secondary1': {'value': '45', 'label': 'CTL'},     # Still placeholder
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