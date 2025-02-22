from typing import Dict, Any, Optional
import pandas as pd
from etl import config

def get_metrics() -> Optional[Dict[str, Dict[str, Dict[str, Any]]]]:
    """Get all metrics from various sources."""
    try:
        metrics = {
            'nutrition': {
                'primary': {'value': 2100, 'label': 'kcal today'},  # From MFP
                'secondary1': {'value': '75.5', 'label': 'kg'},     # Placeholder
                'secondary2': {'value': '120', 'label': 'g protein'} # From MFP
            },
            'recovery': {
                'primary': {'value': 85, 'label': 'recovery'},      # From Whoop
                'secondary1': {'value': '65', 'label': 'battery'},  # From Whoop
                'secondary2': {'value': '45', 'label': 'stress'}    # From Whoop
            },
            'sleep': {
                'primary': {'value': 88, 'label': 'quality'},       # From Whoop
                'secondary1': {'value': '7:30', 'label': 'in bed'}, # From Whoop
                'secondary2': {'value': 'Good', 'label': 'behavior'} # From Whoop Journal
            },
            'running': {
                'primary': {'value': -5, 'label': 'TSB'},          # Calculated
                'secondary1': {'value': '45', 'label': 'CTL'},     # Calculated
                'secondary2': {'value': '35', 'label': 'km L7D'}   # From Garmin
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
        return None 