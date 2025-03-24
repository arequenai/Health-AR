METRIC_THRESHOLDS = {
    'nutrition': {'5': -2000, '4': -1500, '3': -600, '2': 0, '1': 150},
    'recovery': {'5': 95, '4': 90, '3': 80, '2': 60, '1': 50},
    'sleep': {'5': 85, '4': 75, '3': 60, '2': 50, '1': 0},
    'running': {'5': -20, '4': -10, '3': 0, '2': 10, '1': 20},
    'strength': {'5': 1, '4': 2, '3': 3, '2': 4, '1': 5},
    'glucose': {'5': 90, '4': 100, '3': 110, '2': 120, '1': 140}
}

# Thresholds for secondary metrics
# Format: 'metric_name.label': {'good_above'/'good_below'/'bad_above'/'bad_below': value}
# - Use good_above when values above the threshold are good
# - Use good_below when values below the threshold are good
# - Use bad_above when values above the threshold are bad
# - Use bad_below when values below the threshold are bad
SECONDARY_METRIC_THRESHOLDS = {
    # Nutrition metrics
    'nutrition.net above L7d': {'bad_above': 700, 'good_below': 0},
    'nutrition.protein g': {'good_above': 120, 'bad_below': 70},
    
    # Glucose metrics
    'glucose.fasting': {'good_below': 90, 'bad_above': 110},
    'glucose.mean day': {'good_below': 100, 'bad_above': 120},
    
    # Recovery metrics
    'recovery.battery now': {'good_above': 70, 'bad_below': 25},
    'recovery.stress': {'good_below': 25, 'bad_above': 40},
    
    # Sleep metrics
    'sleep.hrs in bed': {'good_above': 7.5, 'bad_below': 6},
    'sleep.behavior': {'good_above': 100, 'bad_below': 70},
    
    # Running metrics
    'running.km run L7d': {'good_above': 40, 'bad_below': 20},
    'running.m gain L7d': {'good_above': 800},
    
    # Strength metrics
    'strength.pullups': {'good_above': 10, 'bad_below': 5},
    'strength.sets L7d': {'good_above': 30, 'bad_below': 10}
} 