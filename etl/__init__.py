# This file makes the etl directory a Python package
# It can be empty, but we can also expose specific functions here for cleaner imports

from .garmin_etl import init_garmin, get_garmin_data
from .whoop_etl import init_whoop, get_sleep_recovery_data

__all__ = [
    'init_garmin', 
    'get_garmin_data',
    'init_whoop',
    'get_sleep_recovery_data'
] 