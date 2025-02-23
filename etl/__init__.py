# This file makes the etl directory a Python package
# It can be empty, but we can also expose specific functions here for cleaner imports

from .garmin_etl import init_garmin, get_garmin_data, run_garmin_etl
from .whoop_etl import init_whoop, get_sleep_recovery_data, run_whoop_etl
from .mfp_etl import init_mfp, get_meal_data, get_meal_daily, run_mfp_etl
from .fitbit_etl import run_fitbit_etl

__all__ = [
    'init_garmin', 
    'get_garmin_data',
    'run_garmin_etl',
    'init_whoop',
    'get_sleep_recovery_data',
    'run_whoop_etl',
    'init_mfp',
    'get_meal_data',
    'get_meal_daily',
    'run_mfp_etl',
    'run_fitbit_etl'
] 