from dotenv import load_dotenv
load_dotenv("Credentials.env")

import os
import json
import datetime
import traceback
import sys
from etl.garmin_etl import init_garmin
import pandas as pd

# Enable more verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

print("=== GARMIN API TEST SCRIPT ===")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

# Initialize Garmin client
EMAIL = os.getenv("USERNAME_G")
PASSWORD = os.getenv("PASSWORD_G")

print(f"Credentials loaded - Email: {'FOUND' if EMAIL else 'NOT FOUND'}, Password: {'FOUND' if PASSWORD else 'NOT FOUND'}")

try:
    print("\nInitializing Garmin client...")
    client = init_garmin(EMAIL, PASSWORD)
    print("Garmin client initialized successfully.")
except Exception as e:
    print(f"Error initializing Garmin client: {e}")
    traceback.print_exc()
    sys.exit(1)

# Set the date to today
test_date = datetime.date.today()
formatted_date = test_date.strftime('%Y-%m-%d')

print(f"\nTesting with date: {formatted_date}")

# Helper function to safely print API responses
def safe_print_response(name, response):
    try:
        print(f"\n{name} response type: {type(response)}")
        
        if response is None:
            print(f"{name} response is None")
            return
            
        if isinstance(response, dict):
            print(f"{name} keys: {list(response.keys())}")
            
            # Sample of a few keys
            for key in list(response.keys())[:5]:
                print(f"  {key}: {type(response[key])}")
                
        elif isinstance(response, list):
            print(f"{name} is a list with {len(response)} items")
            
            if len(response) > 0:
                sample = response[0]
                print(f"First item type: {type(sample)}")
                
                if isinstance(sample, dict):
                    print(f"First item keys: {list(sample.keys())}")
                    
                    # Print first few key-value pairs
                    for key in list(sample.keys())[:5]:
                        print(f"  {key}: {type(sample[key])} = {sample[key]}")
    except Exception as e:
        print(f"Error printing {name} response: {e}")

# Test different API methods
methods_to_try = [
    ("get_body_battery", lambda: client.get_body_battery(formatted_date)),
    ("get_body_composition", lambda: client.get_body_composition(formatted_date)),
    ("get_user_summary", lambda: client.get_user_summary(formatted_date)),
    ("get_stats", lambda: client.get_stats(test_date)),
    ("get_sleep_data", lambda: client.get_sleep_data(formatted_date)),
    ("get_stress_data", lambda: client.get_stress_data(formatted_date)),
    ("get_heart_rates", lambda: client.get_heart_rates(formatted_date)),
    ("get_rhr_day", lambda: client.get_rhr_day(formatted_date)),
]

for method_name, method_call in methods_to_try:
    print(f"\n\n=== Testing {method_name} ===")
    try:
        response = method_call()
        safe_print_response(method_name, response)
        
        # Save to file for inspection
        try:
            with open(f"test_{method_name}.json", "w") as f:
                json.dump(response, f, indent=2, default=str)
            print(f"Saved response to test_{method_name}.json")
        except Exception as e:
            print(f"Error saving response to file: {e}")
            
    except Exception as e:
        print(f"Error calling {method_name}: {e}")
        traceback.print_exc()

print("\nTest completed!") 