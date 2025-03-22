from dotenv import load_dotenv
import myfitnesspal
import datetime
import csv
import os
import logging
import json
import pandas as pd
from http.cookiejar import CookieJar, Cookie
from etl import config  # Use project config


logger = logging.getLogger(__name__)

# Function to get the most recent date from a CSV file
def get_most_recent_date(filename):
    if filename.endswith('.csv'):
        try:
            # Using pandas to handle headers and date parsing more robustly
            df = pd.read_csv(filename, nrows=0)  # Read only headers
            first_column_name = df.columns[0]
            if first_column_name.lower() == 'date':
                # If first column is indeed called 'date', read dates from this column
                df = pd.read_csv(filename, usecols=[first_column_name])
                df[first_column_name] = pd.to_datetime(df[first_column_name], errors='coerce')
                most_recent_date = df[first_column_name].dropna().max()
                return most_recent_date.date() if most_recent_date else None
            else:
                # If not, use the file's last modification time
                return datetime.datetime.fromtimestamp(os.path.getmtime(filename)).date()
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return None
    else:
        # For non-csv files or if any other error occurred
        try:
            return datetime.datetime.fromtimestamp(os.path.getmtime(filename)).date()
        except OSError:
            return None

# Function to delete all data from that date onwards
def delete_data_from_date(filename, date):
    """Delete all data from the given date onwards in a CSV file.
    
    Args:
        filename: Path to CSV file
        date: datetime.date object or string in YYYY-MM-DD format
    """
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
    
    date_str = date.strftime('%Y-%m-%d')
    logger.info(f"Deleting data from {date_str} onwards in {filename}")
    
    temp_filename = filename + '.tmp'
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as csvfile, \
             open(temp_filename, 'w', newline='', encoding='utf-8') as tmpfile:
            reader = csv.reader(csvfile)
            writer = csv.writer(tmpfile)
            
            # Write header
            headers = next(reader)
            writer.writerow(headers)
            
            # Write rows before the date
            for row in reader:
                if row[0] < date_str:
                    writer.writerow(row)
        
        os.replace(temp_filename, filename)
        logger.info(f"Successfully deleted data from {date_str} onwards")
        
    except Exception as e:
        logger.error(f"Error deleting data: {str(e)}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def init_mfp():
    """Initialize and return a MyFitnessPal client."""
    load_dotenv("Credentials.env")
    cookies_str = os.getenv("MFP_COOKIES")
    if not cookies_str:
        raise ValueError("MFP_COOKIES not found in environment variables")
    
    # Convertir la cadena JSON en un diccionario
    cookies_dict = json.loads(cookies_str)

    # Crear un CookieJar y añadir las cookies
    cookiejar = CookieJar()
    for name, value in cookies_dict.items():
        cookie = Cookie(
            version=0,
            name=name,
            value=value,
            port=None,
            port_specified=False,
            domain='.myfitnesspal.com',
            domain_specified=True,
            domain_initial_dot=True,
            path='/',
            path_specified=True,
            secure=True,
            expires=None,
            discard=True,
            comment=None,
            comment_url=None,
            rest={'HttpOnly': None},
            rfc2109=False
        )
        cookiejar.set_cookie(cookie)
    
    # Initialize client without cookies first
    client = myfitnesspal.Client(cookiejar=cookiejar)
    
    return client

# Function to get meal data from MyFitnessPal and append to a CSV file
def get_meal_data(client, filename):
    end_date = datetime.datetime.now().date()
    most_recent_date = get_most_recent_date(filename)

    # Handle first-time run or no data case
    if most_recent_date is None:
        start_date = config.DATA_START_DATE
        logger.info(f"No existing data found. Starting from {start_date}")
    else:
        # Delete the last day to rewrite it
        delete_data_from_date(filename, most_recent_date)
        start_date = most_recent_date

    fieldnames = ['date', 'meal', 'food', 'quant', 'calories', 'carbs', 'fat', 'protein', 'sodium', 'sugar']
    mode = 'a' if os.path.exists(filename) else 'w'
    with open(filename, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if mode == 'w':
            writer.writeheader()
        current_date = start_date
        while current_date <= end_date:
            try:
                diary = client.get_date(current_date.year, current_date.month, current_date.day)
                meal_entries_found = False
                
                for meal in diary.meals:
                    for food in meal.entries:
                        try:
                            # Get nutrition information safely with defaults
                            nutrition = food.nutrition_information or {}
                            
                            row_data = {
                                'date': current_date.strftime('%Y-%m-%d'),
                                'meal': meal.name,
                                'food': food.name,
                                'quant': food.quantity,
                                'calories': nutrition.get('calories', 0),
                                'carbs': nutrition.get('carbohydrates', 0),
                                'fat': nutrition.get('fat', 0),
                                'protein': nutrition.get('protein', 0),
                                'sodium': nutrition.get('sodium', 0),
                                'sugar': nutrition.get('sugar', 0)
                            }
                            writer.writerow(row_data)
                            meal_entries_found = True
                        except Exception as food_error:
                            logger.warning(f"Error processing food entry for {current_date}: {str(food_error)}")
                            continue
                
                if meal_entries_found:
                    logger.info(f'{filename}: Data per meal obtained and (re-)written for {current_date.strftime("%Y-%m-%d")}')
                else:
                    logger.info(f'{filename}: No meal entries found for {current_date.strftime("%Y-%m-%d")}')
            except Exception as e:
                logger.warning(f"Error processing diary for {current_date}: {str(e)}")
            
            current_date += datetime.timedelta(days=1)

# Function to get daily summary data from MyFitnessPal and append to a CSV file
def get_meal_daily(client, filename):
    end_date = datetime.datetime.now().date()
    most_recent_date = get_most_recent_date(filename)

    # Handle first-time run or no data case
    if most_recent_date is None:
        start_date = config.DATA_START_DATE
        logger.info(f"No existing data found. Starting from {start_date}")
    else:
        # Delete the last day to rewrite it
        delete_data_from_date(filename, most_recent_date)
        start_date = most_recent_date

    fieldnames = ['date', 'calories_burned', 'carbs', 'fat', 'protein', 'sodium', 'sugar', 'calories_consumed', 'calories_goal', 'calories_net',
                  'calories_consumed_breakfast', 'calories_consumed_lunch', 'calories_consumed_dinner', 'calories_consumed_snacks']
    mode = 'a' if os.path.exists(filename) else 'w'
    with open(filename, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if mode == 'w':
            writer.writeheader()
        current_date = start_date
        while current_date <= end_date:
            try:
                diary = client.get_date(current_date.year, current_date.month, current_date.day)
                
                # Handle exercise data safely with defaults
                exercises = diary.exercises[0].entries if diary.exercises else []
                calories_burned = sum(entry.get_as_dict()['nutrition_information'].get('calories burned', 0) for entry in exercises)
                
                # Default to empty dict if totals is empty or non-existent
                totals = getattr(diary, 'totals', {}) or {}
                
                # Check if meals exist and have totals; default to 0 if not
                calories_meal = {}
                for i, name in enumerate(['breakfast', 'lunch', 'dinner', 'snacks']):
                    if i < len(diary.meals) and hasattr(diary.meals[i], 'totals') and diary.meals[i].totals:
                        calories_meal[name] = diary.meals[i].totals.get('calories', 0)
                    else:
                        calories_meal[name] = 0
                
                # Use safe get() method with default value for all nutritional data
                row_data = {
                    'date': current_date.strftime('%Y-%m-%d'),
                    'calories_burned': calories_burned,
                    'carbs': totals.get('carbohydrates', 0),
                    'fat': totals.get('fat', 0),
                    'protein': totals.get('protein', 0),
                    'sodium': totals.get('sodium', 0),
                    'sugar': totals.get('sugar', 0),
                    'calories_consumed': totals.get('calories', 0),
                    'calories_goal': getattr(diary, 'goals', {}).get('calories', 0),
                    'calories_net': totals.get('calories', 0) - getattr(diary, 'goals', {}).get('calories', 0),
                    'calories_consumed_breakfast': calories_meal.get('breakfast', 0),
                    'calories_consumed_lunch': calories_meal.get('lunch', 0),
                    'calories_consumed_dinner': calories_meal.get('dinner', 0),
                    'calories_consumed_snacks': calories_meal.get('snacks', 0)
                }
                
                writer.writerow(row_data)
                logger.info(f'{filename}: Data per day obtained and (re-)written for {current_date.strftime("%Y-%m-%d")}')
            except Exception as e:
                # Log the error but still add a row with zeros for this date
                logger.warning(f"Error processing data for {current_date}: {str(e)}. Adding zeros instead.")
                
                # Create a row with all zeros except for the date
                default_row = {key: 0 for key in fieldnames}
                default_row['date'] = current_date.strftime('%Y-%m-%d')
                writer.writerow(default_row)
                
            current_date += datetime.timedelta(days=1)

def run_mfp_etl():
    """Execute MyFitnessPal ETL process."""
    client = init_mfp()
    get_meal_data(client, config.MFP_MEALS_FILE)
    get_meal_daily(client, config.MFP_DAILY_FILE)
    logger.info('MyFitnessPal data updated successfully')

def run_mfp_daily_only():
    """Execute MyFitnessPal ETL process for daily data only (faster)."""
    client = init_mfp()
    get_meal_daily(client, config.MFP_DAILY_FILE)
    logger.info('MyFitnessPal daily data updated successfully')

if __name__ == "__main__":
    run_mfp_etl()
