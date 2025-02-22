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
            diary = client.get_date(current_date.year, current_date.month, current_date.day)
            for meal in diary.meals:
                for food in meal.entries:
                    row_data = {
                        'date': current_date.strftime('%Y-%m-%d'),
                        'meal': meal.name,
                        'food': food.name,
                        'quant': food.quantity,
                        'calories': food.nutrition_information['calories'],
                        'carbs': food.nutrition_information['carbohydrates'],
                        'fat': food.nutrition_information['fat'],
                        'protein': food.nutrition_information['protein'],
                        'sodium': food.nutrition_information['sodium'],
                        'sugar': food.nutrition_information['sugar']
                    }
                    writer.writerow(row_data)
            logger.info(f'{filename}: Data per meal obtained and (re-)written for {current_date.strftime("%Y-%m-%d")}')
            current_date += datetime.timedelta(days=1)

# Function to get daily summary data from MyFitnessPal and append to a CSV file
def get_meal_daily(client, filename):
    end_date = datetime.datetime.now().date() - datetime.timedelta(days=1)
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
            diary = client.get_date(current_date.year, current_date.month, current_date.day)
            exercises = diary.exercises[0].entries if diary.exercises else []
            calories_burned = sum(entry.get_as_dict()['nutrition_information'].get('calories burned', 0) for entry in exercises)
            calories_meal = {name: diary.meals[i].totals['calories'] if len(diary.meals[i].entries) else 0 for i, name in enumerate(['breakfast', 'lunch', 'dinner', 'snacks'])}
            writer.writerow({
                'date': current_date.strftime('%Y-%m-%d'),
                'calories_burned': calories_burned,
                'carbs': diary.totals['carbohydrates'],
                'fat': diary.totals['fat'],
                'protein': diary.totals['protein'],
                'sodium': diary.totals['sodium'],
                'sugar': diary.totals['sugar'],
                'calories_consumed': diary.totals['calories'],
                'calories_goal': diary.goals['calories'],
                'calories_net': diary.totals['calories'] - diary.goals['calories'],
                'calories_consumed_breakfast': calories_meal['breakfast'],
                'calories_consumed_lunch': calories_meal['lunch'],
                'calories_consumed_dinner': calories_meal['dinner'],
                'calories_consumed_snacks': calories_meal['snacks']
            })
            logger.info(f'{filename}: Data per day obtained and (re-)written for {current_date.strftime("%Y-%m-%d")}')
            current_date += datetime.timedelta(days=1)

def run_mfp_etl():
    """Execute MyFitnessPal ETL process."""
    client = init_mfp()
    get_meal_data(client, config.MFP_MEALS_FILE)
    get_meal_daily(client, config.MFP_DAILY_FILE)
    logger.info('MyFitnessPal data updated successfully')

if __name__ == "__main__":
    run_mfp_etl()
