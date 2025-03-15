import os
import logging
import pandas as pd
import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
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
            # Use pandas for more robust CSV handling
            df = pd.read_csv(csvfile)
            if 'date' in df.columns:
                df = df[pd.to_datetime(df['date']) < pd.to_datetime(date_str)]
                df.to_csv(tmpfile, index=False)
            else:
                # If no date column, write everything back
                df.to_csv(tmpfile, index=False)
        
        os.replace(temp_filename, filename)
        logger.info(f"Successfully deleted data from {date_str} onwards")
        
    except Exception as e:
        logger.error(f"Error deleting data: {str(e)}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def init_gsheets():
    """Initialize and return a Google Sheets client."""
    # Define the scope for Google Sheets API
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    
    # Get the credentials file path
    credentials_path = 'gsheets_key.json'
    
    # Authenticate with Google Sheets
    try:
        credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
        client = gspread.authorize(credentials)
        logger.info("Successfully connected to Google Sheets")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Google Sheets: {str(e)}")
        raise

def get_sheet_data(client, spreadsheet_url, filename):
    """Get data from Google Sheets and save it to a CSV file.
    
    Args:
        client: Google Sheets client
        spreadsheet_url: URL of the Google Sheets document
        filename: Path to save the CSV file
    """
    # Extract spreadsheet key from URL
    spreadsheet_id = spreadsheet_url.split('/d/')[1].split('/')[0]
    
    # Open the spreadsheet
    try:
        spreadsheet = client.open_by_key(spreadsheet_id)
        # Get the first worksheet
        worksheet = spreadsheet.get_worksheet(0)
        
        # Get all values from the worksheet
        data = worksheet.get_all_values()
        
        # Create a DataFrame
        df = pd.DataFrame(data[1:], columns=data[0])
        
        # Convert all column names to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # If 'date' column exists, convert to consistent date format
        if 'date' in df.columns:
            # Convert to datetime first to handle various formats
            df['date'] = pd.to_datetime(df['date'])
            # Format to YYYY-MM-DD
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        # Calculate 1RM if weight and reps columns exist
        if 'weight' in df.columns and 'reps' in df.columns:
            # Convert columns to numeric, with errors coerced to NaN
            df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
            df['reps'] = pd.to_numeric(df['reps'], errors='coerce')
            
            # Calculate 1RM using formula: Weight × (1 + (Reps ÷ 30))
            # This is a conservative formula for calculating 1RM
            df['1rm'] = df['weight'] * (1 + (df['reps'] / 30))
            
            # Round to 2 decimal places for readability
            df['1rm'] = df['1rm'].round(2)
            
            logger.info("Added calculated 1RM column based on weight and reps")
        
        # Save to CSV
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        logger.info(f"Data from Google Sheets saved to {filename}")
        
    except Exception as e:
        logger.error(f"Error fetching data from Google Sheets: {str(e)}")
        raise

def run_gsheets_etl():
    """Execute Google Sheets ETL process."""
    client = init_gsheets()
    get_sheet_data(client, config.GSHEETS_URL, config.GSHEETS_FILE)
    logger.info('Google Sheets data updated successfully')

if __name__ == "__main__":
    run_gsheets_etl() 