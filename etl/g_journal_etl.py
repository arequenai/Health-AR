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
    
    # Load credentials from environment variables
    try:
        credentials_dict = {
            "type": "service_account",
            "project_id": os.getenv("GOOGLE_PROJECT_ID"),
            "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID"),
            "private_key": os.getenv("GOOGLE_PRIVATE_KEY").replace('\\n', '\n'),
            "client_email": os.getenv("GOOGLE_CLIENT_EMAIL"),
            "client_id": os.getenv("GOOGLE_CLIENT_ID"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": os.getenv("GOOGLE_CLIENT_CERT_URL")
        }
        
        # Create credentials from dictionary
        credentials = ServiceAccountCredentials.from_json_keyfile_dict(credentials_dict, scope)
        client = gspread.authorize(credentials)
        logger.info("Successfully connected to Google Sheets")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Google Sheets: {str(e)}")
        raise

def get_journal_data(client, spreadsheet_url, filename):
    """Get journal data from Google Sheets and save it to a CSV file.
    
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
        
        # Check if we have a timestamp column
        timestamp_col = next((col for col in df.columns if 'timestamp' in col.lower()), None)
        
        # Convert all column names to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Normalize column names to match required columns
        required_columns = ['date', 'nutrition', 'bed_behavior', 'alcohol', 'bed_full', 'earplugs', 'stretch', 'sick']
        
        # Create a map of lowercase original columns to original columns for easier lookup
        orig_col_map = {col.lower(): col for col in df.columns}
        
        # Create mapping of source columns to target columns
        column_mappings = {
            'date': ['date'],
            'nutrition': ['nutrition'],
            'bed_behavior': ['bed behavior', 'bed_behavior', 'behavior'],
            'alcohol': ['alcohol', 'drink', 'drinks'],
            'bed_full': ['full bed', 'bed_full', 'fullbed'],
            'earplugs': ['earplugs', 'ear plugs', 'ear-plugs'],
            'stretch': ['stretch', 'stretching'],
            'sick': ['sick', 'illness', 'sickness']
        }
        
        # Rename columns and initialize new DataFrame
        df_renamed = pd.DataFrame()
        
        # First initialize all required columns with None
        for target_col in required_columns:
            df_renamed[target_col] = None
        
        # Now fill in any matching columns
        for target_col, source_patterns in column_mappings.items():
            for pattern in source_patterns:
                matching_cols = [col for col in orig_col_map.keys() if pattern in col]
                if matching_cols:
                    # Use the first matching column
                    source_col = orig_col_map[matching_cols[0]]
                    df_renamed[target_col] = df[source_col]
                    logger.info(f"Mapped '{source_col}' to '{target_col}'")
                    break  # Found a match for this target column
        
        # Handle timestamp to date conversions
        if timestamp_col:
            # Convert timestamp to datetime
            timestamp_col_lower = timestamp_col.lower()
            if timestamp_col_lower in df.columns:
                df['timestamp'] = pd.to_datetime(df[timestamp_col_lower], errors='coerce')
            else:
                # Try the original column name if lowercase version not found
                df['timestamp'] = pd.to_datetime(df[timestamp_col], errors='coerce')
                
            # If date is empty, use day before timestamp
            for i, row in df.iterrows():
                # Check if date is empty or invalid in the renamed dataframe
                if i < len(df_renamed) and (pd.isna(df_renamed.at[i, 'date']) or df_renamed.at[i, 'date'] == '' or df_renamed.at[i, 'date'] is None):
                    if not pd.isna(row['timestamp']):
                        # Use day before the timestamp
                        prev_day = row['timestamp'] - pd.Timedelta(days=1)
                        df_renamed.at[i, 'date'] = prev_day.strftime('%Y-%m-%d')
        
        # Convert any date values to a consistent format
        if 'date' in df_renamed.columns:
            # Convert to datetime first to handle various formats
            df_renamed['date'] = pd.to_datetime(df_renamed['date'], errors='coerce')
            
            # Drop rows with invalid dates
            df_renamed = df_renamed.dropna(subset=['date'])
            
            # Format to YYYY-MM-DD
            df_renamed['date'] = df_renamed['date'].dt.strftime('%Y-%m-%d')
        
        # Convert boolean-like values to standard format
        boolean_columns = ['alcohol', 'bed_full', 'earplugs', 'stretch', 'sick']
        for col in boolean_columns:
            if col in df_renamed.columns:
                # Map various true/false values to YES/NO
                df_renamed[col] = df_renamed[col].apply(lambda x: 
                    'YES' if pd.notna(x) and str(x).lower() in ['true', 'yes', 'y', '1', 'checked', 'check', 'si', 'sí'] 
                    else 'NO' if pd.notna(x) and str(x).lower() in ['false', 'no', 'n', '0', 'unchecked', ''] 
                    else x
                )
        
        # Sort by date
        df_renamed = df_renamed.sort_values('date')
        
        # Keep only the last entry per date
        df_renamed = df_renamed.drop_duplicates(subset=['date'], keep='last')
        
        # Save to CSV
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df_renamed.to_csv(filename, index=False)
        logger.info(f"Journal data from Google Sheets saved to {filename}")
        logger.info(f"Processed {len(df_renamed)} journal entries with columns: {', '.join(df_renamed.columns)}")
        
    except Exception as e:
        logger.error(f"Error fetching journal data from Google Sheets: {str(e)}")
        raise

def run_g_journal_etl():
    """Execute Google Sheets Journal ETL process."""
    # Define constants
    GSHEET_JOURNAL_URL = "https://docs.google.com/spreadsheets/d/1aepZFn4ciTdVtC9Kd_ykk7VTD6LSsyWBuD0Z-sVGLcs/edit?gid=1077059296#gid=1077059296"
    JOURNAL_FILE = "data/g_journal.csv"
    
    # Add to config if not already there
    if not hasattr(config, 'GSHEET_JOURNAL_URL'):
        config.GSHEET_JOURNAL_URL = GSHEET_JOURNAL_URL
    
    if not hasattr(config, 'JOURNAL_FILE'):
        config.JOURNAL_FILE = JOURNAL_FILE
    
    try:
        client = init_gsheets()
        get_journal_data(client, config.GSHEET_JOURNAL_URL, config.JOURNAL_FILE)
        logger.info('Google Sheets journal data updated successfully')
    except PermissionError:
        client_email = os.getenv("GOOGLE_CLIENT_EMAIL")
        logger.error(f"Permission denied: You need to share the Google Sheet with the service account email: {client_email}")
        logger.error(f"Please go to your Google Spreadsheet, click the 'Share' button, and add {client_email} with Viewer access.")
        print(f"\n*******************************************")
        print(f"ERROR: Permission denied accessing Google Sheet")
        print(f"You need to share the spreadsheet with: {client_email}")
        print(f"Please go to the Google Sheet at:")
        print(f"{GSHEET_JOURNAL_URL}")
        print(f"Click 'Share' in the top right, and add the email above with Viewer access.")
        print(f"*******************************************\n")
    except Exception as e:
        logger.error(f"Error in journal ETL: {str(e)}")
        raise

if __name__ == "__main__":
    run_g_journal_etl() 