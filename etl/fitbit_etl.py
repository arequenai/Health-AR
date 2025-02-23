import os
import json
import requests
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import logging
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import time
import base64
import fitbit
from etl import config  # Add this import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fitbit API configuration
BASE_URL = "https://api.fitbit.com"
AUTH_URL = "https://www.fitbit.com/oauth2/authorize"
TOKEN_URL = "https://api.fitbit.com/oauth2/token"
REDIRECT_URI = "http://localhost:8080/"
SCOPE = "weight profile"

class TokenHandler(BaseHTTPRequestHandler):
    """Handle OAuth callback and store the authorization code."""
    def do_GET(self):
        """Process the callback from Fitbit OAuth."""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        # Parse the authorization code from the callback URL
        query_components = parse_qs(urlparse(self.path).query)
        
        if 'code' in query_components:
            self.server.auth_code = query_components['code'][0]
            self.wfile.write(b"Authorization successful! You can close this window.")
            # Signal the server to shut down
            self.server.should_stop = True
        else:
            self.wfile.write(b"Authorization failed! Please try again.")
        
    def log_message(self, format, *args):
        """Suppress logging of HTTP requests."""
        return

def get_auth_code(client_id):
    """Get authorization code through OAuth flow."""
    # Construct authorization URL
    auth_params = {
        'response_type': 'code',
        'client_id': client_id,
        'redirect_uri': REDIRECT_URI,
        'scope': SCOPE
    }
    
    auth_url = f"{AUTH_URL}?{'&'.join(f'{k}={v}' for k, v in auth_params.items())}"
    
    # Start local server to receive callback
    server = HTTPServer(('localhost', 8080), TokenHandler)
    server.auth_code = None
    server.should_stop = False
    server.timeout = 1  # Set socket timeout to 1 second
    
    # Open browser for authorization
    webbrowser.open(auth_url)
    logger.info("Opened browser for Fitbit authorization. Please authorize the app...")
    
    # Wait for callback with timeout
    start_time = time.time()
    timeout = 60  # 60 seconds timeout
    
    try:
        while not server.should_stop and time.time() - start_time < timeout:
            try:
                server.handle_request()
            except Exception as e:
                logger.error(f"Error handling request: {str(e)}")
                break
    finally:
        server.server_close()
        logger.info("Authorization server closed")
    
    if not server.auth_code:
        raise TimeoutError("Authorization timed out or failed. Please try again.")
    
    return server.auth_code

def get_tokens(client_id, client_secret, auth_code=None, refresh_token=None):
    """Get access and refresh tokens using authorization code or refresh token."""
    # Create basic auth header
    auth_string = f"{client_id}:{client_secret}"
    auth_bytes = auth_string.encode('ascii')
    base64_auth = base64.b64encode(auth_bytes).decode('ascii')
    
    headers = {
        'Authorization': f'Basic {base64_auth}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    if auth_code:
        # Initial token request
        data = {
            'code': auth_code,
            'grant_type': 'authorization_code',
            'redirect_uri': REDIRECT_URI,
            'client_id': client_id
        }
    elif refresh_token:
        # Token refresh
        data = {
            'grant_type': 'refresh_token',
            'refresh_token': refresh_token,
            'client_id': client_id
        }
    else:
        raise ValueError("Either auth_code or refresh_token must be provided")
    
    response = requests.post(TOKEN_URL, headers=headers, data=data)
    response.raise_for_status()
    
    return response.json()

def init_fitbit():
    """Initialize Fitbit client with OAuth2 authentication."""
    # Load credentials from environment
    client_id = os.getenv("FITBIT_CLIENT_ID")
    client_secret = os.getenv("FITBIT_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        raise ValueError("Fitbit credentials not found in environment variables")
    
    # Check for existing tokens
    token_file = "fitbit_tokens.json"
    try:
        if os.path.exists(token_file):
            with open(token_file, 'r') as f:
                tokens = json.load(f)
            
            # Try to refresh token
            try:
                logger.info("Refreshing Fitbit token...")
                new_tokens = get_tokens(client_id, client_secret, refresh_token=tokens['refresh_token'])
                tokens.update(new_tokens)
                logger.info("Token refreshed successfully")
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in [401, 403]:
                    logger.warning("Token refresh failed, starting new OAuth flow")
                    auth_code = get_auth_code(client_id)
                    tokens = get_tokens(client_id, client_secret, auth_code=auth_code)
                else:
                    raise
        else:
            logger.info("No existing tokens found, starting OAuth flow")
            auth_code = get_auth_code(client_id)
            tokens = get_tokens(client_id, client_secret, auth_code=auth_code)
        
        # Save tokens
        with open(token_file, 'w') as f:
            json.dump(tokens, f)
        
        return tokens
    except Exception as e:
        logger.error(f"Error initializing Fitbit client: {str(e)}")
        raise

def refresh_token_cb(token_dict):
    """Callback function to handle token refresh."""
    # Save the updated tokens
    with open("fitbit_tokens.json", 'w') as f:
        json.dump(token_dict, f)
    return token_dict

def get_body_measurements(tokens):
    """Get weight data from Fitbit."""
    try:
        client = fitbit.Fitbit(
            os.getenv("FITBIT_CLIENT_ID"),
            os.getenv("FITBIT_CLIENT_SECRET"),
            access_token=tokens['access_token'],
            refresh_token=tokens['refresh_token'],
            refresh_cb=refresh_token_cb
        )
        
        # Use config start date and current date
        end_date = datetime.now().date()
        start_date = config.DATA_START_DATE
        logger.info(f"Fetching weight data from {start_date} to {end_date}")
        
        # Get weight data using time series API
        weight_data = []
        
        # Get weight time series
        data = client.time_series('body/weight', base_date=start_date.strftime('%Y-%m-%d'),
                                end_date=end_date.strftime('%Y-%m-%d'))
        entries = data.get('body-weight', [])
        logger.info(f"Retrieved {len(entries)} weight entries")
        
        # Get body fat time series
        fat_data = client.time_series('body/fat', base_date=start_date.strftime('%Y-%m-%d'),
                                    end_date=end_date.strftime('%Y-%m-%d'))
        fat_entries = fat_data.get('body-fat', [])
        logger.info(f"Retrieved {len(fat_entries)} body fat entries")
        
        # Create a dictionary to store body fat values by date
        fat_by_date = {
            entry['dateTime']: float(entry['value'])
            for entry in fat_entries
        }
        
        # Process weight entries and combine with body fat data
        for entry in entries:
            date = entry['dateTime']
            weight_data.append({
                'date': date,
                'weight': float(entry['value']),
                'body_fat': fat_by_date.get(date)  # Add body fat if available for this date
            })
        
        # Convert to DataFrame and sort
        df = pd.DataFrame(weight_data)
        if not df.empty:
            df = df.sort_values('date')
            logger.info(f"Final dataset: {len(df)} entries from {df['date'].min()} to {df['date'].max()}")
        else:
            logger.warning("No weight data found")
        
        return df
    except Exception as e:
        logger.error(f"Error getting weight data: {str(e)}")
        if hasattr(e, 'response'):
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text}")
        return pd.DataFrame()

def run_fitbit_etl():
    """Execute Fitbit ETL process."""
    load_dotenv('Credentials.env')
    try:
        tokens = init_fitbit()
        df = get_body_measurements(tokens)
        if not df.empty:
            # Save to CSV
            df.to_csv(config.FITBIT_MEASUREMENTS_FILE, index=False)
            logger.info(f'Fitbit measurements saved to {config.FITBIT_MEASUREMENTS_FILE}')
        return df
    except Exception as e:
        logger.error(f"Error in Fitbit ETL: {str(e)}")
        return None

def main():
    """Main function to test Fitbit API integration."""
    try:
        df = run_fitbit_etl()
        if df is not None and not df.empty:
            print("\nRecent body measurements:")
            print(df)
        else:
            print("No data retrieved")
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 