import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
from etl import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JefitClient:
    def __init__(self, user_id):
        self.user_id = user_id
        self.base_url = "https://www.jefit.com/members/user-logs"
        
    def get_workout_for_date(self, date):
        """Get workout data for a specific date."""
        try:
            url = f"{self.base_url}/?xid={self.user_id}&dd={date}"
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract workout data
            exercises = []
            workout_table = soup.find('table', {'class': 'workout-table'})
            
            if workout_table:
                for row in workout_table.find_all('tr')[1:]:  # Skip header
                    cols = row.find_all('td')
                    if len(cols) >= 4:
                        exercise = {
                            'name': cols[0].text.strip(),
                            'sets': cols[1].text.strip(),
                            'reps': cols[2].text.strip(),
                            'weight': cols[3].text.strip()
                        }
                        exercises.append(exercise)
            
            return {
                'date': date,
                'exercises': exercises
            }
            
        except Exception as e:
            logger.error(f"Error fetching workout for {date}: {str(e)}")
            return None

def run_jefit_etl():
    """Execute Jefit ETL process."""
    load_dotenv('Credentials.env')
    
    try:
        user_id = os.getenv("JEFIT_USER_ID")
        if not user_id:
            raise ValueError("JEFIT_USER_ID not found in environment variables")
        
        client = JefitClient(user_id)
        
        # Get data for last 30 days
        end_date = datetime.now().date()
        start_date = config.DATA_START_DATE
        
        all_workouts = []
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            workout = client.get_workout_for_date(date_str)
            
            if workout and workout['exercises']:
                all_workouts.append(workout)
                logger.info(f"Retrieved workout for {date_str}")
            else:
                logger.debug(f"No workout found for {date_str}")
            
            current_date += timedelta(days=1)
        
        # Convert to DataFrame
        workout_data = []
        for workout in all_workouts:
            for exercise in workout['exercises']:
                workout_data.append({
                    'date': workout['date'],
                    'exercise': exercise['name'],
                    'sets': exercise['sets'],
                    'reps': exercise['reps'],
                    'weight': exercise['weight']
                })
        
        df = pd.DataFrame(workout_data)
        
        if not df.empty:
            df.to_csv(config.JEFIT_WORKOUTS_FILE, index=False)
            logger.info(f'Jefit workouts saved to {config.JEFIT_WORKOUTS_FILE}')
            return df
        else:
            logger.warning("No workout data found")
            return None
            
    except Exception as e:
        logger.error(f"Error in Jefit ETL: {str(e)}")
        return None

def main():
    """Main function to test Jefit ETL."""
    try:
        df = run_jefit_etl()
        if df is not None and not df.empty:
            print("\nRecent workouts:")
            print(df)
        else:
            print("No data retrieved")
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
