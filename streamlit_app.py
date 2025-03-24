import streamlit as st
import concurrent.futures
from viz.dashboard_components import display_metric_box
from viz.metrics_handler import get_metrics
from viz.styles import DASHBOARD_CSS
from etl import run_garmin_etl, run_whoop_etl, run_mfp_daily_only, run_gsheets_etl, run_glucose_etl
import threading

# Set page config
st.set_page_config(
    page_title="Health Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Apply CSS
st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

# Hide Streamlit elements using an alternative method (in addition to CSS)
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# First, display the metrics with current data
metrics = get_metrics()

# Function to run ETL processes and then rerun the app
def run_background_etl():
    try:
        # Run ETL processes in parallel
        def run_etl(etl_func):
            etl_func()
        
        # Create tasks for parallel execution
        etl_tasks = [
            run_garmin_etl,
            run_whoop_etl,
            run_mfp_daily_only,
            run_gsheets_etl,
            run_glucose_etl
        ]
        
        # Execute tasks in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(run_etl, func) for func in etl_tasks]
            for future in concurrent.futures.as_completed(futures):
                # Just wait for completion
                future.result()
        
        # Rerun the app to refresh the data
        st.rerun()
                
    except Exception as e:
        # Silent exception handling - log but don't show to user
        print(f"Background update failed: {str(e)}")

# Start background thread for ETL processing
if 'etl_started' not in st.session_state:
    st.session_state.etl_started = True
    threading.Thread(target=run_background_etl).start()

# Main layout - display immediately with current data
if metrics:
    for metric_name in ['Nutrition', 'Glucose', 'Recovery', 'Sleep', 'Running', 'Strength']:
        display_metric_box(metric_name,
                         metrics[metric_name.lower()]['primary'],
                         metrics[metric_name.lower()]['secondary1'],
                         metrics[metric_name.lower()]['secondary2'])
