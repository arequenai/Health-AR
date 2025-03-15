import streamlit as st
import concurrent.futures
from viz.dashboard_components import display_metric_box
from viz.metrics_handler import get_metrics
from viz.styles import DASHBOARD_CSS
from etl import run_garmin_etl, run_whoop_etl, run_mfp_daily_only, run_gsheets_etl

# Set page config
st.set_page_config(
    page_title="Health Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply CSS
st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

# Auto-update data on page load
with st.spinner('Updating data...'):
    try:
        # Run ETL processes in parallel
        def run_etl(etl_func):
            etl_func()
        
        # Create tasks for parallel execution
        etl_tasks = [
            run_garmin_etl,
            run_whoop_etl,
            run_mfp_daily_only,
            run_gsheets_etl
        ]
        
        # Execute tasks in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(run_etl, func) for func in etl_tasks]
            for future in concurrent.futures.as_completed(futures):
                # Just wait for completion, no need to track timing
                future.result()
                
    except Exception as e:
        st.error(f"Update failed: {str(e)}")

# Main layout
metrics = get_metrics()
if metrics:
    for metric_name in ['Nutrition', 'Recovery', 'Sleep', 'Running', 'Strength', 'Glucose']:
        display_metric_box(metric_name,
                         metrics[metric_name.lower()]['primary'],
                         metrics[metric_name.lower()]['secondary1'],
                         metrics[metric_name.lower()]['secondary2'])
