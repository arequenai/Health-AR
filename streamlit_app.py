import streamlit as st
import concurrent.futures
from viz.dashboard_components import display_metric_box
from viz.metrics_handler import get_metrics
from viz.styles import DASHBOARD_CSS
from etl import run_garmin_etl, run_whoop_etl, run_mfp_daily_only, run_gsheets_etl, run_glucose_etl, run_g_journal_etl
import threading
from viz.deep_dive.day_view import display_day_view
from viz.weekly_evolution import show_weekly_evolution
from viz.current_week_analysis import show_current_week_analysis
from etl.load_data import load_data

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
            run_glucose_etl,
            run_g_journal_etl
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

# Add navigation to sidebar
with st.sidebar:
    st.title("Navigation")
    page = st.radio(
        "Go to",
        ["Dashboard", "Weekly Evolution", "Current Week Analysis", "Deep Dive: Day View"],
        index=0
    )

# Start background thread for ETL processing (only on dashboard)
if 'etl_started' not in st.session_state and page == "Dashboard":
    st.session_state.etl_started = True
    threading.Thread(target=run_background_etl).start()

# Display the selected page
if page == "Dashboard":
    # Display the dashboard
    metrics = get_metrics()
    if metrics:
        for metric_name in ['Nutrition', 'Glucose', 'Recovery', 'Sleep', 'Running', 'Strength']:
            display_metric_box(metric_name,
                            metrics[metric_name.lower()]['primary'],
                            metrics[metric_name.lower()]['secondary1'],
                            metrics[metric_name.lower()]['secondary2'])
elif page == "Weekly Evolution":
    # Display the weekly evolution page
    # Add a number input for selecting the number of weeks to show
    num_weeks = st.number_input(
        "Number of weeks to display", 
        min_value=2, 
        max_value=52, 
        value=8, 
        step=1,
        help="Select how many recent weeks to display in the charts and summary"
    )
    data = load_data()
    show_weekly_evolution(data, num_weeks)
elif page == "Current Week Analysis":
    # Display the current week analysis page
    # Add a number input for selecting the number of weeks for historical comparison
    num_weeks = st.number_input(
        "Historical comparison weeks", 
        min_value=2, 
        max_value=52, 
        value=8, 
        step=1,
        help="Select how many previous weeks to use for historical averages"
    )
    data = load_data()
    show_current_week_analysis(data, num_weeks)
elif page == "Deep Dive: Day View":
    # Display the day view
    display_day_view()
