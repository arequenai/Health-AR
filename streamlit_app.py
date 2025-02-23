import streamlit as st
from viz.dashboard_components import display_metric_box
from viz.metrics_handler import get_metrics
from viz.styles import DASHBOARD_CSS
from etl import run_garmin_etl, run_whoop_etl, run_mfp_etl, run_fitbit_etl

# Set page config
st.set_page_config(
    page_title="Health Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply CSS
st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

# Main layout
metrics = get_metrics()
if metrics:
    for metric_name in ['Nutrition', 'Recovery', 'Sleep', 'Running', 'Strength', 'Glucose']:
        display_metric_box(metric_name,
                         metrics[metric_name.lower()]['primary'],
                         metrics[metric_name.lower()]['secondary1'],
                         metrics[metric_name.lower()]['secondary2'])

# Update button and status message placeholder
update_col, status_col = st.columns([1, 4])
with update_col:
    update_clicked = st.button("↻", key="update_all")
with status_col:
    status_placeholder = st.empty()

if update_clicked:
    try:
        with st.spinner(' '):
            run_garmin_etl()
            run_whoop_etl()
            run_mfp_etl()
            run_fitbit_etl()
            st.rerun() # Refresh metrics after ETL
        status_placeholder.success("✓")
    except Exception as e:
        status_placeholder.error(f"Update failed: {str(e)}")
