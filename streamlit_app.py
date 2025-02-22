import streamlit as st
from viz.dashboard_components import display_metric_box
from viz.metrics_handler import get_metrics
from viz.styles import DASHBOARD_CSS
from etl import run_garmin_etl, run_whoop_etl, run_mfp_etl

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
    # Display metrics in grid
    for row in [('Nutrition', 'Recovery'), ('Sleep', 'Running'), ('Strength', 'Glucose')]:
        col1, col2 = st.columns(2)
        with col1:
            display_metric_box(row[0], 
                             metrics[row[0].lower()]['primary'],
                             metrics[row[0].lower()]['secondary1'],
                             metrics[row[0].lower()]['secondary2'])
        with col2:
            display_metric_box(row[1],
                             metrics[row[1].lower()]['primary'],
                             metrics[row[1].lower()]['secondary1'],
                             metrics[row[1].lower()]['secondary2'])

# Update button and status message placeholder
update_col, status_col = st.columns([1, 4])
with update_col:
    update_clicked = st.button("↻ Update", key="update_all")
with status_col:
    status_placeholder = st.empty()

if update_clicked:
    try:
        with st.spinner(' '):
            run_garmin_etl()
            run_whoop_etl()
            run_mfp_etl()
        status_placeholder.success("✓ Updated")
    except Exception as e:
        status_placeholder.error(f"Update failed: {str(e)}")
