import streamlit as st
import pandas as pd
from etl import init_garmin, get_garmin_data, init_whoop, get_sleep_recovery_data, init_mfp, get_meal_data, get_meal_daily, config
import os

# Set page config for mobile optimization
st.set_page_config(
    page_title="Health Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for mobile optimization
st.markdown("""
    <style>
    .metric-container {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
    }
    .primary-metric {
        font-size: 24px;
        font-weight: bold;
    }
    .secondary-metric {
        font-size: 14px;
        color: #888;
    }
    .good { color: #00FF00; }
    .warning { color: #FFA500; }
    .bad { color: #FF0000; }
    [data-testid="stHorizontalBlock"] {
        gap: 0rem;
    }
    .level-5 { color: #2ECC40; }  /* Dark green */
    .level-4 { color: #01FF70; }  /* Light green */
    .level-3 { color: #FFDC00; }  /* Yellow */
    .level-2 { color: #FF851B; }  /* Light orange */
    .level-1 { color: #FF4136; }  /* Dark orange */
    </style>
""", unsafe_allow_html=True)

def run_garmin_etl():
    """
    Execute Garmin ETL to update data.
    """
    try:
        email = os.getenv("USERNAME_G")
        password = os.getenv("PASSWORD_G")
        
        garmin_client = init_garmin(email, password)
        df = get_garmin_data(garmin_client)
        
        if df is not None:
            df.to_csv(config.GARMIN_DAILY_FILE, index=False)
            st.success("Garmin ETL executed successfully")
        else:
            st.error("Garmin ETL failed to execute")
    except Exception as e:
        st.error(f"Garmin ETL failed: {str(e)}")

def run_whoop_etl():
    """
    Execute Whoop ETL to update data.
    """
    try:
        un = os.getenv("USERNAME_W")
        pw = os.getenv("PASSWORD_W")
        
        whoop_client = init_whoop(un, pw)
        df = get_sleep_recovery_data(whoop_client)
        
        if df is not None:
            df.to_csv(config.WHOOP_SLEEP_RECOVERY_FILE, index=False)
            st.success("Whoop ETL executed successfully")
        else:
            st.error("Whoop ETL failed to execute")
    except Exception as e:
        st.error(f"Whoop ETL failed: {str(e)}")

def run_mfp_etl():
    """
    Execute MyFitnessPal ETL to update data.
    """
    try:
        client = init_mfp()
        get_meal_data(client, config.MFP_MEALS_FILE)
        get_meal_daily(client, config.MFP_DAILY_FILE)
        st.success("MyFitnessPal ETL executed successfully")
    except Exception as e:
        st.error(f"MyFitnessPal ETL failed: {str(e)}")

def get_metrics():
    """Get all metrics from various sources."""
    try:
        metrics = {
            'nutrition': {
                'primary': {'value': 2100, 'label': 'kcal today'},  # From MFP
                'secondary1': {'value': '75.5', 'label': 'kg'},     # Placeholder
                'secondary2': {'value': '120', 'label': 'g protein'} # From MFP
            },
            'recovery': {
                'primary': {'value': 85, 'label': 'recovery'},      # From Whoop
                'secondary1': {'value': '65', 'label': 'battery'},  # From Whoop
                'secondary2': {'value': '45', 'label': 'stress'}    # From Whoop
            },
            'sleep': {
                'primary': {'value': 88, 'label': 'quality'},       # From Whoop
                'secondary1': {'value': '7:30', 'label': 'in bed'}, # From Whoop
                'secondary2': {'value': 'Good', 'label': 'behavior'} # From Whoop Journal
            },
            'running': {
                'primary': {'value': -5, 'label': 'TSB'},          # Calculated
                'secondary1': {'value': '45', 'label': 'CTL'},     # Calculated
                'secondary2': {'value': '35', 'label': 'km L7D'}   # From Garmin
            },
            'strength': {
                'primary': {'value': 3, 'label': 'days since'},    # Placeholder
                'secondary1': {'value': '12', 'label': 'pullups'}, # Placeholder
                'secondary2': {'value': '45', 'label': 'min'}      # Placeholder
            },
            'glucose': {
                'primary': {'value': 95, 'label': 'mg/dL'},        # Placeholder
                'secondary1': {'value': '85', 'label': 'fasting'}, # Placeholder
                'secondary2': {'value': '105', 'label': 'mean'}    # Placeholder
            }
        }
        return metrics
    except Exception as e:
        st.error(f"Error getting metrics: {str(e)}")
        return None

def get_metric_level(metric_name, value):
    """Determine color level based on metric value."""
    thresholds = {
        'nutrition': {'5': 2000, '4': 1800, '3': 1600, '2': 1400, '1': 0},
        'recovery': {'5': 85, '4': 70, '3': 55, '2': 40, '1': 0},
        'sleep': {'5': 85, '4': 70, '3': 55, '2': 40, '1': 0},
        'running': {'5': 5, '4': 0, '3': -5, '2': -10, '1': -15},
        'strength': {'5': 1, '4': 2, '3': 3, '2': 4, '1': 5},  # Days since (lower is better)
        'glucose': {'5': 90, '4': 100, '3': 110, '2': 120, '1': 130}
    }
    
    for level in range(5, 0, -1):
        if float(value) >= float(thresholds[metric_name][str(level)]):
            return level
    return 1

def display_metric_box(title, primary, secondary1, secondary2):
    """Display a metric box with primary and secondary metrics."""
    level = get_metric_level(title.lower(), primary['value'])
    with st.container():
        st.markdown(f"""
            <div class="metric-container">
                <h3>{title}</h3>
                <div class="primary-metric level-{level}">{primary['value']} <span style="font-size:14px">{primary['label']}</span></div>
                <div class="secondary-metric">{secondary1['value']} {secondary1['label']}</div>
                <div class="secondary-metric">{secondary2['value']} {secondary2['label']}</div>
            </div>
        """, unsafe_allow_html=True)

# Main layout
metrics = get_metrics()
if metrics:
    # First row
    col1, col2 = st.columns(2)
    with col1:
        display_metric_box("Nutrition", 
                         metrics['nutrition']['primary'],
                         metrics['nutrition']['secondary1'],
                         metrics['nutrition']['secondary2'])
    with col2:
        display_metric_box("Recovery",
                         metrics['recovery']['primary'],
                         metrics['recovery']['secondary1'],
                         metrics['recovery']['secondary2'])
    
    # Second row
    col1, col2 = st.columns(2)
    with col1:
        display_metric_box("Sleep",
                         metrics['sleep']['primary'],
                         metrics['sleep']['secondary1'],
                         metrics['sleep']['secondary2'])
    with col2:
        display_metric_box("Running",
                         metrics['running']['primary'],
                         metrics['running']['secondary1'],
                         metrics['running']['secondary2'])
    
    # Third row
    col1, col2 = st.columns(2)
    with col1:
        display_metric_box("Strength",
                         metrics['strength']['primary'],
                         metrics['strength']['secondary1'],
                         metrics['strength']['secondary2'])
    with col2:
        display_metric_box("Glucose",
                         metrics['glucose']['primary'],
                         metrics['glucose']['secondary1'],
                         metrics['glucose']['secondary2'])

# Small update button at the bottom
if st.button("↻ Update Data", key="update_all"):
    try:
        run_garmin_etl()
        run_whoop_etl()
        run_mfp_etl()
    except Exception as e:
        st.error(f"Update failed: {str(e)}")
