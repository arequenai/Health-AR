import streamlit as st
import concurrent.futures
from viz.dashboard_components import display_metric_box
from viz.metrics_handler import get_metrics
from viz.styles import DASHBOARD_CSS
from etl import run_garmin_etl, run_whoop_etl, run_mfp_daily_only, run_gsheets_etl, run_glucose_etl

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

# More aggressive hiding of Streamlit elements
hide_st_style = """
<style>
/* Hide all Streamlit elements */
#MainMenu {visibility: hidden !important;}
footer {visibility: hidden !important;}
header {visibility: hidden !important;}

/* Hide the Streamlit branding */
.stApp div[data-testid="stToolbar"] {visibility: hidden !important;}
.stApp div[data-testid="stDecoration"] {visibility: hidden !important;}
.stApp section[data-testid="stSidebar"] {visibility: hidden !important;}
.stApp button[kind="minimalistic"] {display: none !important;}
.stApp a[href="https://streamlit.io"] {display: none !important;}
.stApp a[href^="https://share.streamlit.io"] {display: none !important;}
.stApp iframe[src^="https://share.streamlit.io"] {display: none !important;}

/* Hide the Streamlit footer brand */
.footer-logo-container {display: none !important;}
.stApp footer {display: none !important;}

/* Hide profile photo and crown */
img[src*="streamlit_logo"] {display: none !important;}
img[src*="streamlit-logo"] {display: none !important;}
a[href^="https://streamlit.io/cloud"] {display: none !important;}
.streamlit-logo {display: none !important;}
.eyeqlp51 {display: none !important;} /* Common class for branding elements */
[data-testid="stSidebarUserContent"] {display: none !important;}
[data-baseweb="button"] {display: none !important;}
[data-baseweb="popover"] {display: none !important;}

/* Additional selector to hide any floating elements */
div[data-testid="stFloatingViewerBadge"] {display: none !important;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Add this at the end of your app to remove any remaining elements with JavaScript
st.markdown("""
<script>
// Function to hide elements after the page loads
function hideStreamlitElements() {
  // Hide any element with streamlit or Streamlit in the class name, id, or data attributes
  document.querySelectorAll('[class*="streamlit"],[class*="Streamlit"],[id*="streamlit"],[id*="Streamlit"],[data-*="streamlit"],[data-*="Streamlit"]').forEach(el => {
    el.style.display = 'none';
  });
  
  // Specifically target the crown and profile elements
  document.querySelectorAll('img[alt="Streamlit logo"]').forEach(el => {
    el.style.display = 'none';
  });
  
  // Try to find the specific elements by their styles or position
  document.querySelectorAll('div[style*="position: fixed"]').forEach(el => {
    if (el.innerHTML.includes('streamlit') || el.innerHTML.includes('Streamlit')) {
      el.style.display = 'none';
    }
  });
}

// Execute after page load
window.addEventListener('load', hideStreamlitElements);
// Also run after a short delay to catch dynamically added elements
setTimeout(hideStreamlitElements, 1000);
</script>
""", unsafe_allow_html=True)

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
            run_gsheets_etl,
            run_glucose_etl
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
