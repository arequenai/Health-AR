import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from .data_utils import get_day_data, get_body_battery_data, get_stress_data, get_sleep_data, get_glucose_day_data
from .styles import DEEP_DIVE_CSS

def create_day_view_chart(selected_date):
    """
    Create a comprehensive day view chart showing various health metrics.
    
    Args:
        selected_date (datetime): The date to create the chart for
        
    Returns:
        plotly.graph_objects.Figure: The created figure
    """
    # Get all the data for the selected date
    data = get_day_data(selected_date)
    
    # Get body battery data
    body_battery_data = get_body_battery_data(selected_date)
    
    # Get stress data
    stress_data = get_stress_data(selected_date)
    
    # Get sleep data
    sleep_data = get_sleep_data(selected_date)
    
    # Get detailed glucose data
    glucose_data = get_glucose_day_data(selected_date)
    
    # Create the figure
    fig = go.Figure()
    
    # Set up the x-axis to show hours of the day
    day_start = datetime.combine(selected_date.date(), datetime.min.time())
    day_end = day_start + timedelta(days=1)
    
    # Setup the layout with multiple y-axes
    fig.update_layout(
        xaxis=dict(
            title="Time of Day",
            type="date",
            range=[day_start, day_end],
            tickformat="%H:%M",
            dtick=3600000,  # 1 hour in milliseconds
        ),
        yaxis=dict(
            title="Body Battery",
            range=[0, 100],
            side="left",
            showgrid=True
        ),
        yaxis2=dict(
            title="Glucose (mg/dL)",
            range=[40, 300],
            side="right",
            overlaying="y",
            showgrid=False,
            anchor="x"
        ),
        yaxis3=dict(
            title="Stress",
            range=[0, 100],
            side="right",
            overlaying="y",
            showgrid=False,
            anchor="free",
            position=1.0
        ),
        margin=dict(l=50, r=50, t=50, b=50),
        height=600,
        hovermode="x unified"
    )
    
    # Add threshold lines for glucose
    fig.add_shape(
        type="line",
        xref="paper", yref="y2",
        x0=0, x1=1, y0=70, y1=70,
        line=dict(color="rgba(255, 0, 0, 0.3)", width=1, dash="dash"),
    )
    
    fig.add_shape(
        type="line",
        xref="paper", yref="y2",
        x0=0, x1=1, y0=180, y1=180,
        line=dict(color="rgba(255, 0, 0, 0.3)", width=1, dash="dash"),
    )
    
    # Add body battery data if available
    if body_battery_data is not None and not body_battery_data.empty:
        fig.add_trace(go.Scatter(
            x=body_battery_data['datetime'],
            y=body_battery_data['body_battery'],
            mode='lines',
            name='Body Battery',
            line=dict(color='#4CAF50', width=3),
            hovertemplate='%{y:.0f}'
        ))
    elif data['garmin'] is not None and 'bodyBatteryMostRecentValue' in data['garmin'] and not pd.isna(data['garmin']['bodyBatteryMostRecentValue']):
        # If we don't have detailed time series data, show the daily value as a horizontal line
        body_battery_value = data['garmin']['bodyBatteryMostRecentValue']
        fig.add_shape(
            type="line",
            xref="paper", yref="y",
            x0=0, x1=1, y0=body_battery_value, y1=body_battery_value,
            line=dict(color='#4CAF50', width=2, dash='dash'),
        )
    
    # Add glucose data if available
    if glucose_data is not None and not glucose_data.empty:
        fig.add_trace(go.Scatter(
            x=glucose_data['datetime'],
            y=glucose_data['glucose'],
            mode='lines',
            name='Glucose',
            line=dict(color='#FF5722', width=3),
            yaxis='y2',
            hovertemplate='%{y:.0f} mg/dL'
        ))
    elif data['glucose_daily'] is not None:
        # If we don't have detailed time series data, show the daily average as a horizontal line
        mean_glucose = data['glucose_daily']['mean_glucose']
        fig.add_shape(
            type="line",
            xref="paper", yref="y2",
            x0=0, x1=1, y0=mean_glucose, y1=mean_glucose,
            line=dict(color='#FF5722', width=2, dash='dash'),
        )
        
        # Add annotation for mean glucose
        fig.add_annotation(
            xref="paper", yref="y2",
            x=0.02, y=mean_glucose,
            text=f"Avg: {int(mean_glucose)} mg/dL",
            showarrow=False,
            font=dict(color='#FF5722', size=10),
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#FF5722",
            borderwidth=1,
        )
        
        # Add fasting glucose marker
        fasting_glucose = data['glucose_daily']['fasting_glucose']
        fig.add_trace(go.Scatter(
            x=[day_start + timedelta(hours=6)],  # Assume fasting is at 6 AM
            y=[fasting_glucose],
            mode='markers',
            name='Fasting Glucose',
            marker=dict(color='#FFEB3B', size=10, symbol='star'),
            yaxis='y2'
        ))
    
    # Add stress data if available
    if stress_data is not None and not stress_data.empty:
        fig.add_trace(go.Scatter(
            x=stress_data['datetime'],
            y=stress_data['stress'],
            mode='lines',
            name='Stress',
            line=dict(color='#d3525b', width=2),
            yaxis='y3'
        ))
    elif data['garmin'] is not None and 'averageStressLevel' in data['garmin'] and not pd.isna(data['garmin']['averageStressLevel']):
        # If detailed stress data isn't available, just show the average value
        # Show horizontal line for average stress
        fig.add_shape(
            type="line",
            xref="paper", yref="y3",
            x0=0, x1=1, y0=data['garmin']['averageStressLevel'], y1=data['garmin']['averageStressLevel'],
            line=dict(color='#d3525b', width=1, dash='dash'),
        )
        
        # Add label for average stress
        fig.add_annotation(
            xref="paper", yref="y3",
            x=0.02, y=data['garmin']['averageStressLevel'],
            text=f"Avg Stress: {int(data['garmin']['averageStressLevel'])}",
            showarrow=False,
            font=dict(color='#d3525b', size=10),
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#d3525b",
            borderwidth=1,
        )
    
    # Add sleep data if available
    if sleep_data is not None:
        sleep_start = sleep_data['sleep_start']
        sleep_end = sleep_data['sleep_end']
        sleep_duration = sleep_data['sleep_duration']
        
        # Add sleep label
        sleep_hours = int(sleep_duration)
        sleep_minutes = int((sleep_duration - sleep_hours) * 60)
        sleep_text = f"Sleep ({sleep_hours}h {sleep_minutes}m)"
        
        # Add sleep score if available
        if 'sleep_score' in sleep_data and sleep_data['sleep_score'] > 0:
            sleep_text += f" - Score: {sleep_data['sleep_score']}"
        
        # If we have detailed sleep levels, display them
        if 'sleep_levels' in sleep_data and sleep_data['sleep_levels']:
            # Define colors for sleep stages - updated to match the image
            sleep_stage_colors = {
                0: "#e986e0",  # Awake - lighter purple
                1: "#6eaff5",  # Light sleep - lighter blue
                2: "#1e58b3",  # Deep sleep - dark blue
                3: "#d53dd3"   # REM sleep - dark purple
            }
            sleep_stage_names = {
                0: "Despierto",
                1: "Ligero",
                2: "Profundo",
                3: "MOR"
            }
            
            for level in sleep_data['sleep_levels']:
                try:
                    # Parse timestamps - we need to handle both formats
                    if isinstance(level['startGMT'], str):
                        start_time = datetime.strptime(level['startGMT'], '%Y-%m-%dT%H:%M:%S.%f') if '.' in level['startGMT'] else datetime.strptime(level['startGMT'], '%Y-%m-%dT%H:%M:%S.0')
                        end_time = datetime.strptime(level['endGMT'], '%Y-%m-%dT%H:%M:%S.%f') if '.' in level['endGMT'] else datetime.strptime(level['endGMT'], '%Y-%m-%dT%H:%M:%S.0')
                        
                        # Convert to local time
                        local_tz = datetime.now().astimezone().tzinfo
                        start_time = start_time.replace(tzinfo=pytz.UTC).astimezone(local_tz).replace(tzinfo=None)
                        end_time = end_time.replace(tzinfo=pytz.UTC).astimezone(local_tz).replace(tzinfo=None)
                    else:
                        # If timestamps are already millisecond values
                        start_time = datetime.fromtimestamp(level['startGMT']/1000)
                        end_time = datetime.fromtimestamp(level['endGMT']/1000)
                        
                    activity_level = level['activityLevel']
                    
                    # Map Garmin sleep levels to our visualization
                    # In the Garmin API:
                    # activityLevel values:
                    # 0.0 = Deep sleep
                    # 1.0 = Light sleep
                    # 2.0 = REM sleep
                    # 3.0 = Awake
                    stage = 0  # Default to Awake
                    
                    # Mapping based on the API response format
                    if activity_level == 0.0:
                        stage = 2  # Deep sleep
                    elif activity_level == 1.0:
                        stage = 1  # Light sleep
                    elif activity_level == 2.0:
                        stage = 3  # REM sleep
                    elif activity_level == 3.0:
                        stage = 0  # Awake
                    
                    # Add the colored rectangle for this sleep stage
                    fig.add_shape(
                        type="rect",
                        x0=start_time, x1=end_time,
                        y0=0, y1=100,
                        fillcolor=sleep_stage_colors[stage],
                        opacity=0.7,
                        layer="below",
                        line_width=0,
                    )
                except Exception as e:
                    st.error(f"Error processing sleep level: {e}")
        else:
            # If detailed sleep levels are not available, just show a basic rectangle for the sleep period
            if sleep_start and sleep_end:
                fig.add_shape(
                    type="rect",
                    x0=sleep_start, x1=sleep_end,
                    y0=0, y1=100,
                    fillcolor="#6eaff5",  # Light blue for sleep
                    opacity=0.5,
                    layer="below",
                    line_width=0,
                )
        
        # Add sleep label
        if sleep_start and sleep_end:
            mid_point = sleep_start + (sleep_end - sleep_start) / 2
            fig.add_annotation(
                x=mid_point,
                y=90,
                text=sleep_text,
                showarrow=False,
                font=dict(color='white', size=10),
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="#6eaff5",
                borderwidth=1,
            )
    
    # Add activities data if available
    if data['activities'] is not None and not data['activities'].empty:
        # Define colors for different activity types
        activity_colors = {
            'running': '#ff9800',  # Orange
            'trail_running': '#ff5722',  # Deep orange
            'treadmill_running': '#ff7043',  # Lighter orange
            'cycling': '#4caf50',  # Green
            'indoor_cycling': '#81c784',  # Light green
            'hiking': '#795548',  # Brown
            'walking': '#9e9e9e',  # Grey
            'swimming': '#03a9f4',  # Light blue
            'lap_swimming': '#039be5',  # Blue
            'open_water_swimming': '#0288d1',  # Darker blue
            'strength_training': '#673ab7',  # Deep purple
            'fitness_equipment': '#9575cd',  # Light purple
            'yoga': '#ec407a',  # Pink
            'cardio': '#f44336',  # Red
            'default': '#607d8b'  # Blue grey - for any other type
        }
        
        # Process each activity
        for idx, activity in data['activities'].iterrows():
            try:
                # Skip if no start_time_local or duration
                if pd.isna(activity['start_time_local']) or pd.isna(activity['duration']):
                    continue
                
                # Parse the start time
                if isinstance(activity['start_time_local'], str):
                    try:
                        # Try different formats
                        if 'T' in activity['start_time_local']:
                            start_time = datetime.strptime(activity['start_time_local'], '%Y-%m-%dT%H:%M:%S.%f') if '.' in activity['start_time_local'] else datetime.strptime(activity['start_time_local'], '%Y-%m-%dT%H:%M:%S')
                        else:
                            start_time = datetime.strptime(activity['start_time_local'], '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        # If standard formats fail, try a more flexible approach
                        start_time = pd.to_datetime(activity['start_time_local']).to_pydatetime()
                else:
                    # If it's already a timestamp or datetime
                    start_time = pd.to_datetime(activity['start_time_local']).to_pydatetime()
                
                # Calculate end time based on duration (in seconds)
                duration_seconds = float(activity['duration'])
                end_time = start_time + timedelta(seconds=duration_seconds)
                
                # Get activity type and corresponding color
                activity_type = str(activity['type']).lower()
                color = activity_colors.get(activity_type, activity_colors['default'])
                
                # Add rectangle for the activity
                # Place activities to the right of sleep data
                fig.add_shape(
                    type="rect",
                    x0=start_time, x1=end_time,
                    y0=0, y1=100,
                    fillcolor=color,
                    opacity=0.7,
                    layer="below",
                    line_width=0,
                    xref="x", yref="y",
                )
                
                # Add label for the activity
                activity_type_display = activity_type.replace('_', ' ').title()
                duration_min = int(duration_seconds / 60)
                
                # Show label in the middle of the activity rectangle
                mid_point = start_time + (end_time - start_time) / 2
                fig.add_annotation(
                    x=mid_point,
                    y=50,  # Middle of the y-axis
                    text=f"{activity_type_display}<br>{duration_min} min",
                    showarrow=False,
                    font=dict(color='white', size=10),
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor=color,
                    borderwidth=1,
                )
                
            except Exception as e:
                st.error(f"Error processing activity: {e}")
    
    return fig

def display_day_view():
    """
    Display the day view deep dive page in Streamlit.
    """
    # Apply CSS
    st.markdown(DEEP_DIVE_CSS, unsafe_allow_html=True)
    
    st.title("Health Deep Dive: Day View")
    
    # Create a container for the date selector
    date_container = st.container()
    
    with date_container:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Use a calendar date picker instead of dropdown
            selected_date = st.date_input(
                'Select a date to view:',
                value=datetime.now().date() - timedelta(days=1),
                format="YYYY-MM-DD"
            )
            # Convert to pandas Timestamp for consistency with the rest of the code
            selected_date = pd.Timestamp(selected_date)
    
    # Create and display the main day view chart
    chart_container = st.container()
    
    with chart_container:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = create_day_view_chart(selected_date)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional sections for the selected day
    data = get_day_data(selected_date)
    
    # Display daily metrics in columns
    col1, col2, col3 = st.columns(3)
    
    # Column 1: Sleep metrics
    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.subheader("Sleep")
        
        # Check if we have detailed sleep data from the API
        sleep_data = get_sleep_data(selected_date)
        if sleep_data and 'duration_seconds' in sleep_data and sleep_data['duration_seconds'] > 0:
            sleep_hours = sleep_data['duration_seconds'] / 3600
            st.metric("Sleep Duration", f"{sleep_hours:.2f} hours")
            
            if 'sleep_score' in sleep_data and sleep_data['sleep_score'] > 0:
                st.metric("Sleep Score", f"{int(sleep_data['sleep_score'])}")
                
            # Display sleep stages if available
            if 'deep_sleep' in sleep_data and sleep_data['deep_sleep'] > 0:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Deep Sleep", f"{sleep_data['deep_sleep']:.1f}h")
                    st.metric("Light Sleep", f"{sleep_data['light_sleep']:.1f}h")
                with col_b:
                    st.metric("REM Sleep", f"{sleep_data['rem_sleep']:.1f}h")
                    st.metric("Awake", f"{sleep_data['awake']:.1f}h")
                
        elif data['garmin'] is not None and 'sleepingSeconds' in data['garmin'] and not pd.isna(data['garmin']['sleepingSeconds']):
            sleep_hours = data['garmin']['sleepingSeconds'] / 3600
            st.metric("Sleep Duration", f"{sleep_hours:.2f} hours")
            
            if 'sleep_score' in data['garmin'] and not pd.isna(data['garmin']['sleep_score']):
                st.metric("Sleep Score", f"{int(data['garmin']['sleep_score'])}")
        else:
            st.write("No sleep data available for this date")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Column 2: Glucose metrics
    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.subheader("Glucose")
        if data['glucose_daily'] is not None:
            mean_glucose = data['glucose_daily']['mean_glucose']
            min_glucose = data['glucose_daily']['min_glucose']
            max_glucose = data['glucose_daily']['max_glucose']
            fasting_glucose = data['glucose_daily']['fasting_glucose']
            
            st.metric("Mean Glucose", f"{int(mean_glucose)} mg/dL")
            st.metric("Fasting Glucose", f"{int(fasting_glucose)} mg/dL")
            st.metric("Min/Max", f"{int(min_glucose)}/{int(max_glucose)} mg/dL")
        else:
            st.write("No glucose data available for this date")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Column 3: Activity metrics
    with col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.subheader("Activity")
        if data['garmin'] is not None:
            steps = data['garmin']['totalSteps'] if 'totalSteps' in data['garmin'] else 0
            calories = data['garmin']['activeKilocalories'] if 'activeKilocalories' in data['garmin'] else 0
            
            st.metric("Steps", f"{int(steps):,}")
            st.metric("Active Calories", f"{int(calories):,}")
            
            if 'bodyBatteryMostRecentValue' in data['garmin'] and not pd.isna(data['garmin']['bodyBatteryMostRecentValue']):
                st.metric("Body Battery", f"{int(data['garmin']['bodyBatteryMostRecentValue'])}")
                
            if 'averageStressLevel' in data['garmin'] and not pd.isna(data['garmin']['averageStressLevel']):
                st.metric("Avg Stress", f"{int(data['garmin']['averageStressLevel'])}")
        else:
            st.write("No activity data available for this date")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display meals table if available
    st.subheader("Meals")
    if 'meals' in data and not data['meals'].empty:
        # Clean up food names
        def clean_food_name(food):
            if pd.isna(food):
                return ""
            
            # Remove everything to the right of a comma
            if ',' in food:
                food = food.split(',')[0]
            
            # Remove everything to the left of a dash
            if ' - ' in food:
                food = food.split(' - ')[1]
                
            # Capitalize first letter
            food = food.capitalize()
            
            return food
        
        data['meals']['food_clean'] = data['meals']['food'].apply(clean_food_name)
        
        # Display meals by type in columns
        meal_types = ['breakfast', 'lunch', 'dinner', 'snacks']
        meal_cols = st.columns(4)
        
        for i, meal_type in enumerate(meal_types):
            with meal_cols[i]:
                st.markdown(f'<div class="meal-container">', unsafe_allow_html=True)
                st.markdown(f'<div class="meal-title">{meal_type.capitalize()}</div>', unsafe_allow_html=True)
                
                meal_data = data['meals'][data['meals']['meal'] == meal_type]
                
                if not meal_data.empty:
                    # Calculate nutritional totals
                    total_calories = meal_data['calories'].sum()
                    total_carbs = meal_data['carbs'].sum()
                    total_fat = meal_data['fat'].sum()
                    total_protein = meal_data['protein'].sum()
                    
                    # Display summary metrics
                    st.markdown(f'<div class="meal-summary">Calories: {int(total_calories)}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="meal-summary">Carbs: {int(total_carbs)}g</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="meal-summary">Fat: {int(total_fat)}g</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="meal-summary">Protein: {int(total_protein)}g</div>', unsafe_allow_html=True)
                    
                    # List food items
                    st.markdown("<div><strong>Foods:</strong></div>", unsafe_allow_html=True)
                    for food in meal_data['food_clean'].unique():
                        if food:  # Only show non-empty food names
                            st.markdown(f'<div class="meal-item">• {food}</div>', unsafe_allow_html=True)
                else:
                    st.write("No data")
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.write("No meal data available for this date") 