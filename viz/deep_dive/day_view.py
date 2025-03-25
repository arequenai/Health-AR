import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from .data_utils import get_day_data, get_body_battery_data, get_stress_data, get_sleep_data
from .styles import DEEP_DIVE_CSS

def create_day_view_chart(selected_date):
    """
    Create a day view chart showing multiple health metrics over time.
    
    Args:
        selected_date (datetime or str): The date to visualize
    
    Returns:
        fig: Plotly figure object
    """
    # Convert to datetime if string
    if isinstance(selected_date, str):
        selected_date = pd.to_datetime(selected_date)
    
    # Get data for the selected date
    data = get_day_data(selected_date)
    
    # Get time series data
    body_battery_data = get_body_battery_data(selected_date)
    stress_data = get_stress_data(selected_date)
    sleep_data = get_sleep_data(selected_date)
    
    # Create an empty Plotly figure
    fig = go.Figure()
    
    # Set up the figure layout
    fig.update_layout(
        title=f'Daily Health Overview: {selected_date.strftime("%A, %B %d, %Y")}',
        height=600,
        margin=dict(l=40, r=40, t=80, b=40),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='rgba(255, 255, 255, 0.85)'),
        xaxis=dict(
            title='Time',
            gridcolor='rgba(100, 100, 100, 0.2)',
            showgrid=True,
            range=[
                datetime.combine(selected_date.date(), datetime.min.time()),
                datetime.combine(selected_date.date() + timedelta(days=1), datetime.min.time())
            ]
        ),
        yaxis=dict(
            title=dict(
                text='Glucose (mg/dL)',
                font=dict(color='white')
            ),
            gridcolor='rgba(100, 100, 100, 0.2)', 
            showgrid=True,
            range=[40, 200]  # Set a reasonable range for glucose
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add secondary y-axes for body battery and stress
    fig.update_layout(
        yaxis2=dict(
            title=dict(
                text='Body Battery',
                font=dict(color='#52b6d3')
            ),
            tickfont=dict(color='#52b6d3'),
            anchor="x",
            overlaying="y",
            side="right",
            range=[0, 100],
            showgrid=False,
        ),
        yaxis3=dict(
            title=dict(
                text='Stress',
                font=dict(color='#d3525b')
            ),
            tickfont=dict(color='#d3525b'),
            anchor="free",
            overlaying="y",
            side="right",
            position=0.95,
            range=[0, 100],
            showgrid=False,
        )
    )

    # Add glucose data if available
    if data['glucose_raw'] is not None and not data['glucose_raw'].empty:
        glucose_data = data['glucose_raw']
        
        # Add glucose trace
        fig.add_trace(go.Scatter(
            x=glucose_data['datetime'],
            y=glucose_data['glucose'],
            mode='lines',
            name='Glucose',
            line=dict(color='white', width=2),
        ))
        
        # Add reference bands
        fig.add_shape(
            type="rect",
            xref="paper", yref="y",
            x0=0, y0=140, x1=1, y1=200,
            fillcolor="red", opacity=0.2, layer="below", line_width=0,
        )
        
        fig.add_shape(
            type="rect",
            xref="paper", yref="y",
            x0=0, y0=70, x1=1, y1=100,
            fillcolor="green", opacity=0.2, layer="below", line_width=0,
        )
        
        fig.add_shape(
            type="rect",
            xref="paper", yref="y",
            x0=0, y0=0, x1=1, y1=70,
            fillcolor="yellow", opacity=0.2, layer="below", line_width=0,
        )
    
    # Add body battery data if available
    if body_battery_data is not None and not body_battery_data.empty:
        fig.add_trace(go.Scatter(
            x=body_battery_data['datetime'],
            y=body_battery_data['body_battery'],
            mode='lines',
            name='Body Battery',
            line=dict(color='#52b6d3', width=2),
            yaxis='y2'
        ))
    elif data['garmin'] is not None:
        # If detailed body battery data isn't available, just show the max and current values
        if 'max_sleep_body_battery' in data['garmin'] and not pd.isna(data['garmin']['max_sleep_body_battery']):
            # Show point for max battery from sleep
            morning_time = datetime.combine(selected_date.date(), datetime.strptime('08:00', '%H:%M').time())
            
            fig.add_trace(go.Scatter(
                x=[morning_time],
                y=[data['garmin']['max_sleep_body_battery']],
                mode='markers',
                name='Max Sleep Body Battery',
                marker=dict(color='#52b6d3', size=10),
                yaxis='y2'
            ))
        
        if 'bodyBatteryMostRecentValue' in data['garmin'] and not pd.isna(data['garmin']['bodyBatteryMostRecentValue']):
            # Show point for current battery
            end_of_day = datetime.combine(selected_date.date(), datetime.strptime('22:00', '%H:%M').time())
            
            fig.add_trace(go.Scatter(
                x=[end_of_day],
                y=[data['garmin']['bodyBatteryMostRecentValue']],
                mode='markers',
                name='Current Body Battery',
                marker=dict(color='#52b6d3', size=10),
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
                    
                    # Add colored rectangle for sleep stage
                    fig.add_shape(
                        type="rect",
                        xref="x", yref="paper",
                        x0=start_time, x1=end_time, y0=0.03, y1=0.17,
                        fillcolor=sleep_stage_colors.get(stage, "#a6a6a6"),
                        line=dict(width=0),
                        layer="below"
                    )
                except Exception as e:
                    print(f"Error processing sleep level: {e}")
                    continue
        else:
            # If we don't have detailed sleep levels, add a basic rectangle for the sleep period
            fig.add_shape(
                type="rect",
                xref="x", yref="paper",
                x0=sleep_start, x1=sleep_end, y0=0.03, y1=0.17,
                fillcolor="#6eaff5",  # Light blue for basic sleep visualization
                line=dict(color="#1e58b3", width=1),
                layer="below"
            )
            
        # Add the sleep label annotation
        fig.add_annotation(
            x=sleep_start + (sleep_end - sleep_start) / 2,
            y=0.1,
            yref="paper",
            text=sleep_text,
            showarrow=False,
            font=dict(color="cornflowerblue", size=12),
            bgcolor="rgba(0,0,0,0.5)",
        )
    
    # Check for sleep that starts on the current day (evening) but extends to the next day
    next_day = selected_date + pd.Timedelta(days=1)
    next_day_sleep = get_sleep_data(next_day)
    
    if next_day_sleep is not None and next_day_sleep['sleep_start'] is not None:
        sleep_start = next_day_sleep['sleep_start']
        
        # Only include if sleep started on the selected date (after 10 PM)
        if sleep_start.date() == selected_date.date() or (
            sleep_start.date() == next_day.date() and sleep_start.hour < 4
        ):
            sleep_end = next_day_sleep['sleep_end']
            sleep_duration = next_day_sleep['sleep_duration']
            
            # Add sleep label for evening sleep
            sleep_hours = int(sleep_duration)
            sleep_minutes = int((sleep_duration - sleep_hours) * 60)
            sleep_text = f"Sleep ({sleep_hours}h {sleep_minutes}m)"
            
            # Add sleep score if available
            if 'sleep_score' in next_day_sleep and next_day_sleep['sleep_score'] > 0:
                sleep_text += f" - Score: {next_day_sleep['sleep_score']}"
            
            # If we have detailed sleep levels, display them
            if 'sleep_levels' in next_day_sleep and next_day_sleep['sleep_levels']:
                # Use the same color scheme as for the morning sleep
                for level in next_day_sleep['sleep_levels']:
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
                        
                        # Only include sleep levels that fall on the current day
                        if start_time.date() != selected_date.date() and start_time.hour >= 4:
                            continue
                        
                        # Adjust end time if it's on the next day
                        if end_time.date() > selected_date.date():
                            end_time = datetime.combine(selected_date.date() + timedelta(days=1), datetime.min.time())
                        
                        # Map Garmin sleep levels to our visualization
                        stage = 0  # Default to Awake
                        if activity_level == 0.0:
                            stage = 2  # Deep sleep
                        elif activity_level == 1.0:
                            stage = 1  # Light sleep
                        elif activity_level == 2.0:
                            stage = 3  # REM sleep
                        elif activity_level == 3.0:
                            stage = 0  # Awake
                        
                        # Add colored rectangle for sleep stage
                        fig.add_shape(
                            type="rect",
                            xref="x", yref="paper",
                            x0=start_time, x1=end_time, y0=0.03, y1=0.17,
                            fillcolor=sleep_stage_colors.get(stage, "#a6a6a6"),
                            line=dict(width=0),
                            layer="below"
                        )
                    except Exception as e:
                        print(f"Error processing evening sleep level: {e}")
                        continue
            else:
                # If we don't have detailed sleep levels, add a basic rectangle for the sleep period
                # Only draw the portion that falls on the selected date
                evening_end = min(sleep_end, datetime.combine(selected_date.date() + timedelta(days=1), datetime.min.time()))
                
                fig.add_shape(
                    type="rect",
                    xref="x", yref="paper",
                    x0=sleep_start, x1=evening_end, y0=0.03, y1=0.17,
                    fillcolor="#6eaff5",  # Light blue for basic sleep visualization
                    line=dict(color="#1e58b3", width=1),
                    layer="below"
                )
            
            # Add the sleep label annotation
            label_x = sleep_start + (min(sleep_end, datetime.combine(selected_date.date() + timedelta(days=1), datetime.min.time())) - sleep_start) / 2
            fig.add_annotation(
                x=label_x,
                y=0.1,
                yref="paper",
                text=sleep_text,
                showarrow=False,
                font=dict(color="cornflowerblue", size=12),
                bgcolor="rgba(0,0,0,0.5)",
            )
    
    # Add exercise data if available
    if 'activities' in data and not data['activities'].empty:
        for _, activity in data['activities'].iterrows():
            # Calculate start and end times
            if 'startTimeLocal' in activity and 'duration' in activity:
                try:
                    start_time = pd.to_datetime(activity['startTimeLocal'])
                    end_time = start_time + pd.Timedelta(seconds=activity['duration'])
                    
                    # Only include activities that fall on the selected date
                    if start_time.date() == selected_date.date() or end_time.date() == selected_date.date():
                        # Add exercise rectangle
                        fig.add_shape(
                            type="rect",
                            xref="x", yref="paper",
                            x0=start_time, x1=end_time, y0=0.25, y1=0.45,
                            fillcolor="rgba(50, 205, 50, 0.3)",
                            line=dict(color="limegreen", width=1),
                            layer="below"
                        )
                        
                        # Add exercise label
                        duration_mins = int(activity['duration'] / 60)
                        activity_type = activity['activityType'] if 'activityType' in activity else 'Exercise'
                        fig.add_annotation(
                            x=start_time + (end_time - start_time) / 2,
                            y=0.35,
                            yref="paper",
                            text=f"{activity_type} ({duration_mins} min)",
                            showarrow=False,
                            font=dict(color="limegreen", size=12),
                            bgcolor="rgba(0,0,0,0.5)",
                        )
                except Exception as e:
                    print(f"Error processing activity: {e}")
                    continue
    
    # Add meal data if available
    if 'meals' in data and not data['meals'].empty:
        # Group meals by meal type
        for meal_type, meal_group in data['meals'].groupby('meal'):
            # Use color based on meal type
            meal_colors = {
                'breakfast': 'rgb(255, 165, 0)',  # orange
                'lunch': 'rgb(255, 215, 0)',      # gold
                'dinner': 'rgb(138, 43, 226)',    # purple
                'snacks': 'rgb(220, 20, 60)'      # crimson
            }
            
            color = meal_colors.get(meal_type, 'gray')
            
            # Calculate approximate meal time based on typical times
            meal_times = {
                'breakfast': datetime.combine(selected_date.date(), datetime.strptime('08:00', '%H:%M').time()),
                'lunch': datetime.combine(selected_date.date(), datetime.strptime('13:00', '%H:%M').time()),
                'dinner': datetime.combine(selected_date.date(), datetime.strptime('20:00', '%H:%M').time()),
                'snacks': datetime.combine(selected_date.date(), datetime.strptime('16:00', '%H:%M').time()),
            }
            
            meal_time = meal_times.get(meal_type)
            
            if meal_time:
                # Sum up the nutritional values
                total_calories = meal_group['calories'].sum()
                total_carbs = meal_group['carbs'].sum()
                
                # Add marker for the meal
                fig.add_trace(go.Scatter(
                    x=[meal_time],
                    y=[60],  # Position near bottom of chart
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color=color
                    ),
                    name=f"{meal_type.capitalize()} ({int(total_calories)} cal, {int(total_carbs)}g carbs)",
                    hoverinfo='text',
                    hovertext=f"{meal_type.capitalize()}<br>{int(total_calories)} calories<br>{int(total_carbs)}g carbs"
                ))
                
                # Add vertical line at meal time
                fig.add_shape(
                    type="line",
                    xref="x", yref="paper",
                    x0=meal_time, x1=meal_time, y0=0, y1=1,
                    line=dict(color=color, width=1, dash="dot"),
                )
    
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