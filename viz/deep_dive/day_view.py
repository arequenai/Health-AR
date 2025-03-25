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
        
        # Add sleep rectangle
        fig.add_shape(
            type="rect",
            xref="x", yref="paper",
            x0=sleep_start, x1=sleep_end, y0=0, y1=0.2,
            fillcolor="rgba(100, 149, 237, 0.3)",
            line=dict(color="cornflowerblue", width=1),
            layer="below"
        )
        
        # Add sleep label
        sleep_hours = int(sleep_duration)
        sleep_minutes = int((sleep_duration - sleep_hours) * 60)
        fig.add_annotation(
            x=sleep_start + (sleep_end - sleep_start) / 2,
            y=0.1,
            yref="paper",
            text=f"Sleep ({sleep_hours}h {sleep_minutes}m)",
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
            # Load available dates from glucose_daily.csv (or another reliable source)
            try:
                df_glucose = pd.read_csv('data/glucose_daily.csv')
                df_glucose['date'] = pd.to_datetime(df_glucose['date'])
                
                # Create a sorted list of dates
                available_dates = sorted(df_glucose['date'].unique(), reverse=True)
                
                # Create a dropdown for date selection
                selected_date = st.selectbox(
                    'Select a date to view:',
                    options=available_dates,
                    format_func=lambda x: x.strftime('%A, %B %d, %Y')
                )
            except Exception as e:
                # Fallback if glucose data isn't available
                try:
                    df_garmin = pd.read_csv('data/garmin_daily.csv')
                    df_garmin['date'] = pd.to_datetime(df_garmin['date'])
                    available_dates = sorted(df_garmin['date'].unique(), reverse=True)
                    selected_date = st.selectbox(
                        'Select a date to view:',
                        options=available_dates,
                        format_func=lambda x: x.strftime('%A, %B %d, %Y')
                    )
                except Exception as e:
                    # Last resort - show date picker with recent dates
                    selected_date = st.date_input(
                        'Select a date to view:',
                        value=datetime.now().date() - timedelta(days=1)
                    )
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
        if data['garmin'] is not None and 'sleepingSeconds' in data['garmin'] and not pd.isna(data['garmin']['sleepingSeconds']):
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