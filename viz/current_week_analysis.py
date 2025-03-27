import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
from datetime import timedelta
import plotly.graph_objects as go
import sys

# Add parent directory to sys.path to allow importing from etl module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from etl import config
from viz.metric_thresholds import METRIC_THRESHOLDS
from viz.styles import LEVEL_COLORS
from viz.weekly_evolution import get_week_start_end, get_metric_level

def gather_comparison_data(data, num_weeks=8):
    """
    Gather data for daily comparison:
    - Current week values for each day
    - Historical average values for the same day of week
    """
    # Define the metrics we want to include
    metrics = {
        'nutrition': {'df': 'nutrition', 'col': 'calories_net', 'name': 'Net Calories', 'better': 'lower'},
        'glucose': {'df': 'glucose', 'col': 'mean_glucose', 'name': 'Glucose', 'better': 'lower'},
        'recovery': {'df': 'garmin', 'col': 'max_battery', 'name': 'Max Battery', 'better': 'higher'},
        'sleep': {'df': 'garmin', 'col': 'sleep_score', 'name': 'Sleep Score', 'better': 'higher'},
        'running': {'df': 'running', 'col': 'km_run', 'name': 'KM Run', 'better': 'higher'},
        'strength': {'df': 'strength', 'col': 'strength_minutes', 'name': 'Strength Minutes', 'better': 'higher'}
    }
    
    # Get current week date range
    today = datetime.datetime.now().date()
    current_week_start, current_week_end = get_week_start_end(datetime.datetime.now())
    
    # Get all dates in the past num_weeks
    end_date = min(today, current_week_end)
    start_date = (datetime.datetime.combine(end_date, datetime.time.min) - 
                  datetime.timedelta(days=num_weeks * 7)).date()
    
    # Create empty DataFrames to hold our results
    current_week_data = pd.DataFrame(columns=['day_of_week', 'day_name'] + 
                                     [m['name'] for m in metrics.values()])
    historical_avg_data = pd.DataFrame(columns=['day_of_week', 'day_name'] + 
                                       [m['name'] for m in metrics.values()])
    
    # Process each day of the week
    for day_offset in range(7):
        day_date = current_week_start + datetime.timedelta(days=day_offset)
        day_of_week = day_date.weekday()  # 0=Monday, 1=Tuesday, etc.
        day_name = day_date.strftime('%A')
        
        if day_date > end_date:
            continue  # Skip future days
        
        # Initialize data row for current week
        current_row = {'day_of_week': day_of_week, 'day_name': day_name}
        historical_row = {'day_of_week': day_of_week, 'day_name': day_name}
        
        # Process each metric
        for metric_key, metric_info in metrics.items():
            df_key = metric_info['df']
            col_name = metric_info['col']
            metric_name = metric_info['name']
            
            if df_key in data:
                df = data[df_key].copy()
                
                # Current week data
                current_day_data = df[df['date'].dt.date == day_date]
                if not current_day_data.empty and col_name in current_day_data.columns:
                    current_row[metric_name] = current_day_data[col_name].mean()
                else:
                    # For strength and running, explicitly set to 0 if no data (rather than None)
                    if metric_key in ['strength', 'running'] and day_date <= today:
                        current_row[metric_name] = 0
                    else:
                        current_row[metric_name] = None
                
                # Historical data (excluding current week)
                historical_data = df[(df['date'].dt.date >= start_date) & 
                                    (df['date'].dt.date < current_week_start) & 
                                    (df['date'].dt.weekday == day_of_week)]
                
                if not historical_data.empty and col_name in historical_data.columns:
                    # Special handling for strength and running - treat missing days as 0
                    if metric_key in ['strength', 'running']:
                        # Get all dates for this day of week in the historical period
                        all_dates = pd.date_range(start=start_date, end=current_week_start - datetime.timedelta(days=1))
                        all_dates = [d.date() for d in all_dates if d.weekday() == day_of_week]
                        
                        # Calculate how many of these dates have data
                        dates_with_data = historical_data['date'].dt.date.unique()
                        total_expected_dates = len(all_dates)
                        
                        if total_expected_dates > 0:
                            # If we have some data, calculate a weighted average including 0 for missing days
                            total_value = historical_data[col_name].sum()
                            historical_row[metric_name] = total_value / total_expected_dates
                        else:
                            historical_row[metric_name] = None
                    else:
                        historical_row[metric_name] = historical_data[col_name].mean()
                else:
                    # For strength and running, explicitly set to 0 if no data (rather than None)
                    if metric_key in ['strength', 'running']:
                        # Only set to 0 if we expect data for this day
                        all_dates = pd.date_range(start=start_date, end=current_week_start - datetime.timedelta(days=1))
                        all_dates = [d.date() for d in all_dates if d.weekday() == day_of_week]
                        if len(all_dates) > 0:
                            historical_row[metric_name] = 0
                        else:
                            historical_row[metric_name] = None
        
        # Add rows to DataFrames
        current_week_data = pd.concat([current_week_data, pd.DataFrame([current_row])], ignore_index=True)
        historical_avg_data = pd.concat([historical_avg_data, pd.DataFrame([historical_row])], ignore_index=True)
    
    return current_week_data, historical_avg_data

def create_comparison_table(current_data, historical_data):
    """Create a comparison table of current week vs historical averages."""
    if current_data.empty or historical_data.empty:
        return None
    
    # Define metrics for the table
    metrics = ['Net Calories', 'Glucose', 'Max Battery', 'Sleep Score', 'KM Run', 'Strength Minutes']
    
    # Define whether higher or lower values are better for each metric
    better_direction = {
        'Net Calories': 'lower',  # Lower net calories is better (calorie deficit)
        'Glucose': 'lower',       # Lower glucose is better
        'Max Battery': 'higher',  # Higher battery is better
        'Sleep Score': 'higher',  # Higher sleep score is better
        'KM Run': 'higher',        # More KM Run is better
        'Strength Minutes': 'higher'  # More strength training is better
    }
    
    # Merge the data into a single DataFrame for easier comparison
    comparison_data = []
    
    for day_name in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        current_day = current_data[current_data['day_name'] == day_name]
        historical_day = historical_data[historical_data['day_name'] == day_name]
        
        if current_day.empty and historical_day.empty:
            continue
            
        day_data = {"Day": day_name}
        
        for metric in metrics:
            # Get current and historical values
            current_val = None if current_day.empty else current_day.iloc[0].get(metric)
            hist_val = None if historical_day.empty else historical_day.iloc[0].get(metric)
            
            # Calculate difference and percentage change
            if current_val is not None and hist_val is not None and hist_val != 0:
                diff = current_val - hist_val
                pct_change = (diff / abs(hist_val)) * 100
                
                # Determine if the change is good or bad
                if better_direction[metric] == 'lower':
                    is_better = diff < 0
                else:
                    is_better = diff > 0
                
                # Format values
                day_data[f"{metric} (Current)"] = current_val
                day_data[f"{metric} (Hist Avg)"] = hist_val
                day_data[f"{metric} (% Change)"] = pct_change
                day_data[f"{metric} (Better)"] = is_better
            else:
                # Handle missing data
                day_data[f"{metric} (Current)"] = current_val
                day_data[f"{metric} (Hist Avg)"] = hist_val
                day_data[f"{metric} (% Change)"] = None
                day_data[f"{metric} (Better)"] = None
        
        comparison_data.append(day_data)
    
    return pd.DataFrame(comparison_data)

def format_comparison_table(df):
    """Format the comparison table with colors and styling."""
    if df is None or df.empty:
        return None
    
    # Define metrics for styling
    metrics = ['Net Calories', 'Glucose', 'Max Battery', 'Sleep Score', 'KM Run', 'Strength Minutes']
    
    # Create a copy of the DataFrame for styling
    styled_df = df.copy()
    
    # Function to apply styling based on better/worse comparison
    def style_cell(val, metric, col_type, is_better):
        if pd.isna(val):
            return ''
        
        if col_type == 'Current' or col_type == 'Hist Avg':
            # Format values as integers
            return f"{int(round(val))}"
        elif col_type == '% Change':
            # Format as percentage with arrow indicator - no HTML needed
            arrow = "↑" if is_better else "↓"
            return f"{arrow} {val:.1f}%"
    
    # Apply styling to each metric column
    for metric in metrics:
        for i, row in df.iterrows():
            current_val = row.get(f"{metric} (Current)")
            hist_val = row.get(f"{metric} (Hist Avg)")
            pct_change = row.get(f"{metric} (% Change)")
            is_better = row.get(f"{metric} (Better)")
            
            if not pd.isna(current_val):
                styled_df.at[i, f"{metric} (Current)"] = style_cell(current_val, metric, 'Current', is_better)
            
            if not pd.isna(hist_val):
                styled_df.at[i, f"{metric} (Hist Avg)"] = style_cell(hist_val, metric, 'Hist Avg', is_better)
            
            if not pd.isna(pct_change):
                styled_df.at[i, f"{metric} (% Change)"] = style_cell(pct_change, metric, '% Change', is_better)
                # Also store whether this change is better
                styled_df.at[i, f"{metric} (Better Color)"] = is_better
    
    # Drop the "Better" columns as they're just for calculation
    for metric in metrics:
        if f"{metric} (Better)" in styled_df.columns:
            styled_df = styled_df.drop(columns=[f"{metric} (Better)"])
    
    return styled_df

def create_daily_table(data, current_week_start, current_week_end):
    """Create a table showing daily metrics for the current week."""
    # Define metrics mapping
    metrics_mapping = {
        'nutrition': {'df': 'nutrition', 'col': 'calories_net', 'name': 'Net Calories', 'better': 'lower', 'type': 'nutrition'},
        'glucose': {'df': 'glucose', 'col': 'mean_glucose', 'name': 'Glucose', 'better': 'lower', 'type': 'glucose'},
        'recovery': {'df': 'garmin', 'col': 'max_battery', 'name': 'Max Battery', 'better': 'higher', 'type': 'recovery'},
        'sleep': {'df': 'garmin', 'col': 'sleep_score', 'name': 'Sleep Score', 'better': 'higher', 'type': 'sleep'},
        'running': {'df': 'running', 'col': 'km_run', 'name': 'KM Run', 'better': 'higher', 'type': 'running_distance'},
        'strength': {'df': 'strength', 'col': 'strength_minutes', 'name': 'Strength Minutes', 'better': 'higher', 'type': 'strength'}
    }
    
    # Get today's date
    today = datetime.datetime.now().date()
    
    # Generate list of dates in the current week up to today
    date_range = pd.date_range(start=current_week_start, end=min(today, current_week_end))
    dates_list = [d.date() for d in date_range]
    
    # Create a DataFrame with metric names as index
    metric_names = [m['name'] for m in metrics_mapping.values()]
    days_df = pd.DataFrame(index=metric_names)
    
    # Create column for each day with abbreviated weekday name
    for date in dates_list:
        days_df[date.strftime('%a')] = None
    
    # Extract and fill daily data
    daily_data = {}
    
    for metric_key, metric_info in metrics_mapping.items():
        df_key = metric_info['df']
        
        if df_key in data:
            # Get raw daily data for this metric
            source_df = data[df_key].copy()
            
            # Filter for current week
            current_week_df = source_df[(source_df['date'].dt.date >= current_week_start) & 
                                      (source_df['date'].dt.date <= min(today, current_week_end))]
            
            if not current_week_df.empty:
                # Get the relevant column for this metric
                col_name = metric_info['col']
                
                # Special handling for different metrics
                if col_name not in current_week_df.columns and df_key == 'strength':
                    # For strength, create daily duration in minutes
                    strength_df = data['activities'].copy() if 'activities' in data else None
                    if strength_df is not None:
                        strength_types = ['strength_training', 'indoor_cardio', 'fitness_equipment', 'training', 'other']
                        strength_df = strength_df[strength_df['type'].str.lower().isin(strength_types)]
                        strength_df = strength_df[(strength_df['date'].dt.date >= current_week_start) & 
                                                (strength_df['date'].dt.date <= min(today, current_week_end))]
                        
                        # Aggregate by day
                        if not strength_df.empty:
                            daily_strength = strength_df.groupby(strength_df['date'].dt.date).agg(
                                strength_minutes=('duration', lambda x: (x.sum() / 60))  # Convert seconds to minutes
                            ).reset_index()
                            daily_strength.columns = ['date', 'strength_minutes']
                            daily_strength['date'] = pd.to_datetime(daily_strength['date'])
                            daily_data[metric_info['name']] = daily_strength
                elif col_name in current_week_df.columns:
                    # Standard metric
                    daily_metric = current_week_df[['date', col_name]].copy()
                    daily_data[metric_info['name']] = daily_metric
    
    # Fill in values from daily data
    for metric_name in metric_names:
        if metric_name in daily_data:
            metric_df = daily_data[metric_name]
            
            for date in dates_list:
                day_data = metric_df[metric_df['date'].dt.date == date]
                
                if not day_data.empty:
                    # Get value for this day
                    if metric_name == 'Strength Minutes':
                        val = day_data['strength_minutes'].iloc[0]
                    elif metric_name == 'KM Run':
                        val = day_data['km_run'].iloc[0]
                    else:
                        col_name = list(day_data.columns)[1]  # Second column has the value
                        val = day_data[col_name].iloc[0]
                    
                    # Round to integer
                    if not pd.isna(val):
                        days_df.at[metric_name, date.strftime('%a')] = round(val)
    
    # Store metric types for coloring
    metric_types = {m['name']: m['type'] for m in metrics_mapping.values()}
    
    return days_df, metric_types

def color_daily_cells(data, metric_types):
    """Apply color styling to the daily table."""
    colored = pd.DataFrame('', index=data.index, columns=data.columns)
    strength_thresholds = {'5': 60, '4': 45, '3': 30, '2': 15, '1': 0}
    
    for metric_name in data.index:
        # Find the metric type
        metric_type = metric_types.get(metric_name)
        if not metric_type:
            continue
            
        for col in data.columns:
            val = data.loc[metric_name, col]
            if pd.isna(val):
                continue
            
            if metric_name == 'Strength Minutes':
                # Apply fixed strength thresholds for daily values (adjusted for single day)
                if val >= float(strength_thresholds['5']):
                    level = 5
                elif val >= float(strength_thresholds['4']):
                    level = 4
                elif val >= float(strength_thresholds['3']):
                    level = 3
                elif val >= float(strength_thresholds['2']):
                    level = 2
                else:
                    level = 1
            else:
                # Use standard metric level function
                level = get_metric_level(val, metric_type, is_weekly=False)
                
            colored.loc[metric_name, col] = f'background-color: {LEVEL_COLORS[level]}'
    
    return colored

def format_value(val):
    """Format values for display in the table."""
    if pd.isna(val):
        return ''
    return f"{int(val)}"

def show_current_week_analysis(data, num_weeks=8):
    """Display the current week analysis page with comparison tables."""
    st.title("Current week")
    st.write("Compare this week's daily performance to historical averages")
    
    # Get current week date range
    today = datetime.datetime.now().date()
    current_week_start, current_week_end = get_week_start_end(datetime.datetime.now())
    
    # Create daily breakdown table
    days_df, metric_types = create_daily_table(data, current_week_start, current_week_end)
    
    # Display daily table
    if days_df is not None and not days_df.empty:
        st.subheader("Current Week Daily Values")
        
        # Apply styling
        styled_days_df = days_df.style.apply(lambda _: color_daily_cells(days_df, metric_types), axis=None)
        styled_days_df = styled_days_df.format(format_value)
        
        # Display the table
        st.dataframe(styled_days_df)
    else:
        st.info("No daily data available for the current week.")
    
    # Gather data for comparison
    current_week_data, historical_avg_data = gather_comparison_data(data, num_weeks)
    
    # Create and format comparison table
    comparison_df = create_comparison_table(current_week_data, historical_avg_data)
    styled_comparison_df = format_comparison_table(comparison_df) if comparison_df is not None else None
    
    if styled_comparison_df is not None:
        st.subheader("Daily Comparison")
        
        # Create a day selector as a dropdown
        days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        available_days = [day for day in days_of_week if day in styled_comparison_df['Day'].values]
        
        if available_days:
            # Find the most recent day to use as default
            weekday_to_idx = {day: idx for idx, day in enumerate(days_of_week)}
            available_days_idx = [weekday_to_idx[day] for day in available_days]
            
            # Get most recent day with data (closest to today's weekday)
            today_weekday_idx = today.weekday()
            
            # Sort available days by proximity to today (going backwards in time)
            # Formula: (today_idx - day_idx) % 7 gives days ago (0 = today, 1 = yesterday, etc.)
            days_ago = [(today_weekday_idx - idx) % 7 for idx in available_days_idx]
            sorted_available = [day for _, day in sorted(zip(days_ago, available_days))]
            
            # Default to the most recent day
            default_day = sorted_available[0] if sorted_available else available_days[0]
            default_idx = available_days.index(default_day)
            
            selected_day = st.selectbox("Select day to compare", available_days, index=default_idx)
            
            # Get data for the selected day
            day_data = styled_comparison_df[styled_comparison_df['Day'] == selected_day]
            
            if not day_data.empty:
                # Create a DataFrame with a row for each metric
                metrics = ['Net Calories', 'Glucose', 'Max Battery', 'Sleep Score', 'KM Run', 'Strength Minutes']
                
                # Create a clean DataFrame for display
                detail_df = pd.DataFrame(index=metrics, columns=['Current', 'Hist Avg', '% Change'])
                
                # Create a styling dataframe to store color information
                color_data = pd.DataFrame(index=metrics, columns=['Current', 'Hist Avg', '% Change'])
                
                # Fill in the data for each metric
                for metric in metrics:
                    current_col = f"{metric} (Current)"
                    hist_col = f"{metric} (Hist Avg)"
                    pct_col = f"{metric} (% Change)"
                    better_col = f"{metric} (Better Color)"
                    
                    # Get the values
                    current_val = day_data[current_col].values[0] if current_col in day_data.columns and not pd.isna(day_data[current_col].values[0]) else None
                    hist_val = day_data[hist_col].values[0] if hist_col in day_data.columns and not pd.isna(day_data[hist_col].values[0]) else None
                    pct_change = day_data[pct_col].values[0] if pct_col in day_data.columns and not pd.isna(day_data[pct_col].values[0]) else None
                    is_better = day_data[better_col].values[0] if better_col in day_data.columns and not pd.isna(day_data[better_col].values[0]) else None
                    
                    # Add to the display dataframe
                    detail_df.at[metric, 'Current'] = current_val if current_val is not None else ''
                    detail_df.at[metric, 'Hist Avg'] = hist_val if hist_val is not None else ''
                    
                    # Format percentage change
                    if pct_change is not None and is_better is not None:
                        arrow = "↑" if is_better else "↓"
                        # Check if pct_change is already a string
                        if isinstance(pct_change, str):
                            if '%' in pct_change:
                                # Check if already has an arrow
                                if pct_change.startswith('↑') or pct_change.startswith('↓'):
                                    # Already has arrow and percentage, use as is
                                    detail_df.at[metric, '% Change'] = pct_change
                                else:
                                    # Has percentage but no arrow
                                    detail_df.at[metric, '% Change'] = f"{arrow} {pct_change}"
                            else:
                                # Try to convert to float and format
                                try:
                                    pct_value = float(pct_change.replace(',', '.'))
                                    detail_df.at[metric, '% Change'] = f"{arrow} {pct_value:.1f}%"
                                except ValueError:
                                    # If conversion fails, just use as is
                                    detail_df.at[metric, '% Change'] = f"{arrow} {pct_change}%"
                        else:
                            # It's a numeric value, format as usual
                            detail_df.at[metric, '% Change'] = f"{arrow} {pct_change:.1f}%"
                        color_data.at[metric, '% Change'] = 'green' if is_better else 'red'
                    
                    # Calculate color levels for current and historical values
                    if current_val is not None:
                        # Get the numeric value for level calculation
                        try:
                            num_val = float(current_val) if isinstance(current_val, (int, float)) else float(''.join(c for c in str(current_val) if c.isdigit() or c == '-' or c == '.'))
                            
                            # Get appropriate level
                            if metric == 'Strength Minutes':
                                strength_thresholds = {'5': 60, '4': 45, '3': 30, '2': 15, '1': 0}
                                if num_val >= float(strength_thresholds['5']):
                                    level = 5
                                elif num_val >= float(strength_thresholds['4']):
                                    level = 4
                                elif num_val >= float(strength_thresholds['3']):
                                    level = 3
                                elif num_val >= float(strength_thresholds['2']):
                                    level = 2
                                else:
                                    level = 1
                            else:
                                metric_type = metric_types.get(metric)
                                level = get_metric_level(num_val, metric_type, is_weekly=False)
                            
                            color_data.at[metric, 'Current'] = LEVEL_COLORS[level]
                        except:
                            pass
                    
                    if hist_val is not None:
                        # Get the numeric value for level calculation
                        try:
                            num_val = float(hist_val) if isinstance(hist_val, (int, float)) else float(''.join(c for c in str(hist_val) if c.isdigit() or c == '-' or c == '.'))
                            
                            # Get appropriate level
                            if metric == 'Strength Minutes':
                                strength_thresholds = {'5': 60, '4': 45, '3': 30, '2': 15, '1': 0}
                                if num_val >= float(strength_thresholds['5']):
                                    level = 5
                                elif num_val >= float(strength_thresholds['4']):
                                    level = 4
                                elif num_val >= float(strength_thresholds['3']):
                                    level = 3
                                elif num_val >= float(strength_thresholds['2']):
                                    level = 2
                                else:
                                    level = 1
                            else:
                                metric_type = metric_types.get(metric)
                                level = get_metric_level(num_val, metric_type, is_weekly=False)
                            
                            color_data.at[metric, 'Hist Avg'] = LEVEL_COLORS[level]
                        except:
                            pass
                
                # Define the styler function to apply coloring
                def color_cells(val, column):
                    styles = []
                    for idx in val.index:
                        if column == '% Change':
                            if idx in color_data.index and pd.notna(color_data.loc[idx, column]):
                                styles.append(f'color: {color_data.loc[idx, column]}')
                            else:
                                styles.append('')
                        else:  # 'Current' or 'Hist Avg'
                            if idx in color_data.index and pd.notna(color_data.loc[idx, column]):
                                styles.append(f'background-color: {color_data.loc[idx, column]}')
                            else:
                                styles.append('')
                    return styles
                
                # Apply styling to each column
                def apply_styles(df):
                    return pd.DataFrame('', index=df.index, columns=df.columns).style\
                        .apply(lambda x: color_cells(x, 'Current'), axis=0, subset=['Current'])\
                        .apply(lambda x: color_cells(x, 'Hist Avg'), axis=0, subset=['Hist Avg'])\
                        .apply(lambda x: color_cells(x, '% Change'), axis=0, subset=['% Change'])
                
                # Format numeric values
                def format_numeric(val):
                    if pd.isna(val) or val == '':
                        return ''
                    # Skip values that are already formatted (like the % Change column)
                    if isinstance(val, str) and ('%' in val or '↑' in val or '↓' in val):
                        return val
                    return f"{int(round(float(val)))}"
                
                # Apply styling and display
                styled_detail_df = detail_df.style\
                    .apply(lambda x: color_cells(x, 'Current'), axis=0, subset=['Current'])\
                    .apply(lambda x: color_cells(x, 'Hist Avg'), axis=0, subset=['Hist Avg'])\
                    .apply(lambda x: color_cells(x, '% Change'), axis=0, subset=['% Change'])\
                    .format(format_numeric, subset=['Current', 'Hist Avg'])
                
                st.dataframe(styled_detail_df)
            else:
                st.info(f"No comparison data available for {selected_day}")
        else:
            st.info("No days available for comparison")
    else:
        st.info("No comparison data available.") 