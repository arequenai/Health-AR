import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
from datetime import timedelta
import plotly.graph_objects as go
import sys
import plotly.graph_objects as go
import plotly.subplots as sp
import dash_core_components as dcc
import dash_html_components as html

# Add parent directory to sys.path to allow importing from etl module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from etl import config
from viz.metric_thresholds import METRIC_THRESHOLDS
from viz.styles import LEVEL_COLORS

# Define separate weekly thresholds for nutrition and CTL
WEEKLY_THRESHOLDS = {
    # For weekly average net calories (different from daily)
    'nutrition': {'5': -200, '4': -100, '3': -50, '2': 00, '1': 150}
}

def get_week_start_end(date):
    """Get the start and end of the ISO week for a given date."""
    start = date - timedelta(days=date.weekday())  # Monday
    end = start + timedelta(days=6)  # Sunday
    return start.date(), end.date()

def aggregate_weekly_data(data):
    """Aggregate data to weekly level."""
    weekly_metrics = {}
    
    # Process MFP data (nutrition)
    if 'nutrition' in data and not data['nutrition'].empty:
        df = data['nutrition'].copy()
        
        # Get today's date
        today = pd.Timestamp.now().date()
        
        # Add week number and year
        df['year_week'] = df['date'].apply(lambda x: f"{x.isocalendar()[0]}-W{x.isocalendar()[1]:02d}")
        df['week_start'], df['week_end'] = zip(*df['date'].apply(get_week_start_end))
        
        # Identify current week
        current_week_start, current_week_end = get_week_start_end(pd.Timestamp.now())
        current_week_id = f"{today.year}-W{today.isocalendar()[1]:02d}"
        
        # Create a filtered dataframe excluding today's data for the current week
        filtered_df = df[~((df['date'].dt.date == today) & (df['year_week'] == current_week_id))]
        
        # Group by week (all weeks)
        all_weekly = df.groupby('year_week').agg(
            avg_calories_net=('calories_net', 'mean'),
            avg_protein=('protein', 'mean'),
            num_days=('date', 'count'),
            week_start=('week_start', 'first'),
            week_end=('week_end', 'first')
        ).reset_index()
        
        # Group by week for filtered data (current week has today excluded)
        filtered_weekly = filtered_df.groupby('year_week').agg(
            avg_calories_net=('calories_net', 'mean'),
            avg_protein=('protein', 'mean'),
            num_days=('date', 'count'),
            week_start=('week_start', 'first'),
            week_end=('week_end', 'first')
        ).reset_index()
        
        # Replace current week data with filtered data
        if current_week_id in all_weekly['year_week'].values:
            # Get index of current week
            idx = all_weekly[all_weekly['year_week'] == current_week_id].index[0]
            
            # If current week exists in filtered data, replace with filtered values
            if current_week_id in filtered_weekly['year_week'].values:
                filtered_current = filtered_weekly[filtered_weekly['year_week'] == current_week_id].iloc[0]
                all_weekly.loc[idx, 'avg_calories_net'] = filtered_current['avg_calories_net']
                all_weekly.loc[idx, 'avg_protein'] = filtered_current['avg_protein']
                all_weekly.loc[idx, 'num_days'] = filtered_current['num_days']
        
        weekly_metrics['nutrition'] = all_weekly
    
    # Process Garmin data (recovery, sleep, running)
    if 'garmin' in data and not data['garmin'].empty:
        df = data['garmin'].copy()
        # Add week number and year
        df['year_week'] = df['date'].apply(lambda x: f"{x.isocalendar()[0]}-W{x.isocalendar()[1]:02d}")
        df['week_start'], df['week_end'] = zip(*df['date'].apply(get_week_start_end))
        
        # Group by week
        weekly_garmin = df.groupby('year_week').agg(
            avg_max_battery=('max_sleep_body_battery', 'mean'),
            avg_sleep_score=('sleep_score', 'mean'),
            # Keep CTL for backward compatibility
            end_of_week_CTL=('CTL', lambda x: x.iloc[-1] if not x.empty else np.nan),
            week_start=('week_start', 'first'),
            week_end=('week_end', 'first')
        ).reset_index()
        
        weekly_metrics['garmin'] = weekly_garmin
    
    # Process Running data
    if 'running' in data and not data['running'].empty:
        df = data['running'].copy()
        
        # Add week number and year
        df['year_week'] = df['date'].apply(lambda x: f"{x.isocalendar()[0]}-W{x.isocalendar()[1]:02d}")
        df['week_start'], df['week_end'] = zip(*df['date'].apply(get_week_start_end))
        
        # Group by week
        weekly_running = df.groupby('year_week').agg(
            total_km_run=('km_run', 'sum'),          # Total distance for the week
            total_minutes_run=('minutes_run', 'sum'),  # Total time for the week
            num_runs=('date', 'count'),               # Number of runs in the week
            week_start=('week_start', 'first'),
            week_end=('week_end', 'first')
        ).reset_index()
        
        weekly_metrics['running'] = weekly_running
    
    # Process Glucose data
    if 'glucose' in data and not data['glucose'].empty:
        df = data['glucose'].copy()
        # Add week number and year
        df['year_week'] = df['date'].apply(lambda x: f"{x.isocalendar()[0]}-W{x.isocalendar()[1]:02d}")
        df['week_start'], df['week_end'] = zip(*df['date'].apply(get_week_start_end))
        
        # Aggregate by week
        weekly_glucose = df.groupby('year_week').agg(
            avg_mean_glucose=('mean_glucose', 'mean'),
            avg_fasting_glucose=('fasting_glucose', 'mean'),
            week_start=('week_start', 'first'),
            week_end=('week_end', 'first')
        ).reset_index()
        
        weekly_metrics['glucose'] = weekly_glucose
    
    # Process Activities data for strength metrics
    if 'strength' in data and not data['strength'].empty:
        df = data['strength'].copy()
        # Add week number and year
        df['year_week'] = df['date'].apply(lambda x: f"{x.isocalendar()[0]}-W{x.isocalendar()[1]:02d}")
        df['week_start'], df['week_end'] = zip(*df['date'].apply(get_week_start_end))
        
        # Aggregate by week
        weekly_strength = df.groupby('year_week').agg(
            total_strength_minutes=('strength_minutes', 'sum'),
            week_start=('week_start', 'first'),
            week_end=('week_end', 'first')
        ).reset_index()
        
        weekly_metrics['strength'] = weekly_strength
    
    return weekly_metrics

def get_metric_level(value, metric_type, is_weekly=False):
    """Determine the level (1-5) of a metric value based on thresholds."""
    # Use weekly thresholds if available and is_weekly is True
    if is_weekly and metric_type in WEEKLY_THRESHOLDS:
        thresholds = WEEKLY_THRESHOLDS[metric_type]
    elif metric_type in METRIC_THRESHOLDS:
        thresholds = METRIC_THRESHOLDS[metric_type]
    else:
        return 3  # Default to middle level if metric type not found
    
    # Special handling for metrics where lower is better (nutrition, glucose)
    if metric_type == 'nutrition':
        # For nutrition, negative is good (calorie deficit)
        if value <= float(thresholds['5']):
            level = 5  # Best - large deficit
        elif value <= float(thresholds['4']):
            level = 4
        elif value <= float(thresholds['3']):
            level = 3  # Small deficit
        elif value <= float(thresholds['2']):
            level = 2  # Small surplus
        elif value <= float(thresholds['1']):
            level = 2  
        else:
            level = 1  # Worst - large surplus
        
        return level
    elif metric_type == 'glucose':
        if value <= float(thresholds['5']):
            return 5
        elif value <= float(thresholds['4']):
            return 4
        elif value <= float(thresholds['3']):
            return 3
        elif value <= float(thresholds['2']):
            return 2
        else:
            return 1
    # Running has special logic depending on whether it's TSB (daily) or CTL (weekly)
    elif metric_type == 'running':
        if is_weekly:  # For CTL, higher is better
            if value >= float(thresholds['5']):
                return 5
            elif value >= float(thresholds['4']):
                return 4
            elif value >= float(thresholds['3']):
                return 3
            elif value >= float(thresholds['2']):
                return 2
            else:
                return 1
        else:  # For TSB (daily), handle normally (higher TSB is better)
            if value >= float(thresholds['5']):
                return 5
            elif value >= float(thresholds['4']):
                return 4
            elif value >= float(thresholds['3']):
                return 3
            elif value >= float(thresholds['2']):
                return 2
            else:
                return 1
    # For metrics where higher is better (recovery, sleep, strength)
    else:
        if value >= float(thresholds['5']):
            return 5
        elif value >= float(thresholds['4']):
            return 4
        elif value >= float(thresholds['3']):
            return 3
        elif value >= float(thresholds['2']):
            return 2
        else:
            return 1

def create_weekly_metric_chart(df, metric_col, title, y_label, metric_type, num_weeks=8):
    """Create a simplified bar chart for weekly metrics with level-based coloring."""
    if df is None or df.empty:
        return None
    
    # Sort by week
    df = df.sort_values('week_start')
    
    # Limit to the last N weeks
    if len(df) > num_weeks:
        df = df.tail(num_weeks)
    
    # Ensure dates are datetime objects for formatting
    if not isinstance(df['week_start'].iloc[0], datetime.datetime):
        df['week_start'] = pd.to_datetime(df['week_start'])
    if not isinstance(df['week_end'].iloc[0], datetime.datetime):
        df['week_end'] = pd.to_datetime(df['week_end'])
    
    # Simplify week labels: Just show week number
    df['week_label'] = df['week_start'].dt.strftime('W%V')  # ISO week number
    
    # Set default level for all metrics
    df['level'] = 3  # Default level (middle)
    
    # Don't apply coloring to running distance
    if metric_type == 'running_distance':
        # Use a neutral color for all running distance bars
        df['color'] = '#808080'  # Gray color for all bars
    elif metric_type == 'strength':
        # Special handling for strength using fixed thresholds
        strength_thresholds = {'5': 180, '4': 150, '3': 120, '2': 60, '1': 0}
        
        def get_strength_level(value):
            if value >= float(strength_thresholds['5']):
                return 5
            elif value >= float(strength_thresholds['4']):
                return 4
            elif value >= float(strength_thresholds['3']):
                return 3
            elif value >= float(strength_thresholds['2']):
                return 2
            else:
                return 1
        
        df['level'] = df[metric_col].apply(get_strength_level)
        df['color'] = df['level'].apply(lambda x: LEVEL_COLORS[x])
    else:
        # Normal coloring based on thresholds for other metrics
        df['level'] = df[metric_col].apply(lambda x: get_metric_level(x, metric_type, is_weekly=True))
        df['color'] = df['level'].apply(lambda x: LEVEL_COLORS[x])
    
    # Convert level to string to ensure it's treated as categorical
    df['level_str'] = df['level'].astype(str)
    
    # Round values to integers for display
    df[f'{metric_col}_int'] = df[metric_col].round().astype(int)
    
    # Determine if current week is in progress
    today = datetime.datetime.now().date()
    current_week_start, current_week_end = get_week_start_end(datetime.datetime.now())
    
    # Mark current week (just for reference, not visual differentiation)
    df['is_current'] = df['week_start'].dt.date.apply(lambda x: x == current_week_start)
    
    # Include metric in title
    enhanced_title = f"{title} ({y_label})"
    
    # Create a simple bar chart first without colors
    fig = go.Figure()
    
    # Add bars one by one with custom colors - all same size
    for i, row in df.iterrows():
        fig.add_trace(go.Bar(
            x=[row['week_label']],
            y=[row[metric_col]],
            text=[row[f'{metric_col}_int']],
            name=f"Level {row['level']}",
            marker_color=row['color'],
            showlegend=False  # Don't show legend for each bar
        ))
    
    # Set the title and labels with simplified layout
    fig.update_layout(
        title={
            'text': enhanced_title,
            'font': {'size': 20}  # Larger title font
        },
        height=400,
        xaxis={
            'type': 'category',
            'categoryorder': 'array',
            'categoryarray': df['week_label'].tolist(),
            'tickfont': {'size': 14},  # Larger x-axis tick font
            'showgrid': False,  # Remove vertical grid lines
        },
        yaxis={
            'visible': False,  # Hide y-axis
            'showgrid': False,  # Remove horizontal grid lines
        },
        bargap=0.3,  # Increase gap between bars for better readability
        margin=dict(l=20, r=20, t=60, b=40),  # Adjust margins
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        showlegend=False  # Hide legend completely
    )
    
    # Format text to show integers with larger font
    fig.update_traces(
        texttemplate='%{text}',
        textposition='inside',
        textfont=dict(size=16)  # Larger bar text
    )
    
    return fig

def create_weekly_summary_table(weekly_metrics, num_weeks=8):
    """Create a summary table for weekly metrics."""
    # Define metrics mapping
    metrics_mapping = {
        'nutrition': {'df': 'nutrition', 'col': 'avg_calories_net', 'name': 'Net Calories', 'better': 'lower'},
        'glucose': {'df': 'glucose', 'col': 'avg_mean_glucose', 'name': 'Glucose', 'better': 'lower'},
        'recovery': {'df': 'garmin', 'col': 'avg_max_battery', 'name': 'Max Battery', 'better': 'higher'},
        'sleep': {'df': 'garmin', 'col': 'avg_sleep_score', 'name': 'Sleep Score', 'better': 'higher'},
        'running': {'df': 'running', 'col': 'total_km_run', 'name': 'KM Run', 'better': 'higher', 'type': 'running_distance'},
        'strength': {'df': 'strength', 'col': 'total_strength_minutes', 'name': 'Strength Minutes', 'better': 'higher'}
    }
    
    # Get all week labels from the first available metric
    week_labels = None
    for metric_info in metrics_mapping.values():
        if metric_info['df'] in weekly_metrics and not weekly_metrics[metric_info['df']].empty:
            df = weekly_metrics[metric_info['df']].copy().sort_values('week_start')
            if len(df) > num_weeks:
                df = df.tail(num_weeks)
            # Ensure week_start is datetime
            df['week_start'] = pd.to_datetime(df['week_start'])
            week_labels = df['week_start'].dt.strftime('W%V').tolist()
            break
    
    if not week_labels:
        return None
    
    # Initialize summary data with metric names
    summary_data = {metric_info['name']: [] for metric_info in metrics_mapping.values()}
    
    # Add data for each metric
    for metric_key, metric_info in metrics_mapping.items():
        if metric_info['df'] in weekly_metrics and not weekly_metrics[metric_info['df']].empty:
            df = weekly_metrics[metric_info['df']].copy().sort_values('week_start')
            if len(df) > num_weeks:
                df = df.tail(num_weeks)
            # Ensure week_start is datetime
            df['week_start'] = pd.to_datetime(df['week_start'])
            
            # Add values for each week
            for week in week_labels:
                week_data = df[df['week_start'].dt.strftime('W%V') == week]
                if not week_data.empty:
                    value = week_data.iloc[0][metric_info['col']]
                    summary_data[metric_info['name']].append(int(round(value)))
                else:
                    summary_data[metric_info['name']].append(None)
    
    # Create DataFrame and transpose it
    summary_df = pd.DataFrame(summary_data, index=week_labels).T
    
    return summary_df

def color_weekly_cells(data):
    """Apply color styling to the weekly summary table."""
    metrics_mapping = {
        'nutrition': {'df': 'nutrition', 'col': 'avg_calories_net', 'name': 'Net Calories', 'better': 'lower'},
        'glucose': {'df': 'glucose', 'col': 'avg_mean_glucose', 'name': 'Glucose', 'better': 'lower'},
        'recovery': {'df': 'garmin', 'col': 'avg_max_battery', 'name': 'Max Battery', 'better': 'higher'},
        'sleep': {'df': 'garmin', 'col': 'avg_sleep_score', 'name': 'Sleep Score', 'better': 'higher'},
        'running': {'df': 'running', 'col': 'total_km_run', 'name': 'KM Run', 'better': 'higher', 'type': 'running_distance'},
        'strength': {'df': 'strength', 'col': 'total_strength_minutes', 'name': 'Strength Minutes', 'better': 'higher'}
    }
    
    colored = pd.DataFrame('', index=data.index, columns=data.columns)
    
    # Override the broken strength thresholds with proper ones
    strength_thresholds = {'5': 180, '4': 150, '3': 120, '2': 60, '1': 0}
    
    for metric_name in data.index:
        # Find the metric type
        metric_type = next((k for k, v in metrics_mapping.items() if v['name'] == metric_name), None)
        if not metric_type:
            continue
        
        # Skip coloring for KM Run
        if metric_name == 'KM Run':
            continue
        
        # Apply normal coloring for all metrics except running
        if metric_type == 'strength':
            # Special handling for strength using fixed thresholds
            for col_idx, val in enumerate(data.loc[metric_name]):
                if pd.isna(val):
                    continue
                
                # Apply fixed strength thresholds
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
                
                colored.loc[metric_name, colored.columns[col_idx]] = f'background-color: {LEVEL_COLORS[level]}'
        else:
            # Standard level-based coloring for other metrics
            for col_idx, val in enumerate(data.loc[metric_name]):
                if pd.isna(val):
                    continue
                
                level = get_metric_level(val, metric_type, is_weekly=True)
                colored.loc[metric_name, colored.columns[col_idx]] = f'background-color: {LEVEL_COLORS[level]}'
    
    return colored

def format_value(val):
    """Format values for display in the table."""
    if pd.isna(val):
        return ''
    return f"{int(val)}"

def show_weekly_evolution(data, num_weeks=8):
    """Display the weekly evolution page with summary table and charts."""
    
    # Aggregate weekly data
    weekly_metrics = aggregate_weekly_data(data)
    
    # Create and display the weekly summary table
    summary_df = create_weekly_summary_table(weekly_metrics, num_weeks)
    
    if summary_df is not None:
        st.subheader("Weekly Metrics Summary")
        
        # Apply styling
        styled_df = summary_df.style.apply(lambda _: color_weekly_cells(summary_df), axis=None)
        styled_df = styled_df.format(format_value)
        
        # Display the table
        st.dataframe(styled_df)
    else:
        st.info("No weekly data available for the summary table.")
    
    # Define available metrics for the chart
    metrics_options = {
        'Net Calories': {'df': 'nutrition', 'col': 'avg_calories_net', 'title': 'Weekly Average Net Calories', 'unit': 'Calories (kcal)', 'type': 'nutrition'},
        'Glucose': {'df': 'glucose', 'col': 'avg_mean_glucose', 'title': 'Weekly Average Glucose', 'unit': 'Glucose (mg/dL)', 'type': 'glucose'},
        'Max Battery': {'df': 'garmin', 'col': 'avg_max_battery', 'title': 'Weekly Average Max Battery', 'unit': 'Battery Level', 'type': 'recovery'},
        'Sleep Score': {'df': 'garmin', 'col': 'avg_sleep_score', 'title': 'Weekly Average Sleep Score', 'unit': 'Sleep Score', 'type': 'sleep'},
        'Running Distance': {'df': 'running', 'col': 'total_km_run', 'title': 'Weekly Running Distance', 'unit': 'Kilometers', 'type': 'running_distance'},
        'Strength Minutes': {'df': 'strength', 'col': 'total_strength_minutes', 'title': 'Weekly Strength Training Minutes', 'unit': 'Minutes', 'type': 'strength'}
    }
    
    # Add dropdown to select metric for detailed view
    selected_metric = st.selectbox(
        "Select metric to display",
        options=list(metrics_options.keys()),
        index=0
    )
    
    # Get the selected metric configuration
    metric_config = metrics_options[selected_metric]
    
    # Display the selected chart
    if metric_config['df'] in weekly_metrics:
        fig = create_weekly_metric_chart(
            weekly_metrics[metric_config['df']],
            metric_config['col'],
            metric_config['title'],
            metric_config['unit'],
            metric_config['type'],
            num_weeks=num_weeks
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No data available for {selected_metric}") 