import streamlit as st
from typing import Dict, Any

def get_metric_level(metric_name: str, value: float | str) -> int:
    """Determine color level based on metric value."""
    from viz.metric_thresholds import METRIC_THRESHOLDS
    thresholds = METRIC_THRESHOLDS[metric_name]
    
    # Special handling for nutrition where lower is better
    if metric_name == 'nutrition':
        for level in range(5, 0, -1):
            if float(value) <= float(thresholds[str(level)]):
                return level
        return 1
    
    # Special handling for glucose where a range is ideal (not too high, not too low)
    if metric_name == 'glucose':
        # For glucose, lower values in the thresholds are better
        for level in range(5, 0, -1):
            if float(value) <= float(thresholds[str(level)]):
                return level
        return 1
    
    # Normal handling for other metrics where higher is better
    for level in range(5, 0, -1):
        if float(value) >= float(thresholds[str(level)]):
            return level
    return 1

def get_secondary_metric_level(metric_name: str, metric_key: str, value: float | str) -> str:
    """Determine secondary metric level (good, bad, or normal)."""
    from viz.metric_thresholds import SECONDARY_METRIC_THRESHOLDS
    
    # Skip if value is not numeric or not in thresholds
    if value == '-' or f"{metric_name}.{metric_key}" not in SECONDARY_METRIC_THRESHOLDS:
        return "normal"
    
    try:
        thresholds = SECONDARY_METRIC_THRESHOLDS[f"{metric_name}.{metric_key}"]
        
        # Special handling for time format (e.g., "7:30" for sleep hours)
        if metric_key == "hrs in bed" and ":" in str(value):
            hours, minutes = str(value).split(":")
            float_value = float(hours) + float(minutes) / 60
        # Handle values that might be strings with decimal points
        else:
            # Replace comma with period for decimal representation if needed
            str_value = str(value).replace(',', '.')
            float_value = float(str_value)
        
        if "good_above" in thresholds and float_value >= thresholds["good_above"]:
            return "good"
        elif "good_below" in thresholds and float_value <= thresholds["good_below"]:
            return "good"
        elif "bad_above" in thresholds and float_value >= thresholds["bad_above"]:
            return "bad"
        elif "bad_below" in thresholds and float_value <= thresholds["bad_below"]:
            return "bad"
        else:
            return "normal"
    except (ValueError, TypeError) as e:
        print(f"Error processing {metric_name}.{metric_key} value '{value}': {e}")
        return "normal"  # Default to normal if can't convert to float

def display_metric_box(title: str, primary: Dict[str, Any], 
                      secondary1: Dict[str, Any], secondary2: Dict[str, Any]) -> None:
    """Display a metric box with primary and secondary metrics."""
    color_value = primary.get('color_value', primary['value'])
    level = get_metric_level(title.lower(), color_value)
    
    # Get levels for secondary metrics
    sec1_level = get_secondary_metric_level(title.lower(), secondary1['label'], secondary1['value'])
    sec2_level = get_secondary_metric_level(title.lower(), secondary2['label'], secondary2['value'])
    
    with st.container():
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-values">
                    <div class="metric-group">
                        <div class="primary-metric level-{level}">{primary['value']}</div>
                        <div class="metric-label">{primary['label']}</div>
                    </div>
                    <div class="metric-group">
                        <div class="secondary-metric secondary-{sec1_level}">{secondary1['value']}</div>
                        <div class="metric-label">{secondary1['label']}</div>
                    </div>
                    <div class="metric-group">
                        <div class="secondary-metric secondary-{sec2_level}">{secondary2['value']}</div>
                        <div class="metric-label">{secondary2['label']}</div>
                    </div>
                </div>
                <div class="metric-title">{title}</div>
            </div>
        """, unsafe_allow_html=True) 