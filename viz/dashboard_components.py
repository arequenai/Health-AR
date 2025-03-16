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

def display_metric_box(title: str, primary: Dict[str, Any], 
                      secondary1: Dict[str, Any], secondary2: Dict[str, Any]) -> None:
    """Display a metric box with primary and secondary metrics."""
    color_value = primary.get('color_value', primary['value'])
    level = get_metric_level(title.lower(), color_value)
    with st.container():
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-values">
                    <div class="metric-group">
                        <div class="primary-metric level-{level}">{primary['value']}</div>
                        <div class="metric-label">{primary['label']}</div>
                    </div>
                    <div class="metric-group">
                        <div class="secondary-metric">{secondary1['value']}</div>
                        <div class="metric-label">{secondary1['label']}</div>
                    </div>
                    <div class="metric-group">
                        <div class="secondary-metric">{secondary2['value']}</div>
                        <div class="metric-label">{secondary2['label']}</div>
                    </div>
                </div>
                <div class="metric-title">{title}</div>
            </div>
        """, unsafe_allow_html=True) 