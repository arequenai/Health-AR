import streamlit as st
from typing import Dict, Any

def get_metric_level(metric_name: str, value: float) -> int:
    """Determine color level based on metric value."""
    from viz.metric_thresholds import METRIC_THRESHOLDS
    thresholds = METRIC_THRESHOLDS[metric_name]
    
    for level in range(5, 0, -1):
        if float(value) >= float(thresholds[str(level)]):
            return level
    return 1

def display_metric_box(title: str, primary: Dict[str, Any], 
                      secondary1: Dict[str, Any], secondary2: Dict[str, Any]) -> None:
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