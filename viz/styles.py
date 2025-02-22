DASHBOARD_CSS = """
    <style>
    .metric-container {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 8px;
        margin: 4px 0px;
        width: 160px;  /* Fixed width */
        display: inline-block;  /* Make boxes flow side by side */
    }
    .primary-metric {
        font-size: 22px;
        font-weight: bold;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .secondary-metric {
        font-size: 12px;
        color: #888;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    h3 {
        font-size: 14px;
        margin: 0;
        padding: 0;
    }
    .level-5 { color: #2ECC40; }
    .level-4 { color: #01FF70; }
    .level-3 { color: #FFDC00; }
    .level-2 { color: #FF851B; }
    .level-1 { color: #FF4136; }
    [data-testid="stHorizontalBlock"] {
        gap: 0rem;
        padding: 0 !important;
        margin: 0 !important;
        justify-content: flex-start !important;  /* Align from left */
        width: 340px !important;  /* Fixed width for two boxes + spacing */
    }
    div[data-testid="column"] {
        padding: 0 !important;
        margin: 0 !important;
        width: 170px !important;  /* Fixed width for columns */
        flex: none !important;  /* Prevent flex resizing */
    }
    </style>
""" 