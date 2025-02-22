DASHBOARD_CSS = """
    <style>
    .metric-container {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 8px 12px;
        margin: 2px 0;
        width: 100%;
        max-width: 400px;
        height: 70px;
        display: flex;
        flex-direction: row;  /* Changed to row to put title on left */
        align-items: center;  /* Center vertically */
        gap: 20px;           /* Space between title and values */
    }
    .metric-values {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 20px;
        flex-grow: 1;        /* Take remaining space */
    }
    .metric-group {
        text-align: center;
        min-width: 60px;
    }
    .primary-metric {
        font-size: 22px;
        font-weight: bold;
    }
    .secondary-metric {
        font-size: 16px;
        color: #888;
    }
    .metric-label {
        font-size: 11px;
        color: #666;
        margin-top: 2px;
    }
    .metric-title {
        font-size: 13px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        width: 80px;         /* Fixed width for title */
        text-align: left;    /* Align text to left */
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