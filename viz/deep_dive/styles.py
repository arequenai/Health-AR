DEEP_DIVE_CSS = """
<style>
.deep-dive-container {
    background-color: #1E1E1E;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
}

.deep-dive-title {
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 20px;
}

.metric-box {
    background-color: #2D2D2D;
    border-radius: 8px;
    padding: 10px;
    margin: 5px 0;
    text-align: center;
}

.metric-value {
    font-size: 24px;
    font-weight: bold;
}

.metric-label {
    font-size: 14px;
    color: #888;
}

/* Customizing the date selector */
.stSelectbox > div > div {
    background-color: #2D2D2D;
    border-radius: 5px;
}

/* Chart area */
.chart-container {
    background-color: #2D2D2D;
    border-radius: 10px;
    padding: 10px;
    margin: 15px 0;
}

/* Additional styles for day view */
.day-view-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.time-range {
    display: flex;
    gap: 10px;
    align-items: center;
}

/* Styling for meal tables */
.meal-container {
    background-color: #2D2D2D;
    border-radius: 5px;
    padding: 10px;
    margin-top: 10px;
}

.meal-title {
    font-size: 16px;
    font-weight: bold;
    margin-bottom: 5px;
}

.meal-summary {
    font-size: 14px;
    color: #AAA;
    margin-bottom: 5px;
}

.meal-item {
    font-size: 13px;
    padding: 2px 0;
}

/* Date picker customization */
/* Highlight today's date with circle */
.stDateInput .react-datepicker__day--today {
    border-radius: 50% !important;
    background-color: #4CAF50 !important;
    color: white !important;
    font-weight: bold !important;
}

/* Gray out future dates */
.stDateInput .react-datepicker__day--future {
    color: #888 !important;
    background-color: #e0e0e0 !important;
    cursor: not-allowed !important;
}

/* Custom class for future dates - added via JavaScript */
.react-datepicker__day--future-date {
    color: #888 !important;
    background-color: #e0e0e0 !important;
    cursor: not-allowed !important;
}
</style>

<script>
// Add JavaScript to gray out future dates in calendar
document.addEventListener('DOMContentLoaded', function() {
    // Function to add gray styling to future dates
    function grayOutFutureDates() {
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        
        // Find all date cells in the calendar
        const dateCells = document.querySelectorAll('.react-datepicker__day');
        
        dateCells.forEach(cell => {
            // Parse the date from the cell
            const dateText = cell.textContent;
            const currentMonth = document.querySelector('.react-datepicker__current-month').textContent;
            const year = new Date().getFullYear();
            const dateStr = `${dateText} ${currentMonth} ${year}`;
            const cellDate = new Date(dateStr);
            
            // If the date is in the future, add the class
            if (cellDate > today) {
                cell.classList.add('react-datepicker__day--future-date');
            }
        });
    }
    
    // Apply when the datepicker is opened
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList' && document.querySelector('.react-datepicker')) {
                grayOutFutureDates();
            }
        });
    });
    
    // Start observing the document for calendar appearance
    observer.observe(document.body, { childList: true, subtree: true });
});
</script>
""" 