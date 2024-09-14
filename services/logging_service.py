import logging
import os
from flask import Response, Flask

app = Flask(__name__)

LOG_FILE = 'app.log'
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Configure logging
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG, format=LOG_FORMAT)
logger = logging.getLogger()

# Filter to exclude logging requests to /logs/data
class ExcludeLogsFilter(logging.Filter):
    def filter(self, record):
        return not ("/logs/data" in record.getMessage())

# Apply the filter to the Flask default logger (werkzeug)
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.addFilter(ExcludeLogsFilter())

def clean_logs():
    """Clear the log file."""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w'):
            pass  # This will clear the log file without deleting it
        logger.info('Log file cleaned up at startup')
        logger.info('###########################################')

def log_message(level, message):
    """Log a message with the specified level."""
    if level == 'info':
        logger.info(message)
    elif level == 'debug':
        logger.debug(message)
    elif level == 'warning':
        logger.warning(message)
    elif level == 'error':
        logger.error(message)
    else:
        logger.debug(message)

def log_exception(exception):
    """Log an exception with a full traceback."""
    logger.error("Exception occurred", exc_info=True)

@app.route('/logs')
def get_logs():
    """Return logs in an HTML response with real-time refreshing."""
    html_content = f"""
    <html>
    <head>
        <style>
            body {{
                background-color: black;
                color: white;
                font-family: monospace;
            }}
            pre {{
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
            button {{
                background-color: red;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                font-size: 16px;
                margin-top: 10px;
                cursor: pointer;
            }}
            button:hover {{
                background-color: darkred;
            }}
        </style>
        <script>
            function fetchLogs() {{
                fetch('/logs/data')
                .then(response => response.text())
                .then(data => {{
                    document.getElementById('log-content').textContent = data;
                }});
            }}
            
            setInterval(fetchLogs, 5000);  // Fetch logs every 5 seconds
        </script>
    </head>
    <body onload="fetchLogs()">
        <h1>Log Output</h1>
        <pre id="log-content">Loading logs...</pre>
        <form action="/clear_logs" method="POST">
            <button type="submit">Clear Logs</button>
        </form>
    </body>
    </html>
    """
    return Response(html_content, mimetype='text/html')

@app.route('/logs/data')
def get_logs_data() -> str:
    """Return raw log data as plain text for real-time updates."""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            log_content = f.read()
        return log_content
    else:
        return "Log file does not exist."
