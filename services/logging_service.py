import logging
import os
from flask import Response

LOG_FILE = 'app.log'
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Configure logging
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG, format=LOG_FORMAT)
logger = logging.getLogger()

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

def get_logs():
    """Read the log file and return its content as a styled HTML response."""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            log_content = f.read()

        # Return logs as HTML with black background and white text, and a button to clear logs
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
        </head>
        <body>
            <pre>{log_content}</pre>
            <form action="/clear_logs" method="POST">
                <button type="submit">Clear Logs</button>
            </form>
        </body>
        </html>
        """

        return Response(html_content, mimetype='text/html')
    else:
        return "Log file does not exist.", 404
