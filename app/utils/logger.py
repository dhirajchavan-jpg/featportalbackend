# app/utils/logger.py
import logging
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo  # Built-in in Python 3.9+

def setup_logger():
    """Setup application logger with Indian Standard Time (IST)."""
    
    # 1. Define the Timezone (Asia/Kolkata)
    ist_timezone = ZoneInfo("Asia/Kolkata")

    # 2. Override the global logging converter to use IST
    # This ensures %(asctime)s in the log format uses Indian time
    def ist_converter(*args):
        return datetime.now(ist_timezone).timetuple()
    
    logging.Formatter.converter = ist_converter

    # 3. Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 4. Create log filename with IST timestamp
    # We use datetime.now(ist_timezone) to make sure the filename matches the content
    timestamp = datetime.now(ist_timezone).strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"app_{timestamp}.log"
    
    # 5. Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized in IST. Log file: {log_file}")
    
    return logger