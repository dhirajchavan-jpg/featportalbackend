import re
from datetime import datetime, date
from pathlib import Path
from typing import List

# Define your logs directory
LOG_DIR = Path("logs")

# Regex to capture just the DATE part (YYYY-MM-DD) from the start of the line
DATE_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})")

def get_logs_by_date_range(start_date: date, end_date: date) -> str:
    matching_logs = []
    log_files = sorted(LOG_DIR.rglob("*.log"))
    
    for log_file in log_files:
        if not log_file.exists(): continue
            
        try:
            with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                # Track if we are currently inside a valid date block
                keep_current_block = False
                
                for line in f:
                    match = DATE_PATTERN.match(line)
                    
                    if match:
                        # It's a new log entry. Check the date.
                        log_date_str = match.group(1)
                        try:
                            log_date = datetime.strptime(log_date_str, "%Y-%m-%d").date()
                            if start_date <= log_date <= end_date:
                                keep_current_block = True
                                matching_logs.append(line)
                            else:
                                keep_current_block = False
                        except ValueError:
                            keep_current_block = False
                    
                    # If it's NOT a new entry (no date), but we are keeping the block
                    # (this captures stack traces and multi-line messages)
                    elif keep_current_block:
                        matching_logs.append(line)
                        
        except Exception as e:
            print(f"Error reading {log_file}: {e}")

    return "".join(matching_logs)