import subprocess
import os
import sys
import time
import signal
import socket
import logging
from datetime import datetime
from pathlib import Path
try:
    from zoneinfo import ZoneInfo # Python 3.9+
except ImportError:
    pass # Fallback if module not found
# --- LOGGER SETUP (Ref: app/utils/logger.py) ---
def setup_logger():
    """Setup logger with IST timezone and file storage."""
    # 1. Define Timezone (Asia/Kolkata)
    try:
        ist_timezone = ZoneInfo("Asia/Kolkata")
    except NameError:
        ist_timezone = None
    # 2. Override logging time converter to IST
    if ist_timezone:
        def ist_converter(*args):
            return datetime.now(ist_timezone).timetuple()
        logging.Formatter.converter = ist_converter
    # 3. Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    # 4. Create unique log filename
    # Example: logs/ollama_cluster_20240115_120000.log
    timestamp = datetime.now(ist_timezone).strftime("%Y%m%d_%H%M%S") if ist_timezone else datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"ollama_cluster_{timestamp}.log"
    # 5. Configure Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'), # Save to file
            logging.StreamHandler(sys.stdout)                # Print to console
        ]
    )
    return logging.getLogger("OllamaCluster")
# Initialize Logger
logger = setup_logger()
# --- CONFIGURATION ---
INSTANCES = [
    {"port": 11434, "gpu": 0},
    {"port": 11435, "gpu": 1},
    {"port": 11436, "gpu": 2}
]
processes = []
def is_port_in_use(port):
    """Checks if a port is already being used."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0
def start_ollama(port, gpu_id):
    """Starts an Ollama instance on a specific port and GPU."""
    # Check port availability
    if is_port_in_use(port):
        logger.warning(f"Port {port} is ALREADY IN USE. Skipping start.")
        return None
    logger.info(f"Starting Instance on Port {port} using GPU {gpu_id}...")
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"0.0.0.0:{port}"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Start process
    proc = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return proc
def cleanup(signum, frame):
    """Clean shutdown handler."""
    logger.info(" Stopping active instances...")
    for p in processes:
        if p is None: continue
        try:
            if sys.platform == "win32":
                subprocess.call(['taskkill', '/F', '/T', '/PID', str(p.pid)])
            else:
                p.terminate()
        except Exception as e:
            logger.error(f"Error killing process {p.pid}: {e}")
    logger.info("Bye!")
    sys.exit(0)
if __name__ == "__main__":
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    logger.info(f"Checking {len(INSTANCES)} ports...")
    try:
        for config in INSTANCES:
            p = start_ollama(config["port"], config["gpu"])
            if p is not None:
                processes.append(p)
                time.sleep(2) # Small delay to be safe
        logger.info(" SYSTEM READY.")
        logger.info("   If a port was skipped, it means Ollama was already running there (safe).")
        logger.info("   Press Ctrl+C to stop the instances initiated by this script.")
        while True:
            time.sleep(1)
    except Exception as e:
        logger.error(f"Critical Error: {e}", exc_info=True)
        cleanup(None, None)