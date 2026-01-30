# ---------------------------
# Screen Screenshot Tracker - Scalable Architecture
# Zero local storage, in-memory processing with retry queue
# Now with HTTPS Control Server for Dashboard Integration
# ---------------------------
import os
import io
import csv
import time
import socket
import signal
import logging
import requests
import re
import sys
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading
import pickle
from collections import deque
from dataclasses import dataclass
from typing import Optional
from queue import Queue
import json

from datetime import datetime
import pytz
import mss
from PIL import Image
import platform
import uuid
import getpass
import subprocess
import psutil
import ssl
import tempfile
import ipaddress



# CONFIGURATION
SERVER_BASE = "http://101.53.247.91:3344"  # Update this to match your server IP
ZONE = pytz.timezone("Asia/Karachi")
SCREENSHOTS_PER_SECOND = 1  # 1 screenshot per second
PC_CONFIG_JSON = "pc_config.json"

# BACKEND API CONFIGURATION (for database sync)
#BACKEND_API_BASE = "http://101.53.247.91:8090"  # Update to your main backend server IP/port

#BACKEND_API_BASE = "http://192.168.1.200:8070"

# Temp directory for disk backup (safety net)
TEMP_DIR = os.path.join(os.path.expanduser("~"), "temp", "tracker_uploads")

# Configure logging - FILE ONLY (no console output)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Remove existing handlers
logger.handlers.clear()

# File handler only (no console logging)
file_handler = logging.FileHandler('app.log', mode='a')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
file_handler.setFormatter(file_formatter)

logger.addHandler(file_handler)

hostname = socket.gethostname()

# Thread synchronization
running = threading.Event()
# Don't set running yet - we wait for /start command

# Shutdown flag to track if graceful shutdown is in progress
shutdown_in_progress = threading.Event()

# Tracking state - controlled by dashboard
tracking_active = threading.Event()
tracking_paused = threading.Event()  # New: for pause/resume functionality
current_writer_name = None
current_title = None
session_folder = None
session_start_time = None

# Screenshot counter
screenshot_count = 0
capture_start_time = None
last_rate_display = 0

# Upload queue for asynchronous uploads (separate from retry queue)
upload_queue = Queue(maxsize=500)  # Buffer up to 500 PNG screenshots

# Disk write queue for asynchronous disk backup (non-blocking)
disk_write_queue = Queue(maxsize=500)  # Buffer up to 500 file writes

# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

# Reusable mss context (faster than creating new one each time)
_mss_context = None

def get_mss_context():
    """Get or create reusable mss context."""
    global _mss_context
    if _mss_context is None:
        _mss_context = mss.mss()
    return _mss_context

# Reusable HTTP session with connection pooling (faster uploads)
_http_session = None

def get_http_session():
    """Get or create reusable HTTP session with connection pooling."""
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,  # Number of connection pools
            pool_maxsize=20,      # Max connections per pool
        )
        _http_session.mount("http://", adapter)
        _http_session.mount("https://", adapter)
    return _http_session


def sanitize_folder_name(name: str) -> str:
    """Sanitize a string to be safe for use as a folder name."""
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Replace multiple spaces/underscores with single underscore
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    # Remove leading/trailing whitespace and underscores
    sanitized = sanitized.strip('_ ')
    # Limit length to avoid path issues
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    return sanitized or "unnamed"


# ─── RETRY QUEUE SYSTEM ──────────────────────────────────────────────────

@dataclass
class FailedUpload:
    """Represents a failed screenshot upload."""
    image_data: bytes  # Image bytes (PNG or WEBP format)
    filename: str
    folder: str
    retry_count: int = 0
    first_failed_time: float = None
    last_retry_time: float = None
    
    def __post_init__(self):
        if self.first_failed_time is None:
            self.first_failed_time = time.time()
        self.last_retry_time = time.time()


class RetryQueue:
    """Manages failed uploads with retry logic and disk fallback."""
    
    def __init__(self, max_memory_items=100, fallback_dir=None):
        self.memory_queue = deque()  # In-memory queue
        self.max_memory_items = max_memory_items
        self.fallback_dir = fallback_dir or os.path.join(
            os.path.expanduser("~"), ".tracker_retry_queue"
        )
        self.queue_file = os.path.join(self.fallback_dir, "retry_queue.pkl")
        self.lock = threading.Lock()
        self.max_retries = 10  # Maximum retry attempts
        self.retry_delays = [1, 2, 5, 10, 30, 60, 300, 600, 1800, 3600]  # Exponential backoff
        
        # Load existing queue from disk on startup
        self._load_queue()
    
    def add_failed_upload(self, image_data: bytes, filename: str, folder: str):
        """Add a failed upload to the retry queue."""
        with self.lock:
            failed = FailedUpload(
                image_data=image_data,
                filename=filename,
                folder=folder
            )
            
            if len(self.memory_queue) < self.max_memory_items:
                # Add to memory queue
                self.memory_queue.append(failed)
                logger.info(f"Added {filename} to memory retry queue ({len(self.memory_queue)}/{self.max_memory_items})")
            else:
                # Fallback to disk
                self._save_to_disk(failed)
                logger.warning(f"Memory queue full, saved {filename} to disk fallback")
            
            # Persist queue state
            self._save_queue_state()
    
    def get_next_retry(self) -> Optional[FailedUpload]:
        """Get next item ready for retry."""
        with self.lock:
            if not self.memory_queue:
                # Try loading from disk
                self._load_from_disk()
            
            if not self.memory_queue:
                return None
            
            # Check if enough time has passed since last retry
            failed = self.memory_queue[0]
            delay = self._get_retry_delay(failed.retry_count)
            
            if time.time() - failed.last_retry_time >= delay:
                return self.memory_queue.popleft()
            
            return None
    
    def mark_success(self, failed: FailedUpload):
        """Mark upload as successful and remove from queue."""
        with self.lock:
            logger.info(f"Successfully uploaded {failed.filename} after {failed.retry_count} retries")
            self._save_queue_state()
    
    def mark_failed_again(self, failed: FailedUpload):
        """Mark retry as failed, increment count, and re-queue."""
        with self.lock:
            failed.retry_count += 1
            failed.last_retry_time = time.time()
            
            if failed.retry_count >= self.max_retries:
                # Max retries reached - save to permanent failure log
                logger.error(f"Max retries reached for {failed.filename}, saving to failure log")
                self._save_to_failure_log(failed)
            else:
                # Re-queue for another retry
                if len(self.memory_queue) < self.max_memory_items:
                    self.memory_queue.append(failed)
                else:
                    self._save_to_disk(failed)
            
            self._save_queue_state()
    
    def _get_retry_delay(self, retry_count: int) -> float:
        """Get delay in seconds for retry attempt."""
        if retry_count < len(self.retry_delays):
            return self.retry_delays[retry_count]
        return self.retry_delays[-1]  # Use max delay
    
    def _save_to_disk(self, failed: FailedUpload):
        """Save failed upload to disk as fallback."""
        os.makedirs(self.fallback_dir, exist_ok=True)
        filepath = os.path.join(
            self.fallback_dir,
            f"{int(time.time())}_{failed.filename}"
        )
        with open(filepath, 'wb') as f:
            pickle.dump(failed, f)
        logger.debug(f"Saved failed upload to disk: {filepath}")
    
    def _load_from_disk(self):
        """Load failed uploads from disk back to memory queue."""
        if not os.path.exists(self.fallback_dir):
            return
        
        files = sorted([
            f for f in os.listdir(self.fallback_dir)
            if f.endswith('.pkl') and not f.startswith('retry_queue')
        ])
        
        loaded = 0
        for filename in files:
            if len(self.memory_queue) >= self.max_memory_items:
                break
            
            filepath = os.path.join(self.fallback_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    failed = pickle.load(f)
                    self.memory_queue.append(failed)
                    os.remove(filepath)  # Remove after loading
                    loaded += 1
            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")
        
        if loaded > 0:
            logger.info(f"Loaded {loaded} failed uploads from disk to memory queue")
    
    def _save_queue_state(self):
        """Save queue metadata to disk."""
        os.makedirs(self.fallback_dir, exist_ok=True)
        state = {
            'memory_queue_size': len(self.memory_queue),
            'timestamp': time.time()
        }
        with open(self.queue_file, 'wb') as f:
            pickle.dump(state, f)
    
    def _load_queue(self):
        """Load queue state on startup."""
        if os.path.exists(self.fallback_dir):
            self._load_from_disk()
            logger.info(f"Loaded retry queue: {len(self.memory_queue)} items in memory")
    
    def _save_to_failure_log(self, failed: FailedUpload):
        """Save permanently failed uploads to a log directory."""
        failure_log_dir = os.path.join(self.fallback_dir, "permanent_failures")
        os.makedirs(failure_log_dir, exist_ok=True)
        
        # Save image file
        image_path = os.path.join(failure_log_dir, failed.filename)
        with open(image_path, 'wb') as f:
            f.write(failed.image_data)
        
        # Save metadata
        metadata_path = os.path.join(failure_log_dir, f"{failed.filename}.meta")
        with open(metadata_path, 'w') as f:
            f.write(f"Filename: {failed.filename}\n")
            f.write(f"Folder: {failed.folder}\n")
            f.write(f"Retry Count: {failed.retry_count}\n")
            f.write(f"First Failed: {time.ctime(failed.first_failed_time)}\n")
        
        logger.error(f"Permanently failed upload saved to: {image_path}")
    
    def get_queue_stats(self) -> dict:
        """Get statistics about the retry queue."""
        with self.lock:
            disk_count = 0
            if os.path.exists(self.fallback_dir):
                disk_count = len([
                    f for f in os.listdir(self.fallback_dir)
                    if f.endswith('.pkl') and not f.startswith('retry_queue')
                ])
            
            return {
                'memory_queue_size': len(self.memory_queue),
                'disk_queue_size': disk_count,
                'total_pending': len(self.memory_queue) + disk_count
            }


# Global retry queue instance
retry_queue = RetryQueue(max_memory_items=100)

# ─── HTTP HELPERS ─────────────────────────────────────────────────────

def check_server_connectivity() -> tuple[bool, str]:
    """Check if server is reachable. Returns (is_connected, error_message)."""
    try:
        r = requests.get(f"{SERVER_BASE}/exists_folder", params={"folder": "test"}, timeout=5)
        r.raise_for_status()
        return True, ""
    except requests.exceptions.ConnectionError as e:
        return False, f"Cannot connect to server at {SERVER_BASE} - {e}"
    except requests.exceptions.Timeout:
        return False, f"Server at {SERVER_BASE} did not respond in time"
    except Exception as e:
        return False, f"Error connecting to server: {e}"

def remote_exists(folder_rel: str) -> bool:
    """Check if remote folder exists. Returns False if server is not available."""
    try:
        r = requests.get(f"{SERVER_BASE}/exists_folder", params={"folder": folder_rel}, timeout=5)
        r.raise_for_status()
        return r.json().get("exists", False)
    except requests.RequestException:
        return False


def get_text(path: str) -> Optional[str]:
    """Get text file from server."""
    r = requests.get(f"{SERVER_BASE}/files/{path}", timeout=10)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.text


def upload_file_data(buf_or_path, filename: str, folder: str):
    """Upload file data from BytesIO buffer or file path to server (optimized with connection pooling)."""
    upload_url = f"{SERVER_BASE}/upload"
    logger.info(f"[UPLOAD] Attempting to upload {filename} to {folder}")
    logger.info(f"[UPLOAD] Server URL: {upload_url}")
    
    session = get_http_session()  # Use reusable session with connection pooling
    
    if isinstance(buf_or_path, str):
        # File path (for config files)
        f = open(buf_or_path, 'rb')
        file_size = os.path.getsize(buf_or_path)
        logger.info(f"[UPLOAD] Reading from file: {buf_or_path} ({file_size} bytes)")
    else:
        # BytesIO buffer (for screenshots)
        buf_or_path.seek(0)
        f = buf_or_path
        file_size = len(buf_or_path.getvalue()) if hasattr(buf_or_path, 'getvalue') else "unknown"
        logger.info(f"[UPLOAD] Reading from buffer ({file_size} bytes)")
    
    files = {'file': (filename, f)}
    data = {'folder': folder}
    
    try:
        logger.info(f"[UPLOAD] Sending POST request to {upload_url}...")
        # Use shorter timeout for small PNG files
        # Connection pooling makes subsequent uploads faster
        response = session.post(
            upload_url,
            files=files,
            data=data,
            timeout=10  # 10 seconds is enough for small WEBP files
        )
        logger.info(f"[UPLOAD] Response received: Status {response.status_code}")
        response.raise_for_status()
        result = response.json()
        saved_path = result.get('path', 'unknown')
        logger.info(f"[UPLOAD SUCCESS] {filename} -> {saved_path}")
        return True
    except requests.exceptions.ConnectionError as e:
        error_msg = f"[UPLOAD FAILED] {filename}: Cannot connect to server at {upload_url}"
        logger.error(error_msg)
        logger.error(f"[UPLOAD FAILED] Error details: {e}")
        raise
    except requests.exceptions.Timeout as e:
        error_msg = f"[UPLOAD FAILED] {filename}: Timeout after 10 seconds"
        logger.error(error_msg)
        logger.error(f"[UPLOAD FAILED] Error details: {e}")
        raise
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if hasattr(e, 'response') and e.response else "unknown"
        error_msg = f"[UPLOAD FAILED] {filename}: HTTP {status_code}"
        logger.error(error_msg)
        logger.error(f"[UPLOAD FAILED] Error details: {e}")
        try:
            error_body = e.response.text[:200] if hasattr(e, 'response') and e.response else "N/A"
            logger.error(f"[UPLOAD FAILED] Response body: {error_body}")
        except:
            pass
        raise
    except Exception as e:
        error_msg = f"[UPLOAD FAILED] {filename}: Unexpected error - {type(e).__name__}: {e}"
        logger.error(error_msg)
        import traceback
        logger.error(f"[UPLOAD FAILED] Traceback: {traceback.format_exc()}")
        raise
    finally:
        if isinstance(buf_or_path, str):
            f.close()


# ─── STATE MANAGEMENT ─────────────────────────────────────────────────

def load_state() -> dict[str, str]:
    """Load state from server."""
    try:
        txt = get_text(PC_CONFIG_JSON)
        if txt:
            return json.loads(txt)
    except requests.RequestException:
        logger.debug("Server not available, using empty state")
    return {}


def save_state(state: dict[str, str]):
    """Save state to server."""
    try:
        data = json.dumps(state, indent=2).encode("utf-8")
        upload_file_data(io.BytesIO(data), PC_CONFIG_JSON, "")
        logger.info("State saved to server")
    except requests.RequestException as e:
        logger.error(f"Failed to save state to server: {e}")


# ─── BACKEND DATABASE SYNC ──────────────────────────────────────────

def get_writer_info_from_filename():
    """Extract writer name and ID from the script's own filename."""
    try:
        filename = os.path.basename(sys.argv[0])
        logger.info(f"Parsing filename for writer info: {filename}")
        
        # Find all number sequences and their positions
        matches = list(re.finditer(r'(\d+)', filename))
        if matches:
            last_match = matches[-1] # Get the last number sequence found
            writer_id = int(last_match.group(1))
            
            # The name is everything BEFORE that last number
            name_part = filename[:last_match.start()].strip('_')
            
            # Clean up name: 
            # 1. remove "tracker" (case insensitive)
            # 2. replace underscores with spaces (as requested for Khizra_Razzaki -> Khizra Razzaki)
            name_part = name_part.replace('Tracker', '').replace('tracker', '')
            name_part = name_part.replace('_', ' ').strip()
            
            if not name_part:
                name_part = "UnknownWriter"
                
            return name_part, writer_id
    except Exception as e:
        logger.error(f"Failed to parse writer info from filename: {e}")
    return None, None


def sync_db_tracker_status(status_bool: bool):
    """
    Update tracker_status in the backend database.
    This call is now public and does not require a JWT token.
    """
    name, writer_id = get_writer_info_from_filename()
    if not writer_id:
        logger.warning("Could not determine writer_id from filename. DB sync skipped.")
        return False
        
    logger.info(f"Syncing tracker_status for Writer ID {writer_id} ({name}) to {status_bool}")
    
    try:
        # Update status via public endpoint
        update_url = f"{BACKEND_API_BASE}/update-writer-tracker-status"
        payload = {"writer_id": writer_id, "tracker_status": status_bool}
        
        # No Authorization header needed anymore
        update_response = requests.post(update_url, json=payload, timeout=10)
        
        if update_response.status_code == 200:
            if status_bool:
                msg = "Tracker started"
                print(msg)
                logger.info(msg)
            else:
                logger.info(f"Successfully updated DB tracker_status to {status_bool}")
        else:
            error_msg = f"Error: API returned {update_response.status_code}"
            print(error_msg)
            logger.error(error_msg)
            update_response.raise_for_status()
            
        return True
    except Exception as e:
        error_msg = f"Error updating tracker status: {e}"
        print(error_msg)
        logger.error(error_msg)
        return False



def get_win_cmd(cmd: str) -> str:
    """Helper to run wmic/powershell commands and return the result."""
    try:
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
        lines = [line.strip() for line in output.split('\n') if line.strip()]
        if len(lines) > 1:
            return lines[1] # Skip header
        return "unknown"
    except:
        return "unknown"

def collect_system_metadata(writer_name: str, title: str) -> dict:
    """Collect system metadata including detailed hardware specs."""
    try:
        mac_num = uuid.getnode()
        mac = ':'.join(['{:02x}'.format((mac_num >> elements) & 0xff) for elements in range(0,2*6,2)][::-1])
    except:
        mac = "unknown"

    try:
        user = getpass.getuser()
    except:
        user = "unknown"

    try:
        ip_addr = socket.gethostbyname(socket.gethostname())
    except:
        ip_addr = "unknown"

    # Enhanced Hardware Specs
    cpu_name = get_win_cmd("wmic cpu get name")
    gpu_name = get_win_cmd("wmic path win32_VideoController get name")
    
    ram_raw = psutil.virtual_memory().total if hasattr(psutil, 'virtual_memory') else 0
    ram_gb = f"{round(ram_raw / (1024**3), 2)} GB" if ram_raw > 0 else "unknown"
    
    storage_gb = "unknown"
    try:
        total_b = 0
        for disk in psutil.disk_partitions():
            if 'cdrom' in disk.opts or disk.fstype == '':
                continue
            try:
                total_b += psutil.disk_usage(disk.mountpoint).total
            except:
                continue
        if total_b > 0:
            storage_gb = f"{round(total_b / (1024**3), 2)} GB"
    except:
        pass

    return {
        "writer_name": writer_name,
        "title": title,
        "hostname": socket.gethostname(),
        "user": user,
        "os_system": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": cpu_name if cpu_name != "unknown" else platform.processor(),
        "graphics_card": gpu_name,
        "installed_ram": ram_gb,
        "total_storage": storage_gb,
        "python_version": platform.python_version(),
        "mac_address": mac,
        "ip_address": ip_addr,
        "timestamp": datetime.now(ZONE).isoformat()
    }


# ─── RUN LOG ──────────────────────────────────────────────────────────

LOG_CSV = "pc_log.csv"

def append_run_log(writer_name: str, title: str, start_iso: str, end_iso: str):
    """Append run log to server if available, otherwise skip."""
    try:
        txt = get_text(LOG_CSV) or ''
        rows = list(csv.reader(txt.splitlines())) if txt else []
        if not rows:
            rows = [['Writer', 'Title', 'Start Time', 'End Time']]
        rows.append([writer_name, title, start_iso, end_iso])
        buf = io.StringIO()
        csv.writer(buf).writerows(rows)
        upload_file_data(io.BytesIO(buf.getvalue().encode()), LOG_CSV, "")
    except requests.RequestException:
        logger.debug("Server not available, skipping run log upload")


# ─── SCREENSHOT CAPTURE ────────────────────────────────────────────────

def take_screenshot():
    """Capture screenshot directly as PNG format (OPTIMIZED for fastest capture + upload)."""
    global screenshot_count, session_folder
    
    if not session_folder:
        logger.warning("No session folder set, skipping screenshot")
        return False
    
    try:
        # 1. Capture with mss (reuse context for speed)
        # monitors[0] = virtual/all monitors combined (extended desktop)
        # monitors[1] = first physical monitor
        # monitors[2+] = additional physical monitors
        sct = get_mss_context()
        monitor = sct.monitors[0]  # Capture all monitors combined (extended desktop)
        screenshot = sct.grab(monitor)
        
        # 2. Convert mss screenshot to PIL Image
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        
        # 3. Resize if screen is very large (reduces encoding time AND file size)
        # This helps both capture speed and upload speed
        max_dimension = 2560  # Full HD - adjust if you need higher resolution
        if img.width > max_dimension or img.height > max_dimension:
            img.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
        
        # 4. Save directly as PNG
        # PNG is lossless; compress_level 3 provides a good balance between speed and size
        png_buf = io.BytesIO()
        img.save(png_buf, 'PNG', compress_level=3)
        png_data = png_buf.getvalue()
        png_buf.close()
        
        # 5. Generate filename
        timestamp = datetime.now(ZONE).strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"screenshot_{timestamp}.PNG"
        temp_filepath = os.path.join(TEMP_DIR, filename)
        
        # 6. Queue for asynchronous disk write (non-blocking) - save PNG for backup
        try:
            disk_write_queue.put_nowait((png_data, temp_filepath))
        except:
            # Disk queue full - log but continue
            logger.warning(f"Disk write queue full for {filename}")
        
        # 7. Queue for upload
        try:
            upload_queue.put_nowait((png_data, filename, session_folder, temp_filepath))
            screenshot_count += 1
            return True
        except Exception as queue_error:
            # Upload queue full - add to retry queue as fallback
            logger.warning(f"Upload queue full for {filename}, adding to retry queue: {queue_error}")
            retry_queue.add_failed_upload(png_data, filename, session_folder)
            return False
            
    except Exception as e:
        error_msg = str(e)
        if "DISPLAY" in error_msg or "display" in error_msg.lower():
            logger.error(f"Failed to take screenshot - DISPLAY not accessible: {e}")
            logger.error("Screenshot capture requires access to X11 display. Ensure $DISPLAY is set.")
        else:
            logger.error(f"Failed to take screenshot: {e}")
        return False


# ─── DISK WRITE WORKER ─────────────────────────────────────────────────

def disk_write_worker():
    """Background thread that writes files to disk asynchronously."""
    logger.info("Disk write worker thread started")
    
    while running.is_set() or disk_write_queue.qsize() > 0:
        try:
            # Get next item from disk write queue
            try:
                image_data, temp_filepath = disk_write_queue.get(timeout=1)
            except:
                continue  # Timeout, check if still running
            
            # Write to disk
            try:
                with open(temp_filepath, 'wb') as f:
                    f.write(image_data)
                disk_write_queue.task_done()
            except Exception as write_error:
                logger.error(f"Failed to write {os.path.basename(temp_filepath)} to disk: {write_error}")
                disk_write_queue.task_done()
                
        except Exception as e:
            logger.error(f"Error in disk write worker: {e}")
            time.sleep(0.1)
    
    logger.info("Disk write worker thread stopped")


# ─── UPLOAD WORKER ─────────────────────────────────────────────────────

def upload_worker():
    """Background thread that processes upload queue asynchronously and deletes temp files."""
    worker_id = threading.current_thread().name
    logger.info(f"[UPLOAD WORKER {worker_id}] Started")
    
    # Continue processing even after capture stops (until queue and temp are empty)
    while True:
        try:
            # Get next item from upload queue (blocking with timeout)
            try:
                image_data, filename, folder, temp_filepath = upload_queue.get(timeout=1)
                logger.info(f"[UPLOAD WORKER {worker_id}] Processing: {filename} (queue size: {upload_queue.qsize()})")
            except:
                # Timeout - check if we should continue
                # Continue if: queue not empty, temp directory has files, or running flag is set
                if not running.is_set():
                    # Capture stopped - check if there's more work
                    if upload_queue.qsize() == 0:
                        # Check temp directory
                        temp_files = []
                        if os.path.exists(TEMP_DIR):
                            temp_files = [
                                f for f in os.listdir(TEMP_DIR)
                                if f.startswith("screenshot_") and (f.endswith(".png") or f.endswith(".webp"))
                            ]
                        if len(temp_files) == 0:
                            # No more work, but wait a bit in case retry worker adds more
                            time.sleep(2)
                            continue
                continue
            
            # Attempt upload
            try:
                logger.info(f"[UPLOAD WORKER {worker_id}] Uploading {filename} to {folder}...")
                upload_file_data(io.BytesIO(image_data), filename, folder)
                
                # Delete temp file after successful upload
                if os.path.exists(temp_filepath):
                    try:
                        os.remove(temp_filepath)
                        logger.info(f"[UPLOAD WORKER {worker_id}] Deleted temp file: {filename}")
                    except Exception as del_error:
                        logger.warning(f"[UPLOAD WORKER {worker_id}] Failed to delete temp file {filename}: {del_error}")
                else:
                    logger.debug(f"[UPLOAD WORKER {worker_id}] Temp file already removed: {filename}")
                
                upload_queue.task_done()
            except Exception as upload_error:
                # Upload failed - keep temp file, add to retry queue
                logger.error(f"[UPLOAD WORKER {worker_id}] Upload failed for {filename}: {upload_error}")
                logger.info(f"[UPLOAD WORKER {worker_id}] Keeping temp file for retry: {temp_filepath}")
                retry_queue.add_failed_upload(image_data, filename, folder)
                upload_queue.task_done()
                
        except Exception as e:
            logger.error(f"[UPLOAD WORKER {worker_id}] Error in upload worker: {e}")
            import traceback
            logger.error(f"[UPLOAD WORKER {worker_id}] Traceback: {traceback.format_exc()}")
            time.sleep(0.1)
    
    logger.info(f"[UPLOAD WORKER {worker_id}] Stopped - all uploads complete")


# ─── RETRY WORKER ─────────────────────────────────────────────────────

def retry_worker():
    """Background thread that continuously retries failed uploads from temp directory."""
    logger.info("[RETRY WORKER] Started")
    
    # Continue processing even after capture stops (until temp directory is empty)
    while True:
        try:
            # First, check temp directory for any leftover files
            if os.path.exists(TEMP_DIR):
                temp_files = [
                    f for f in os.listdir(TEMP_DIR)
                    if f.startswith("screenshot_") and (f.endswith(".png") or f.endswith(".webp"))
                ]
                
                if temp_files:
                    logger.info(f"[RETRY WORKER] Found {len(temp_files)} files in temp directory")
                
                for temp_filename in temp_files:
                    temp_filepath = os.path.join(TEMP_DIR, temp_filename)
                    try:
                        # Read file from disk
                        if not os.path.exists(temp_filepath):
                            logger.warning(f"[RETRY WORKER] Temp file not found (may have been deleted): {temp_filepath}")
                            continue
                        
                        logger.info(f"[RETRY WORKER] Reading temp file: {temp_filepath}")
                        with open(temp_filepath, 'rb') as f:
                            image_data = f.read()
                        
                        file_size = len(image_data)
                        logger.info(f"[RETRY WORKER] File size: {file_size} bytes")
                        
                        # Attempt upload using current session folder
                        try:
                            if session_folder:
                                logger.info(f"[RETRY WORKER] Uploading temp file: {temp_filename} to {session_folder}")
                                upload_file_data(io.BytesIO(image_data), temp_filename, session_folder)
                                # Success - delete temp file
                                os.remove(temp_filepath)
                                logger.info(f"[RETRY WORKER] Successfully uploaded and deleted: {temp_filename}")
                        except Exception as upload_error:
                            logger.error(f"[RETRY WORKER] Failed to upload temp file {temp_filename}: {upload_error}")
                            # Keep file for next retry cycle
                            
                    except Exception as file_error:
                        logger.error(f"[RETRY WORKER] Error processing temp file {temp_filename}: {file_error}")
                        import traceback
                        logger.error(f"[RETRY WORKER] Traceback: {traceback.format_exc()}")
            
            # Then process retry queue
            failed = retry_queue.get_next_retry()
            
            if failed is None:
                # No items ready for retry, check temp directory
                if not running.is_set():
                    # Capture stopped - check if temp directory is empty
                    temp_files = []
                    if os.path.exists(TEMP_DIR):
                        temp_files = [
                            f for f in os.listdir(TEMP_DIR)
                            if f.startswith("screenshot_") and (f.endswith(".png") or f.endswith(".webp"))
                        ]
                    if len(temp_files) == 0:
                        # No more work - check retry queue stats
                        stats = retry_queue.get_queue_stats()
                        if stats['total_pending'] == 0:
                            # Really done - exit worker
                            logger.info("[RETRY WORKER] All uploads complete, exiting")
                            break
                
                time.sleep(5)  # Check temp directory every 5 seconds
                continue
            
            # Attempt retry
            logger.info(f"[RETRY WORKER] Retrying upload: {failed.filename} (attempt {failed.retry_count + 1})")
            
            try:
                upload_file_data(
                    io.BytesIO(failed.image_data),
                    failed.filename,
                    failed.folder
                )
                # Success!
                retry_queue.mark_success(failed)
                logger.info(f"[RETRY WORKER] Successfully uploaded {failed.filename} after retry")
                
                # Also check if temp file exists and delete it
                temp_filepath = os.path.join(TEMP_DIR, failed.filename)
                if os.path.exists(temp_filepath):
                    try:
                        os.remove(temp_filepath)
                        logger.info(f"[RETRY WORKER] Deleted temp file: {failed.filename}")
                    except:
                        pass
                
            except Exception as retry_error:
                # Retry failed again
                logger.warning(f"[RETRY WORKER] Retry failed for {failed.filename}: {retry_error}")
                retry_queue.mark_failed_again(failed)
                
        except Exception as e:
            logger.error(f"[RETRY WORKER] Error in retry worker: {e}")
            import traceback
            logger.error(f"[RETRY WORKER] Traceback: {traceback.format_exc()}")
            time.sleep(5)  # Wait before retrying worker itself
    
    logger.info("[RETRY WORKER] Stopped - all uploads complete")





# ─── SCREENSHOT CAPTURE LOOP ───────────────────────────────────────────

def screenshot_capture_loop():
    """Main loop to capture and upload screenshots at configured rate."""
    global capture_start_time, last_rate_display, screenshot_count
    
    interval = 1.0 / SCREENSHOTS_PER_SECOND
    
    logger.info(f"Screenshot capture loop ready, waiting for tracking to be activated...")
    
    while running.is_set():
        # Wait for tracking to be activated
        if not tracking_active.is_set():
            time.sleep(0.5)
            continue
        
        # Check if tracking is paused
        if tracking_paused.is_set():
            time.sleep(0.5)  # Sleep while paused
            continue
        
        # Initialize capture timing when tracking starts
        if capture_start_time is None:
            capture_start_time = time.time()
            last_rate_display = capture_start_time
            logger.info(f"Starting screenshot capture at {SCREENSHOTS_PER_SECOND} screenshots per second")
        
        try:
            loop_start = time.time()
            
            # Capture and queue for upload
            take_screenshot()
            
            # Display rate every 5 seconds (logged only, no console output)
            current_time = time.time()
            if current_time - last_rate_display >= 5:
                elapsed_total = current_time - capture_start_time
                if elapsed_total > 0:
                    actual_rate = screenshot_count / elapsed_total
                    queue_size = upload_queue.qsize()
                    logger.info(f"Capture rate: {actual_rate:.1f}/sec, Total: {screenshot_count}, Upload queue: {queue_size}")
                last_rate_display = current_time
            
            # Calculate sleep time to maintain rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, interval - elapsed)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # If we're taking longer than expected per screenshot, log a warning
                if elapsed > interval * 1.5:  # More than 50% over target
                    logger.warning(f"Screenshot capture taking too long: {elapsed:.3f}s (target: {interval:.3f}s)")
                
        except Exception as e:
            logger.error(f"Error in screenshot_capture_loop: {e}")
            time.sleep(0.1)
    
    # Final statistics
    if capture_start_time:
        elapsed_total = time.time() - capture_start_time
        if elapsed_total > 0:
            final_rate = screenshot_count / elapsed_total
            logger.info(f"Screenshot capture stopped. Final rate: {final_rate:.1f} screenshots/sec, Total captured: {screenshot_count}")
    
    logger.info("Screenshot capture loop stopped")



# ─── SIGNAL HANDLING ──────────────────────────────────────────────────

def signal_handler(signum, frame):
    """Handle Ctrl+C signal. First press starts graceful shutdown, subsequent presses are ignored."""
    global running, shutdown_in_progress
    
    if shutdown_in_progress.is_set():
        # Shutdown already in progress, ignore subsequent Ctrl+C
        print("\nShutdown in progress. Please wait for all uploads to complete...")
        logger.info("Additional Ctrl+C pressed during shutdown (ignored)")
        return

    # First Ctrl+C - start graceful shutdown
    shutdown_in_progress.set()
    tracking_active.clear()
    running.clear()
    print("\nShutdown initiated. Waiting for all uploads to complete...")
    logger.info("Ctrl+C received - starting graceful shutdown")







# ─── STATUS POLLING WORKER ─────────────────────────────────────────────

def status_polling_worker(writer_id: int, name: str):
    """Poll server for writer work status and control tracking."""
    global current_title, session_folder, current_writer_name

    start_msg = f"[POLLER] Started polling work status for writer {writer_id} ({name})"
    logger.info(start_msg)
    print(start_msg)
    
    poll_url = f"{BACKEND_API_BASE}/get-writer-work-status"
    
    while running.is_set():
        try:
            # Send GET request
            params = {"writer_id": writer_id}
            response = requests.get(poll_url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                # Expected format: {'writer', 'work-status', 'keyword'}
                work_status = data.get('work-status')
                keyword = data.get('keyword')
                
                if work_status and work_status.lower() == 'start':
                    # Update session folder based on keyword if available
                    if keyword:
                        sanitized_keyword = sanitize_folder_name(keyword)
                        # Update if keyword is valid and different (or just ensure it's set)
                        if sanitized_keyword and sanitized_keyword != current_title:
                            current_title = sanitized_keyword
                            # Ensure we have a writer name (should be set in main)
                            writer_name_to_use = current_writer_name if current_writer_name else sanitize_folder_name(name)
                            session_folder = f"{writer_name_to_use}/{current_title}"
                            logger.info(f"[POLLER] Updated session folder to: {session_folder}")

                    if not tracking_active.is_set():
                        tracking_active.set()
                        msg = f"[POLLER] Received 'Start' command - Tracking STARTED"
                        logger.info(msg)
                        print(msg)
                elif work_status and work_status.lower() == 'stop':
                    if tracking_active.is_set():
                        tracking_active.clear()
                        msg = f"[POLLER] Received 'Stop' command - Tracking STOPPED"
                        logger.info(msg)
                        print(msg)
            elif response.status_code == 404:
                msg = f"[POLLER] Error: Endpoint not found (404)"
                logger.warning(msg)
                print(msg)
            else:
                msg = f"[POLLER] Failed to get status: HTTP {response.status_code}"
                logger.warning(msg)
                print(msg)
                
        except Exception as e:
            msg = f"[POLLER] Error polling status: {e}"
            logger.error(msg)
            print(msg)
            # Brief sleep on error to avoid rapid looping
            time.sleep(2)
        
        time.sleep(2)  # Poll every 2 seconds
    
    stop_msg = "[POLLER] Polling worker stopped"
    logger.info(stop_msg)
    print(stop_msg)


# ─── MAIN ─────────────────────────────────────────────────────────────

def main():
    global running
    
    # Check if DISPLAY is set (required for screenshot capture on Linux)
    if platform.system() == 'Linux' and not os.environ.get('DISPLAY'):
        print("\n" + "="*70)
        print("ERROR: $DISPLAY environment variable is not set!")
        print("="*70)
        print("\nThe tracker cannot capture screenshots without access to the X11 display.")
        print("\nTo fix this issue, you have two options:\n")
        print("Option 1: If you're in a terminal, export the DISPLAY variable:")
        print("  export DISPLAY=:0")
        print("  python3 main.py\n")
        print("Option 2: Run the tracker from a desktop environment GUI terminal.\n")
        print("Note: The tracker MUST be run on the machine being monitored,")
        print("      not via SSH unless you're using X11 forwarding.\n")
        print("="*70)
        logger.error("$DISPLAY not set - cannot capture screenshots")
        sys.exit(1)
    
    # NEW: Sync tracker_status to database on script startup (False - wait for start command)
    sync_db_tracker_status(True) 

    
    # Set running flag (means the app is alive, not actively capturing)
    running.set()
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check server connectivity (silently, log only)
    is_connected, error_msg = check_server_connectivity()
    if is_connected:
        logger.info(f"Server connectivity check: SUCCESS")
    else:
        logger.error(f"Server connectivity check: FAILED - {error_msg}")
    
    logger.info(f"Rate: {SCREENSHOTS_PER_SECOND} screenshots per second")
    logger.info(f"Temp directory: {TEMP_DIR}")

    
    # Check for leftover files in temp directory (log only)
    leftover_files = []
    if os.path.exists(TEMP_DIR):
        leftover_files = [
            f for f in os.listdir(TEMP_DIR)
            if f.startswith("screenshot_") and (f.endswith(".png") or f.endswith(".webp"))
        ]
        if leftover_files:
            logger.info(f"Found {len(leftover_files)} leftover files in temp directory")
    
    # Check retry queue stats (log only)
    stats = retry_queue.get_queue_stats()
    if stats['total_pending'] > 0:
        logger.info(f"Retry queue: {stats['memory_queue_size']} in memory, {stats['disk_queue_size']} on disk")
    
    # Number of upload worker threads for parallel uploads
    NUM_UPLOAD_WORKERS = 8  # 8 parallel upload workers for maximum throughput
    
    # Start disk write worker thread (writes files to disk asynchronously)
    disk_write_thread = threading.Thread(target=disk_write_worker, daemon=True)
    disk_write_thread.start()
    logger.info("Disk write worker thread started")
    
    # Start multiple upload worker threads for parallel uploads (faster upload throughput)
    upload_threads = []
    for i in range(NUM_UPLOAD_WORKERS):
        upload_thread = threading.Thread(target=upload_worker, daemon=True)
        upload_thread.start()
        upload_threads.append(upload_thread)
        logger.info(f"Upload worker thread {i+1} started")
    
    # Start retry worker thread (handles failed uploads)
    retry_thread = threading.Thread(target=retry_worker, daemon=True)
    retry_thread.start()
    logger.info("Retry worker thread started")


    
    # --- STANDALONE INITIALIZATION ---
    # Determine writer and session info from filename
    name, writer_id = get_writer_info_from_filename()
    if writer_id:
        global current_writer_name, current_title, session_folder, session_start_time
        current_writer_name = sanitize_folder_name(name)
        # Use a timestamped title since we aren't receiving one from the dashboard
        timestamp_str = datetime.now(ZONE).strftime("%Y%m%d_%H%M%S")
        current_title = f"Session_{timestamp_str}"
        session_folder = f"{current_writer_name}/{current_title}"
        session_start_time = datetime.now(ZONE).isoformat()
        
        # Upload metadata for this session
        try:
            metadata = collect_system_metadata(name, current_title)
            metadata_json = json.dumps(metadata, indent=2).encode("utf-8")
            upload_file_data(io.BytesIO(metadata_json), "metadata.json", session_folder)
            logger.info(f"Session metadata uploaded for {name} - {current_title}")
        except Exception as e:
            logger.error(f"Failed to upload session metadata: {e}")
            
        # Start status polling worker
        polling_thread = threading.Thread(target=status_polling_worker, args=(writer_id, name), daemon=True)
        polling_thread.start()
        logger.info(f"Status polling worker started for {name} (ID: {writer_id})")

    else:
        logger.error("Could not determine Writer ID from filename. Tracking will wait for manual activation (if any).")
    

    
    try:
        screenshot_capture_loop()
    except KeyboardInterrupt:
        # This should not happen now due to signal handler, but keep as fallback
        if not shutdown_in_progress.is_set():
            shutdown_in_progress.set()
            tracking_active.clear()
            running.clear()
            print("\nShutdown initiated. Waiting for all uploads to complete...")
        logger.info("Interrupted by user")
    finally:
        # Stop capture loop but keep upload workers running
        if not shutdown_in_progress.is_set():
            shutdown_in_progress.set()
            tracking_active.clear()
            running.clear()
        logger.info("Stopping screenshot capture...")
        
        # Wait for disk write queue to empty first (silently)
        logger.info("Waiting for disk writes to complete...")
        while disk_write_queue.qsize() > 0:
            time.sleep(1)
        
        # Wait for upload queue to empty (no time limit - wait until done, silently)
        logger.info("Waiting for all uploads to complete...")
        last_queue_size = -1
        last_temp_count = -1
        last_pending = -1
        last_update_time = time.time()
        
        # Signal handler during shutdown - ignore all Ctrl+C presses
        def shutdown_signal_handler(signum, frame):
            """Handle Ctrl+C during shutdown - ignore all presses until uploads complete."""
            print("\nShutdown in progress. Please wait for all uploads to complete...")
            logger.info("Ctrl+C pressed during shutdown (ignored - waiting for uploads to complete)")
        
        # Set the shutdown signal handler to ignore all Ctrl+C
        signal.signal(signal.SIGINT, shutdown_signal_handler)
        
        while True:
            queue_size = upload_queue.qsize()
            
            # Check temp directory
            temp_files = []
            if os.path.exists(TEMP_DIR):
                temp_files = [
                    f for f in os.listdir(TEMP_DIR)
                    if f.startswith("screenshot_") and (f.endswith(".png") or f.endswith(".webp"))
                ]
            
            # Check retry queue
            stats = retry_queue.get_queue_stats()
            total_pending = stats['total_pending']
            
            # Log progress every 3 seconds (no console output)
            current_time = time.time()
            if (current_time - last_update_time >= 3) or \
               (abs(queue_size - last_queue_size) > 10) or \
               (abs(len(temp_files) - last_temp_count) > 10) or \
               (abs(total_pending - last_pending) > 5):
                
                status_parts = []
                if queue_size > 0:
                    status_parts.append(f"{queue_size} uploading")
                if len(temp_files) > 0:
                    status_parts.append(f"{len(temp_files)} in temp")
                if total_pending > 0:
                    status_parts.append(f"{total_pending} retrying")
                
                if status_parts:
                    logger.info(f"Shutdown progress: {', '.join(status_parts)}")
                
                last_queue_size = queue_size
                last_temp_count = len(temp_files)
                last_pending = total_pending
                last_update_time = current_time
            
            # Check if everything is done
            if queue_size == 0 and len(temp_files) == 0 and total_pending == 0:
                # Double-check after a brief pause
                time.sleep(1)
                final_check_queue = upload_queue.qsize()
                final_check_temp = []
                if os.path.exists(TEMP_DIR):
                    final_check_temp = [
                        f for f in os.listdir(TEMP_DIR)
                        if f.startswith("screenshot_") and (f.endswith(".png") or f.endswith(".webp"))
                    ]
                final_check_stats = retry_queue.get_queue_stats()
                
                if final_check_queue == 0 and len(final_check_temp) == 0 and final_check_stats['total_pending'] == 0:
                    logger.info("All uploads completed successfully")
                    print("All uploads completed. Exiting...")
                    break
            
            time.sleep(1)  # Check every second
        
        # Final statistics (log only)
        if capture_start_time:
            elapsed_total = time.time() - capture_start_time
            total_captured = screenshot_count
            if elapsed_total > 0:
                final_rate = total_captured / elapsed_total
                logger.info(f"Session stats: {total_captured} screenshots in {elapsed_total:.1f}s = {final_rate:.1f}/sec")
        
        # NEW: Sync tracker_status to database on script shutdown (False)
        sync_db_tracker_status(False)


if __name__ == '__main__':
    main()