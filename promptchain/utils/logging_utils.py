"""Logging utilities for PromptChain."""

import os
import json
from datetime import datetime
from typing import Any, Dict, Optional
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use the name of the current module

class RunLogger:
    def __init__(self, log_dir: Optional[str] = None, session_filename: Optional[str] = None):
        """Initialize the run logger.

        Console logging of summaries is always enabled.
        File logging (JSONL) is enabled if a log_dir is provided.

        Args:
            log_dir: Optional directory to store detailed JSONL log files.
                     If None, file logging is disabled.
            session_filename: Optional filename for session-based logging. If provided, all logs go to this file.
        """
        self.log_dir = log_dir
        self.session_filename = session_filename
        if self.log_dir:
            self._ensure_log_dir()
        if self.log_dir and not self.session_filename:
            # If no session filename provided, generate one at init (for backward compatibility)
            self.session_filename = self._get_log_filename()
    
    def _ensure_log_dir(self):
        """Create the log directory if it doesn't exist."""
        # This is only called if self.log_dir is not None
        if not os.path.exists(self.log_dir):
            try:
                os.makedirs(self.log_dir)
                logger.info(f"Created log directory for file logging: {self.log_dir}")
            except OSError as e:
                logger.error(f"Failed to create log directory {self.log_dir}: {e}. File logging disabled.")
                self.log_dir = None # Disable file logging if dir creation fails
    
    def _get_log_filename(self) -> Optional[str]:
        """Generate a log filename with timestamp if file logging is enabled."""
        if not self.log_dir:
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") # Add microseconds for uniqueness
        return os.path.join(self.log_dir, f"run_{timestamp}.jsonl") # Use .jsonl for line-based JSON
    
    def set_session_filename(self, session_filename: str):
        """Set the session log filename (for session-based logging)."""
        self.session_filename = session_filename

    def log_run(self, run_data: Dict[str, Any]):
        """Log a run summary to console and optionally full data to a file.
        
        Args:
            run_data: Dictionary containing run information
        """
        # Add timestamp to run data
        run_data["timestamp"] = datetime.now().isoformat()
        
        # 1. Log to Console (Always)
        try:
            # Create a concise summary for console logging
            event = run_data.get('event', 'unknown_event')
            details = {k: v for k, v in run_data.items() if k not in ['timestamp', 'event'] and v is not None}
            # Truncate long strings for readability
            for key, value in details.items():
                if isinstance(value, str) and len(value) > 100:
                    details[key] = value[:100] + "..."
            # Determine log level based on event type (optional)
            level = logging.INFO
            if 'error' in event.lower() or 'fail' in event.lower():
                level = logging.ERROR
            elif 'warn' in event.lower():
                level = logging.WARNING

            log_message = f"[RunLog] Event: {event} | Details: {json.dumps(details)}"
            logger.log(level, log_message)
        except Exception as console_log_e:
            # Log the error itself to the console logger
            logger.error(f"Error formatting log for console: {console_log_e}")
        
        # 2. Log to JSONL File (Only if log_dir is set)
        if self.log_dir:
            filename = self.session_filename or self._get_log_filename()
            if filename:
                try:
                    # Use 'a' for append mode with JSON Lines format
                    with open(filename, 'a') as f:
                        json.dump(run_data, f)
                        f.write('\n') # Add newline for JSONL format
                except Exception as e:
                    logger.error(f"Error logging run to file {filename}: {str(e)}")
    
    def get_recent_runs(self, n: int = 5) -> list:
        """Get the n most recent runs (reads last n lines from the latest file).
        
        Note: This implementation is simplified for JSONL. It reads the *latest file's*
        last n lines, not necessarily the absolute last n log entries across all files.
        A more robust implementation might Tail the file or use a different storage method.
        
        Args:
            n: Number of recent run entries to retrieve (default: 5)
            
        Returns:
            List of run data dictionaries, or empty list if file logging is disabled or error occurs.
        """
        if not self.log_dir:
            logger.info("File logging is disabled, cannot get recent runs from file.")
            return []

        try:
            # Find the latest log file
            files = [os.path.join(self.log_dir, f) for f in os.listdir(self.log_dir) if f.startswith("run_") and f.endswith(".jsonl")]
            if not files:
                return []
            latest_file = max(files, key=os.path.getctime)
            
            # Read the last n lines (or fewer if file is short)
            recent_runs = []
            with open(latest_file, 'r') as f:
                # Read all lines, then take the last n
                lines = f.readlines()
                for line in lines[-n:]:
                    if line.strip(): # Avoid empty lines
                        try:
                            recent_runs.append(json.loads(line))
                        except json.JSONDecodeError as json_err:
                            logger.warning(f"Could not parse line in {latest_file}: {json_err} - Line: '{line.strip()}'")
            return recent_runs
        except FileNotFoundError:
            logger.warning(f"Log directory or file not found when retrieving recent runs: {self.log_dir}")
            return []
        except Exception as e:
            logger.error(f"Error retrieving recent runs: {str(e)}")
            return [] 