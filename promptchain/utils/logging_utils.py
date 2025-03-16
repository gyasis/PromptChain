"""Logging utilities for PromptChain."""

import os
import json
from datetime import datetime
from typing import Any, Dict
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RunLogger:
    def __init__(self, log_dir: str = "logs"):
        """Initialize the run logger.
        
        Args:
            log_dir: Directory to store log files (default: "logs")
        """
        self.log_dir = log_dir
        self._ensure_log_dir()
    
    def _ensure_log_dir(self):
        """Create the log directory if it doesn't exist."""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            logger.info(f"Created log directory: {self.log_dir}")
    
    def _get_log_filename(self) -> str:
        """Generate a log filename with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.log_dir, f"run_{timestamp}.json")
    
    def log_run(self, run_data: Dict[str, Any]):
        """Log a run with its data.
        
        Args:
            run_data: Dictionary containing run information
        """
        # Add timestamp to run data
        run_data["timestamp"] = datetime.now().isoformat()
        
        # Generate filename and save
        filename = self._get_log_filename()
        try:
            with open(filename, 'w') as f:
                json.dump(run_data, f, indent=2)
            logger.info(f"Run logged to: {filename}")
        except Exception as e:
            logger.error(f"Error logging run to {filename}: {str(e)}")
    
    def get_recent_runs(self, n: int = 5) -> list:
        """Get the n most recent runs.
        
        Args:
            n: Number of recent runs to retrieve (default: 5)
            
        Returns:
            List of run data dictionaries
        """
        try:
            # Get all log files
            files = [f for f in os.listdir(self.log_dir) if f.startswith("run_") and f.endswith(".json")]
            # Sort by timestamp (newest first)
            files.sort(reverse=True)
            
            # Load the n most recent files
            recent_runs = []
            for f in files[:n]:
                with open(os.path.join(self.log_dir, f), 'r') as file:
                    recent_runs.append(json.load(file))
            return recent_runs
        except Exception as e:
            logger.error(f"Error retrieving recent runs: {str(e)}")
            return [] 