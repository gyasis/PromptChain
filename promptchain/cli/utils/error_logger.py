"""Error logging utilities for PromptChain CLI.

This module provides structured error logging to JSONL session files
for debugging and observability (T143).
"""

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ErrorLogger:
    """Logs errors to session JSONL files for debugging.

    Captures comprehensive error information including:
    - Timestamp
    - Error type (class name)
    - Error message
    - Stack trace
    - Context (what operation was being performed)
    - Additional metadata

    Logs are written to: ~/.promptchain/sessions/<session-id>/errors.jsonl
    """

    def __init__(self, session_dir: Path):
        """Initialize error logger for a specific session.

        Args:
            session_dir: Path to session directory (e.g., ~/.promptchain/sessions/<session-id>)
        """
        self.session_dir = Path(session_dir)
        self.error_log_path = self.session_dir / "errors.jsonl"

        # Ensure session directory exists
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def log_error(
        self,
        error: Exception,
        context: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        include_traceback: bool = True,
    ):
        """Log an error to the session's errors.jsonl file.

        Args:
            error: The exception that occurred
            context: Description of what was being done when error occurred
            metadata: Additional context data (e.g., user input, agent name)
            include_traceback: Whether to include full stack trace
        """
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "metadata": metadata or {},
        }

        # Add stack trace if requested
        if include_traceback:
            error_entry["stack_trace"] = traceback.format_exc()

        # Write to JSONL file (append mode)
        try:
            with open(self.error_log_path, "a") as f:
                json.dump(error_entry, f)
                f.write("\n")
        except Exception as log_error:
            # If we can't log to file, at least print to stderr
            import sys

            print(
                f"ERROR: Failed to log error to {self.error_log_path}: {log_error}",
                file=sys.stderr,
            )

    def log_error_dict(self, error_dict: Dict[str, Any]):
        """Log an error from a pre-formatted dictionary.

        Useful for logging errors that have already been structured
        (e.g., from Message.metadata).

        Args:
            error_dict: Dictionary with error information
        """
        # Ensure timestamp exists
        if "timestamp" not in error_dict:
            error_dict["timestamp"] = datetime.now().isoformat()

        # Write to JSONL file
        try:
            with open(self.error_log_path, "a") as f:
                json.dump(error_dict, f)
                f.write("\n")
        except Exception as log_error:
            import sys

            print(
                f"ERROR: Failed to log error dict to {self.error_log_path}: {log_error}",
                file=sys.stderr,
            )

    def get_recent_errors(self, n: int = 10) -> list:
        """Get the n most recent errors from the log.

        Args:
            n: Number of recent errors to retrieve

        Returns:
            List of error dictionaries
        """
        if not self.error_log_path.exists():
            return []

        try:
            with open(self.error_log_path, "r") as f:
                lines = f.readlines()
                errors = []
                for line in lines[-n:]:
                    if line.strip():
                        try:
                            errors.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                return errors
        except Exception:
            return []

    def clear_errors(self):
        """Clear all errors from the log file."""
        if self.error_log_path.exists():
            self.error_log_path.unlink()


def create_error_logger(session_id: str, sessions_dir: Optional[Path] = None) -> ErrorLogger:
    """Factory function to create an ErrorLogger for a session.

    Args:
        session_id: Session UUID
        sessions_dir: Base sessions directory (default: ~/.promptchain/sessions)

    Returns:
        ErrorLogger instance for the session
    """
    if sessions_dir is None:
        sessions_dir = Path.home() / ".promptchain" / "sessions"

    session_dir = sessions_dir / session_id
    return ErrorLogger(session_dir)
