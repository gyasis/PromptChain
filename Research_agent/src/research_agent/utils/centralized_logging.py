"""
Centralized Logging System for Research Agent

Captures ALL logs (console, debug, errors) into a single file for complete traceability.
"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


class CentralizedLogger:
    """Centralized logging system that captures everything to a single file"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize centralized logging only once"""
        if not CentralizedLogger._initialized:
            self._setup_logging()
            CentralizedLogger._initialized = True
    
    def _setup_logging(self):
        """Set up comprehensive logging configuration"""
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Generate unique log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_dir / f"research_agent_{timestamp}.log"
        
        # Remove all existing handlers to start fresh
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create comprehensive formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler - captures EVERYTHING
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Capture all levels
        file_handler.setFormatter(formatter)
        
        # Console handler - for user visibility
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)  # Less verbose for console
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        root_logger.setLevel(logging.DEBUG)  # Capture everything
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Capture warnings
        logging.captureWarnings(True)
        
        # Redirect stdout and stderr to logging
        self._redirect_output_streams()
        
        # Log initialization
        root_logger.info("="*80)
        root_logger.info(f"CENTRALIZED LOGGING INITIALIZED")
        root_logger.info(f"Log file: {self.log_file.absolute()}")
        root_logger.info("="*80)
    
    def _redirect_output_streams(self):
        """Redirect stdout and stderr to logging system"""
        
        class StreamToLogger:
            """Redirect stream output to logger"""
            def __init__(self, logger, log_level, original_stream):
                self.logger = logger
                self.log_level = log_level
                self.original_stream = original_stream
                self.linebuf = ''
            
            def write(self, buf):
                # Avoid infinite recursion - don't log logging errors
                if "--- Logging error ---" in buf:
                    return
                    
                for line in buf.rstrip().splitlines():
                    if line.strip():  # Skip empty lines
                        try:
                            self.logger.log(self.log_level, line.rstrip())
                        except:
                            # If logging fails, write to original stream
                            self.original_stream.write(line + '\n')
            
            def flush(self):
                pass
        
        # Create loggers for stdout and stderr
        stdout_logger = logging.getLogger('STDOUT')
        stderr_logger = logging.getLogger('STDERR')
        
        # Save original streams
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        # Only redirect if not already redirected
        if not isinstance(sys.stdout, StreamToLogger):
            sys.stdout = StreamToLogger(stdout_logger, logging.INFO, original_stdout)
        if not isinstance(sys.stderr, StreamToLogger):
            sys.stderr = StreamToLogger(stderr_logger, logging.ERROR, original_stderr)
    
    def get_log_file(self) -> Path:
        """Get the current log file path"""
        return self.log_file
    
    def log_section(self, title: str, level: int = logging.INFO):
        """Log a section separator for better organization"""
        logger = logging.getLogger()
        logger.log(level, "")
        logger.log(level, "="*80)
        logger.log(level, f" {title}")
        logger.log(level, "="*80)
    
    def log_error_with_context(self, error: Exception, context: str = ""):
        """Log an error with full context and traceback"""
        import traceback
        logger = logging.getLogger()
        
        logger.error(f"ERROR CONTEXT: {context}")
        logger.error(f"ERROR TYPE: {type(error).__name__}")
        logger.error(f"ERROR MESSAGE: {str(error)}")
        logger.error("TRACEBACK:")
        for line in traceback.format_exc().splitlines():
            logger.error(f"  {line}")


def setup_centralized_logging() -> CentralizedLogger:
    """
    Set up centralized logging for the Research Agent
    
    Returns:
        CentralizedLogger instance
    """
    return CentralizedLogger()


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured logger instance
    """
    # Ensure centralized logging is set up
    setup_centralized_logging()
    return logging.getLogger(name)