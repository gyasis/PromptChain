"""
Logging utilities for the UseCase Validator
"""
import logging
import os
import sys
from datetime import datetime

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Configure logger level
        debug_mode = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
        level = logging.DEBUG if debug_mode else logging.INFO
        logger.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler for detailed logs
        os.makedirs('logs', exist_ok=True)
        file_handler = logging.FileHandler(
            f'logs/usecase_validator_{datetime.now().strftime("%Y%m%d")}.log'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Prevent duplicate logs
        logger.propagate = False
    
    return logger



