"""
Cross-platform file locking for atomic configuration operations.

Provides safe file operations that prevent race conditions between CLI and web interfaces
when updating configuration files.
"""

import os
import time
import fcntl
import tempfile
from pathlib import Path
from contextlib import contextmanager
from typing import Union, Generator

import logging

logger = logging.getLogger(__name__)


class FileLockError(Exception):
    """Exception raised when file locking operations fail"""
    pass


@contextmanager  
def atomic_write(file_path: Union[str, Path], mode: str = 'w', encoding: str = 'utf-8', 
                backup: bool = True) -> Generator:
    """
    Context manager for atomic file writes with backup and rollback.
    
    Args:
        file_path: Path to the file to write
        mode: File open mode
        encoding: File encoding
        backup: Whether to create backup before writing
        
    Yields:
        File handle for writing
        
    Example:
        with atomic_write('config.yaml') as f:
            yaml.dump(data, f)
    """
    file_path = Path(file_path)
    temp_file = None
    backup_file = None
    
    try:
        # Create temporary file in same directory to ensure atomic move
        temp_fd, temp_path = tempfile.mkstemp(
            dir=file_path.parent,
            prefix=f'.{file_path.name}.tmp.',
            suffix=''
        )
        temp_file = Path(temp_path)
        
        # Create backup if file exists and backup is enabled
        if backup and file_path.exists():
            backup_file = file_path.with_suffix(f'{file_path.suffix}.backup')
            backup_file.write_bytes(file_path.read_bytes())
            logger.debug(f"Created backup: {backup_file}")
        
        # Yield file handle for writing
        with os.fdopen(temp_fd, mode, encoding=encoding) as f:
            yield f
            f.flush()
            os.fsync(f.fileno())  # Ensure data is written to disk
        
        # Atomic move (rename) - this is atomic on most filesystems
        if os.name == 'nt':  # Windows
            if file_path.exists():
                file_path.unlink()  # Windows requires removing target first
        
        temp_file.rename(file_path)
        logger.debug(f"Atomically updated: {file_path}")
        
        # Clean up backup file if successful
        if backup_file and backup_file.exists():
            backup_file.unlink()
            
    except Exception as e:
        logger.error(f"Atomic write failed for {file_path}: {e}")
        
        # Rollback: restore from backup if available
        if backup_file and backup_file.exists():
            try:
                backup_file.rename(file_path)
                logger.info(f"Restored from backup: {file_path}")
            except Exception as restore_error:
                logger.error(f"Failed to restore backup: {restore_error}")
        
        # Clean up temporary file
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass  # Best effort cleanup
                
        raise FileLockError(f"Atomic write failed: {e}") from e


@contextmanager
def file_lock(file_path: Union[str, Path], timeout: float = 10.0, exclusive: bool = True) -> Generator:
    """
    Cross-platform file locking context manager.
    
    Args:
        file_path: Path to file to lock
        timeout: Maximum time to wait for lock acquisition
        exclusive: Whether to use exclusive (write) lock vs shared (read) lock
        
    Yields:
        File handle with lock acquired
        
    Example:
        with file_lock('config.yaml', timeout=5.0) as f:
            data = yaml.safe_load(f)
    """
    file_path = Path(file_path)
    lock_file = None
    
    try:
        # Create lock file
        lock_path = file_path.with_suffix(f'{file_path.suffix}.lock')
        
        start_time = time.time()
        while True:
            try:
                # Try to create lock file exclusively
                lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                lock_file = os.fdopen(lock_fd, 'w')
                
                # Write PID to lock file for debugging
                lock_file.write(f"{os.getpid()}\n{time.time()}\n")
                lock_file.flush()
                
                logger.debug(f"Acquired lock: {lock_path}")
                break
                
            except OSError:
                # Lock file exists, check if it's stale
                if time.time() - start_time > timeout:
                    raise FileLockError(f"Timeout waiting for lock: {lock_path}")
                
                # Check if lock is stale (process no longer exists)
                if lock_path.exists():
                    try:
                        with open(lock_path, 'r') as f:
                            lines = f.readlines()
                            if len(lines) >= 2:
                                lock_pid = int(lines[0].strip())
                                lock_time = float(lines[1].strip())
                                
                                # Check if process is still running
                                try:
                                    os.kill(lock_pid, 0)  # Signal 0 checks if process exists
                                except (OSError, ProcessLookupError):
                                    # Process doesn't exist, remove stale lock
                                    logger.warning(f"Removing stale lock: {lock_path}")
                                    lock_path.unlink()
                                    continue
                    except (ValueError, IOError):
                        # Corrupted lock file, remove it
                        logger.warning(f"Removing corrupted lock: {lock_path}")
                        try:
                            lock_path.unlink()
                            continue
                        except OSError:
                            pass
                
                # Wait briefly before retrying
                time.sleep(0.1)
        
        # Open the actual file for reading/writing
        mode = 'r+' if file_path.exists() else 'w+'
        try:
            with open(file_path, mode, encoding='utf-8') as f:
                # Apply additional fcntl lock on Unix systems
                if hasattr(fcntl, 'LOCK_EX'):
                    lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
                    fcntl.flock(f.fileno(), lock_type | fcntl.LOCK_NB)
                    logger.debug(f"Applied fcntl lock: {file_path}")
                
                yield f
                
        except OSError as e:
            if e.errno == 11:  # EAGAIN - would block
                raise FileLockError(f"File is locked by another process: {file_path}")
            raise
            
    finally:
        # Release lock
        if lock_file:
            try:
                lock_file.close()
            except Exception:
                pass
                
        if lock_path and lock_path.exists():
            try:
                lock_path.unlink()
                logger.debug(f"Released lock: {lock_path}")
            except Exception as e:
                logger.warning(f"Failed to remove lock file: {e}")


class ConfigurationLock:
    """
    High-level configuration file locking manager.
    
    Provides convenient methods for reading and writing configuration files
    with proper locking to prevent race conditions between CLI and web interfaces.
    """
    
    def __init__(self, config_path: Union[str, Path], lock_timeout: float = 10.0):
        self.config_path = Path(config_path)
        self.lock_timeout = lock_timeout
    
    def read_config(self) -> str:
        """
        Read configuration file with shared lock.
        
        Returns:
            Configuration file contents as string
        """
        try:
            with file_lock(self.config_path, timeout=self.lock_timeout, exclusive=False) as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {self.config_path}")
            return ""
    
    def write_config(self, content: str, backup: bool = True) -> None:
        """
        Write configuration file with exclusive lock and atomic operation.
        
        Args:
            content: Configuration content to write
            backup: Whether to create backup before writing
        """
        # Ensure parent directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with atomic_write(self.config_path, backup=backup) as f:
            f.write(content)
            
        logger.info(f"Configuration updated: {self.config_path}")
    
    def update_config(self, update_func, backup: bool = True):
        """
        Update configuration file atomically using update function.
        
        Args:
            update_func: Function that takes current content and returns updated content
            backup: Whether to create backup before writing
        """
        with file_lock(self.config_path, timeout=self.lock_timeout, exclusive=True) as f:
            current_content = f.read()
            updated_content = update_func(current_content)
            
            if updated_content != current_content:
                # Write updated content atomically
                with atomic_write(self.config_path, backup=backup) as write_f:
                    write_f.write(updated_content)
                    
                logger.info(f"Configuration updated via function: {self.config_path}")
            else:
                logger.debug("No changes to configuration")
                
    def is_locked(self) -> bool:
        """
        Check if configuration file is currently locked.
        
        Returns:
            True if file is locked, False otherwise
        """
        lock_path = self.config_path.with_suffix(f'{self.config_path.suffix}.lock')
        return lock_path.exists()
    
    def force_unlock(self) -> bool:
        """
        Force remove lock file (use with caution).
        
        Returns:
            True if lock was removed, False if no lock existed
        """
        lock_path = self.config_path.with_suffix(f'{self.config_path.suffix}.lock')
        if lock_path.exists():
            try:
                lock_path.unlink()
                logger.warning(f"Forcibly removed lock: {lock_path}")
                return True
            except OSError as e:
                logger.error(f"Failed to remove lock: {e}")
                raise FileLockError(f"Cannot remove lock: {e}")
        return False