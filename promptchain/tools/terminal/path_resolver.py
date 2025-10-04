"""
Path Resolution Module for Terminal Tool

Provides intelligent executable path resolution with multi-tier search strategy:
- Environment-specific paths (activated environments)
- System PATH search  
- Common installation locations
- Cached path lookup for performance

Author: PromptChain Team
License: MIT
"""

import os
import shutil
import platform
from pathlib import Path
from typing import Dict, List, Optional, Set
import stat


class PathResolver:
    """Resolves executable paths with multi-tier search strategy and caching"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize path resolver
        
        Args:
            config: Configuration dictionary with path settings
        """
        self.config = config or {}
        self.cache = {}
        self.failed_cache = set()  # Cache of paths we know don't exist
        
        # Configuration options
        self.enable_cache = self.config.get('enable_path_cache', True)
        self.cache_ttl = self.config.get('cache_ttl_seconds', 300)  # 5 minutes
        self.custom_search_paths = self.config.get('custom_search_paths', [])
        self.priority_paths = self.config.get('priority_paths', [])
        
        # Platform-specific settings
        self.is_windows = platform.system() == 'Windows'
        self.path_separator = ';' if self.is_windows else ':'
        self.executable_extensions = self._get_executable_extensions()
        
        # Common installation paths by platform
        self.common_paths = self._get_common_paths()
        
    def _get_executable_extensions(self) -> List[str]:
        """Get list of executable extensions for the current platform"""
        if self.is_windows:
            # Windows executable extensions
            pathext = os.environ.get('PATHEXT', '.COM;.EXE;.BAT;.CMD;.VBS;.JS')
            return [ext.lower() for ext in pathext.split(';') if ext]
        else:
            # Unix-like systems don't typically use extensions for executables
            return ['']
    
    def _get_common_paths(self) -> List[Path]:
        """Get common installation paths for the current platform"""
        if self.is_windows:
            common_paths = [
                Path('C:/Windows/System32'),
                Path('C:/Windows'),
                Path('C:/Program Files/Git/bin'),
                Path('C:/Program Files/Git/cmd'),
                Path('C:/msys64/usr/bin'),
                Path('C:/msys64/mingw64/bin'),
                Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'Python' / 'Python39' / 'Scripts' if os.environ.get('LOCALAPPDATA') else None,
                Path(os.environ.get('APPDATA', '')) / 'npm' if os.environ.get('APPDATA') else None,
            ]
        else:
            common_paths = [
                Path('/usr/local/bin'),
                Path('/usr/bin'),
                Path('/bin'),
                Path('/usr/local/sbin'),
                Path('/usr/sbin'),
                Path('/sbin'),
                Path('/opt/homebrew/bin'),  # macOS Homebrew on Apple Silicon
                Path('/usr/local/homebrew/bin'),  # macOS Homebrew Intel
                Path('/snap/bin'),  # Ubuntu Snap packages
                Path('/usr/games'),
                Path('/usr/local/games'),
            ]
            
        # Add user-specific paths
        home = Path.home()
        user_paths = [
            home / '.local' / 'bin',
            home / 'bin',
            home / '.cargo' / 'bin',  # Rust cargo
            home / 'go' / 'bin',      # Go binaries
            home / '.npm-global' / 'bin',  # npm global
            home / '.gem' / 'ruby' / 'bin',  # Ruby gems
        ]
        
        if not self.is_windows:
            common_paths.extend(user_paths)
        
        # Filter out None values and non-existent paths for performance
        return [path for path in common_paths if path and path.exists()]
    
    def resolve_executable(self, command: str) -> Optional[str]:
        """
        Resolve the full path to an executable using multi-tier search
        
        Args:
            command: The command/executable name to resolve
            
        Returns:
            Full path to executable if found, None otherwise
        """
        if not command:
            return None
            
        # Extract the base command (first word) from complex commands
        base_cmd = command.split()[0] if ' ' in command else command
        
        # Check cache first
        if self.enable_cache and base_cmd in self.cache:
            cached_path = self.cache[base_cmd]
            # Verify cached path still exists
            if Path(cached_path).exists():
                return cached_path
            else:
                # Remove stale cache entry
                del self.cache[base_cmd]
                
        # Check failed cache to avoid repeated failed searches
        if base_cmd in self.failed_cache:
            return None
            
        # Multi-tier search strategy
        resolved_path = (
            self._check_absolute_path(base_cmd) or
            self._search_priority_paths(base_cmd) or  
            self._search_system_path(base_cmd) or
            self._search_environment_paths(base_cmd) or
            self._search_common_paths(base_cmd) or
            self._search_custom_paths(base_cmd)
        )
        
        # Update cache
        if resolved_path:
            if self.enable_cache:
                self.cache[base_cmd] = resolved_path
            return resolved_path
        else:
            # Cache failed lookup to avoid repeated searches
            self.failed_cache.add(base_cmd)
            return None
    
    def _check_absolute_path(self, command: str) -> Optional[str]:
        """Check if command is already an absolute path"""
        path = Path(command)
        if path.is_absolute() and self._is_executable(path):
            return str(path)
        return None
    
    def _search_priority_paths(self, command: str) -> Optional[str]:
        """Search in configured priority paths first"""
        for priority_path in self.priority_paths:
            path = Path(priority_path)
            if path.exists() and path.is_dir():
                executable_path = self._find_executable_in_dir(path, command)
                if executable_path:
                    return executable_path
        return None
    
    def _search_system_path(self, command: str) -> Optional[str]:
        """Search using system PATH via shutil.which()"""
        # Use shutil.which() for standard PATH search
        resolved = shutil.which(command)
        if resolved:
            return resolved
            
        # On Windows, also try with different extensions
        if self.is_windows:
            for ext in self.executable_extensions:
                if ext and not command.lower().endswith(ext):
                    resolved = shutil.which(command + ext)
                    if resolved:
                        return resolved
                        
        return None
    
    def _search_environment_paths(self, command: str) -> Optional[str]:
        """Search in environment-specific paths (activated conda, venv, etc.)"""
        env_paths = []
        
        # Check for activated conda environment
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            env_paths.extend([
                Path(conda_prefix) / 'bin',
                Path(conda_prefix) / 'Scripts',  # Windows conda
                Path(conda_prefix) / 'Library' / 'bin',  # Windows conda libraries
            ])
            
        # Check for activated virtual environment
        virtual_env = os.environ.get('VIRTUAL_ENV')
        if virtual_env:
            env_paths.extend([
                Path(virtual_env) / 'bin',
                Path(virtual_env) / 'Scripts',  # Windows venv
            ])
            
        # Check for activated NVM
        nvm_current = os.environ.get('NVM_CURRENT')
        if nvm_current:
            nvm_dir = os.environ.get('NVM_DIR', str(Path.home() / '.nvm'))
            env_paths.append(Path(nvm_dir) / 'versions' / 'node' / nvm_current / 'bin')
            
        # Search in environment paths
        for env_path in env_paths:
            if env_path.exists() and env_path.is_dir():
                executable_path = self._find_executable_in_dir(env_path, command)
                if executable_path:
                    return executable_path
                    
        return None
    
    def _search_common_paths(self, command: str) -> Optional[str]:
        """Search in common installation locations"""
        for common_path in self.common_paths:
            if common_path.exists() and common_path.is_dir():
                executable_path = self._find_executable_in_dir(common_path, command)
                if executable_path:
                    return executable_path
        return None
    
    def _search_custom_paths(self, command: str) -> Optional[str]:
        """Search in custom configured paths"""
        for custom_path in self.custom_search_paths:
            path = Path(custom_path)
            if path.exists() and path.is_dir():
                executable_path = self._find_executable_in_dir(path, command)
                if executable_path:
                    return executable_path
        return None
    
    def _find_executable_in_dir(self, directory: Path, command: str) -> Optional[str]:
        """Find an executable in a specific directory"""
        # Direct match
        executable_path = directory / command
        if self._is_executable(executable_path):
            return str(executable_path)
            
        # Try with extensions on Windows
        if self.is_windows:
            for ext in self.executable_extensions:
                if ext:
                    extended_path = directory / (command + ext)
                    if self._is_executable(extended_path):
                        return str(extended_path)
                        
        return None
    
    def _is_executable(self, path: Path) -> bool:
        """Check if a path points to an executable file"""
        if not path.exists() or not path.is_file():
            return False
            
        if self.is_windows:
            # On Windows, check file extension
            return path.suffix.lower() in [ext.lower() for ext in self.executable_extensions if ext]
        else:
            # On Unix-like systems, check execute permission
            try:
                return os.access(path, os.X_OK)
            except (OSError, PermissionError):
                return False
    
    def validate_executable(self, path: str) -> Dict[str, any]:
        """
        Validate an executable and return detailed information
        
        Args:
            path: Path to the executable
            
        Returns:
            Dictionary with validation results and metadata
        """
        path_obj = Path(path)
        
        result = {
            'path': path,
            'exists': False,
            'is_file': False,
            'is_executable': False,
            'is_readable': False,
            'size': 0,
            'permissions': None,
            'error': None
        }
        
        try:
            if path_obj.exists():
                result['exists'] = True
                result['is_file'] = path_obj.is_file()
                
                if result['is_file']:
                    stat_info = path_obj.stat()
                    result['size'] = stat_info.st_size
                    result['permissions'] = oct(stat_info.st_mode)
                    
                    # Check permissions
                    result['is_readable'] = os.access(path, os.R_OK)
                    result['is_executable'] = self._is_executable(path_obj)
                    
        except (OSError, PermissionError) as e:
            result['error'] = str(e)
            
        return result
    
    def get_command_info(self, command: str) -> Dict[str, any]:
        """
        Get comprehensive information about a command
        
        Args:
            command: Command to analyze
            
        Returns:
            Dictionary with command information and resolution details
        """
        base_cmd = command.split()[0] if ' ' in command else command
        resolved_path = self.resolve_executable(base_cmd)
        
        info = {
            'command': command,
            'base_command': base_cmd,
            'resolved_path': resolved_path,
            'found': resolved_path is not None,
            'search_attempted': True,
            'cached': base_cmd in self.cache,
            'validation': None
        }
        
        if resolved_path:
            info['validation'] = self.validate_executable(resolved_path)
            
        return info
    
    def clear_cache(self):
        """Clear the executable path cache"""
        self.cache.clear()
        self.failed_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'cache_hits': len(self.cache),
            'cache_misses': len(self.failed_cache),
            'total_lookups': len(self.cache) + len(self.failed_cache)
        }
    
    def add_custom_path(self, path: str):
        """Add a custom search path"""
        if path not in self.custom_search_paths:
            self.custom_search_paths.append(path)
    
    def remove_custom_path(self, path: str):
        """Remove a custom search path"""
        if path in self.custom_search_paths:
            self.custom_search_paths.remove(path)
    
    def list_search_paths(self) -> List[str]:
        """Get list of all search paths in order of priority"""
        paths = []
        
        # Priority paths
        paths.extend(self.priority_paths)
        
        # System PATH
        system_path = os.environ.get('PATH', '')
        if system_path:
            paths.extend(system_path.split(self.path_separator))
        
        # Environment paths
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            paths.append(str(Path(conda_prefix) / 'bin'))
            
        virtual_env = os.environ.get('VIRTUAL_ENV')
        if virtual_env:
            paths.append(str(Path(virtual_env) / 'bin'))
            
        # Common paths
        paths.extend([str(path) for path in self.common_paths])
        
        # Custom paths
        paths.extend(self.custom_search_paths)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for path in paths:
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)
                
        return unique_paths
    
    def diagnose_command(self, command: str) -> str:
        """
        Provide diagnostic information for command resolution
        
        Args:
            command: Command to diagnose
            
        Returns:
            Formatted diagnostic report
        """
        base_cmd = command.split()[0] if ' ' in command else command
        info = self.get_command_info(command)
        
        report = [
            f"Command Diagnostic Report: {command}",
            "=" * 50,
            f"Base command: {base_cmd}",
            f"Found: {'Yes' if info['found'] else 'No'}",
        ]
        
        if info['resolved_path']:
            report.append(f"Resolved path: {info['resolved_path']}")
            
            validation = info['validation']
            report.extend([
                f"File exists: {validation['exists']}",
                f"Is executable: {validation['is_executable']}",
                f"File size: {validation['size']} bytes",
                f"Permissions: {validation['permissions']}",
            ])
            
            if validation['error']:
                report.append(f"Error: {validation['error']}")
        else:
            report.append("Resolution failed - command not found in any search path")
            
        report.extend([
            "",
            "Search paths (in order):",
        ])
        
        for i, path in enumerate(self.list_search_paths(), 1):
            exists = "✓" if Path(path).exists() else "✗"
            report.append(f"  {i:2d}. {exists} {path}")
            
        cache_stats = self.get_cache_stats()
        report.extend([
            "",
            f"Cache statistics:",
            f"  Cache hits: {cache_stats['cache_hits']}",
            f"  Cache misses: {cache_stats['cache_misses']}",
            f"  Total lookups: {cache_stats['total_lookups']}"
        ])
        
        return "\n".join(report)