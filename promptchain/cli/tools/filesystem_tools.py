"""Filesystem utility tools for PromptChain CLI.

Provides path resolution and filesystem utilities that always work with
absolute paths to avoid ambiguity and AI security restrictions.

T119: Path resolution tools for absolute path enforcement.

Security modes:
- STRICT: Always warn on boundary violations, require confirmation
- TRUSTED: No boundary warnings (user trusts the session)
- DEFAULT: Warn once per unique outside-path, then allow
"""

import os
import glob as glob_module
from pathlib import Path
from typing import List, Optional, Union, Tuple

from .registry import ToolRegistry, ToolCategory
from ..security_context import get_security_context


# Store the working directory at import time for boundary checks
_initial_working_dir = os.path.abspath(os.getcwd())


def is_within_working_dir(path: str, working_dir: Optional[str] = None) -> Tuple[bool, str]:
    """Check if a path is within the working directory.

    Args:
        path: Path to check
        working_dir: Working directory to check against (defaults to initial cwd)

    Returns:
        Tuple of (is_within, reason_message)
    """
    base_dir = working_dir or _initial_working_dir
    base_dir = os.path.abspath(base_dir)

    # Resolve the path
    abs_path = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))

    # Check if path is within working directory
    try:
        # Use os.path.commonpath for accurate check
        common = os.path.commonpath([abs_path, base_dir])
        is_within = common == base_dir
    except ValueError:
        # Different drives on Windows
        is_within = False

    if is_within:
        return True, f"Path is within working directory: {base_dir}"
    else:
        return False, f"WARNING: Path '{abs_path}' is OUTSIDE working directory '{base_dir}'"


def check_path_with_security(path: str) -> Tuple[bool, Optional[str], bool]:
    """Check path access using session security context.

    Args:
        path: Path to check

    Returns:
        Tuple of (should_proceed, warning_message, requires_confirmation)
        - should_proceed: Whether operation should continue
        - warning_message: Warning to display (or None)
        - requires_confirmation: Whether user confirmation is needed
    """
    try:
        security_ctx = get_security_context()
        return security_ctx.check_path_access(path)
    except Exception:
        # Fallback to simple boundary check if security context unavailable
        is_within, message = is_within_working_dir(path)
        if is_within:
            return True, None, False
        else:
            return True, message, False  # DEFAULT mode behavior: warn but proceed


def get_filesystem_tools_registry() -> ToolRegistry:
    """Create and return a registry with filesystem tools registered.

    Returns:
        ToolRegistry with filesystem tools
    """
    registry = ToolRegistry()

    @registry.register(
        category=ToolCategory.FILESYSTEM,
        description=(
            "RESOLVE PATH: Convert a KNOWN path to its FULL ABSOLUTE path.\n\n"
            "USE WHEN:\n"
            "- You already have an EXACT path (e.g., '../file.py', '~/docs', './src')\n"
            "- Converting relative paths to absolute paths\n"
            "- Resolving ~, .., environment variables, or symlinks\n"
            "- Verifying a path exists before file operations\n\n"
            "DO NOT USE WHEN:\n"
            "- You need to SEARCH/FIND files (use find_paths instead)\n"
            "- You don't know the exact path name\n"
            "- You want to list directory contents (use list_directory)\n\n"
            "SECURITY: Respects session security mode (/security). "
            "STRICT mode blocks access until confirmed. TRUSTED mode allows all. "
            "DEFAULT mode warns once per outside path."
        ),
        parameters={
            "path": {
                "type": "string",
                "required": True,
                "description": "The path to resolve (can be relative, contain ~, .., etc.)"
            },
            "base_dir": {
                "type": "string",
                "required": False,
                "description": "Base directory for relative paths (defaults to current working directory)"
            },
            "check_boundary": {
                "type": "boolean",
                "required": False,
                "description": "Check if path is within working directory (default: True)"
            }
        },
        tags=["path", "filesystem", "resolve", "absolute"],
        examples=[
            "resolve_path('../hybridrag') -> {'path': '/home/user/Documents/code/hybridrag', 'outside_working_dir': True, 'warning': '...'}",
            "resolve_path('~/Documents') -> {'path': '/home/user/Documents', 'outside_working_dir': True}",
            "resolve_path('./src') -> {'path': '/home/user/project/src', 'outside_working_dir': False}"
        ]
    )
    def resolve_path(
        path: str,
        base_dir: Optional[str] = None,
        check_boundary: bool = True
    ) -> dict:
        """Resolve a path to its full absolute path with boundary checking.

        Args:
            path: Path to resolve (can be relative, contain ~, .., etc.)
            base_dir: Base directory for relative paths
            check_boundary: Whether to check if path is within working directory

        Returns:
            Dictionary with 'path', 'exists', 'outside_working_dir', and optional 'warning'
        """
        # Expand user home directory (~)
        expanded = os.path.expanduser(path)

        # Expand environment variables
        expanded = os.path.expandvars(expanded)

        # Handle base directory for relative paths
        if not os.path.isabs(expanded):
            if base_dir:
                base = os.path.expanduser(os.path.expandvars(base_dir))
                expanded = os.path.join(base, expanded)
            else:
                expanded = os.path.join(os.getcwd(), expanded)

        # Normalize and resolve to absolute path
        absolute = os.path.abspath(os.path.normpath(expanded))

        # Try to resolve symlinks if path exists
        if os.path.exists(absolute):
            try:
                absolute = os.path.realpath(absolute)
            except OSError:
                pass  # Keep the normalized path if realpath fails

        result = {
            "path": absolute,
            "exists": os.path.exists(absolute),
            "is_file": os.path.isfile(absolute),
            "is_directory": os.path.isdir(absolute),
        }

        # Check working directory boundary using security context
        if check_boundary:
            should_proceed, warning_msg, requires_confirmation = check_path_with_security(absolute)
            is_within, _ = is_within_working_dir(absolute)
            result["outside_working_dir"] = not is_within

            if warning_msg:
                result["warning"] = warning_msg

            if requires_confirmation:
                result["requires_confirmation"] = True
                result["warning"] = (
                    f"STRICT MODE: Path '{absolute}' is outside working directory.\n"
                    f"Working directory: {_initial_working_dir}\n"
                    "User confirmation required before accessing this path."
                )

            if not should_proceed:
                result["access_denied"] = True

        return result

    @registry.register(
        category=ToolCategory.FILESYSTEM,
        description=(
            "FIND PATHS: SEARCH for files or directories by name pattern.\n\n"
            "USE WHEN:\n"
            "- You need to LOCATE/FIND something (don't know exact path)\n"
            "- Searching for files matching a pattern (*.py, test_*.py)\n"
            "- Finding files in subdirectories recursively\n"
            "- Case-insensitive search needed (finds 'HybridRag' with 'hybridrag*')\n\n"
            "DO NOT USE WHEN:\n"
            "- You already KNOW the exact path (use resolve_path instead)\n"
            "- Searching file CONTENTS (use ripgrep_search instead)\n"
            "- Just listing a directory (use list_directory instead)\n\n"
            "SECURITY: Respects session security mode (/security). "
            "In STRICT mode, outside paths are blocked if previously denied. "
            "Returns paths grouped by inside/outside working directory."
        ),
        parameters={
            "pattern": {
                "type": "string",
                "required": True,
                "description": "Glob pattern to match (e.g., '*.py', '**/test_*.py', 'hybridrag*'). Use * for wildcards."
            },
            "search_dir": {
                "type": "string",
                "required": False,
                "description": "Directory to search in (defaults to current working directory)"
            },
            "recursive": {
                "type": "boolean",
                "required": False,
                "description": "Search recursively in subdirectories (default: True)"
            },
            "include_parent": {
                "type": "boolean",
                "required": False,
                "description": "Also search parent directories (default: False). When True, only searches shallow (1 level) in parent to avoid hanging."
            },
            "case_insensitive": {
                "type": "boolean",
                "required": False,
                "description": "Case-insensitive matching (default: True). Finds 'HybridRag' when searching 'hybridrag*'."
            },
            "max_results": {
                "type": "integer",
                "required": False,
                "description": "Maximum number of results to return (default: 100). Prevents hanging on large directories."
            },
            "max_depth": {
                "type": "integer",
                "required": False,
                "description": "Maximum directory depth to search (default: 5). Set to 1 for shallow search."
            }
        },
        tags=["find", "glob", "search", "filesystem", "absolute"],
        examples=[
            "find_paths('hybridrag*', '/home/user/Documents') -> {'paths': [...], 'outside_working_dir': True}",
            "find_paths('*.py', recursive=False) -> {'paths': [...], 'count': 5}"
        ]
    )
    def find_paths(
        pattern: str,
        search_dir: Optional[str] = None,
        recursive: bool = True,
        include_parent: bool = False,
        case_insensitive: bool = True,
        max_results: int = 100,
        max_depth: int = 5
    ) -> dict:
        """Find files/directories matching pattern and return absolute paths.

        Args:
            pattern: Glob pattern to match
            search_dir: Directory to search in
            recursive: Whether to search recursively
            include_parent: Also search in parent directories (shallow only)
            case_insensitive: Case-insensitive matching (default True)
            max_results: Maximum results to return (prevents hanging)
            max_depth: Maximum directory depth to search

        Returns:
            Dictionary with 'paths', 'count', and boundary check info
        """
        import fnmatch
        import re

        all_paths = []
        truncated = False

        # Convert glob pattern to regex for case-insensitive matching
        def pattern_matches(name: str, pat: str, case_insensitive: bool) -> bool:
            """Check if name matches pattern, optionally case-insensitive."""
            if case_insensitive:
                return fnmatch.fnmatch(name.lower(), pat.lower())
            return fnmatch.fnmatch(name, pat)

        # Determine search directories
        search_dirs = []

        if search_dir:
            base = os.path.expanduser(os.path.expandvars(search_dir))
            base = os.path.abspath(base)
            search_dirs.append((base, recursive, max_depth))
        else:
            base = os.getcwd()
            search_dirs.append((base, recursive, max_depth))

        # Add parent directories if requested - SHALLOW ONLY to prevent hanging
        if include_parent:
            parent = os.path.dirname(base)
            if parent and parent != base:
                # Only search 1 level deep in parent to avoid massive traversal
                search_dirs.append((parent, False, 1))  # Non-recursive, depth 1

        def depth_limited_search(base_path: str, pat: str, do_recursive: bool, depth: int) -> List[str]:
            """Search with depth limiting and case-insensitive matching."""
            results = []

            if depth <= 0:
                return results

            try:
                entries = os.listdir(base_path)
            except PermissionError:
                return results

            # Check each entry against pattern
            for entry in entries:
                if len(results) >= max_results:
                    break

                full_path = os.path.join(base_path, entry)

                # Check if entry matches pattern (case-insensitive if enabled)
                if pattern_matches(entry, pat, case_insensitive):
                    results.append(full_path)

                # Recurse into subdirectories if enabled
                if do_recursive and depth > 1 and os.path.isdir(full_path):
                    if not entry.startswith('.'):  # Skip hidden dirs
                        sub_results = depth_limited_search(full_path, pat, True, depth - 1)
                        results.extend(sub_results)

            return results[:max_results]

        for search_base, do_recursive, depth in search_dirs:
            if len(all_paths) >= max_results:
                truncated = True
                break

            matches = depth_limited_search(search_base, pattern, do_recursive, depth)
            all_paths.extend(matches)

        # Limit total results
        if len(all_paths) > max_results:
            all_paths = all_paths[:max_results]
            truncated = True

        # Convert all to absolute paths
        absolute_paths = [os.path.abspath(os.path.realpath(p)) for p in all_paths]

        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for p in absolute_paths:
            if p not in seen:
                seen.add(p)
                unique_paths.append(p)

        unique_paths = sorted(unique_paths)

        # Check for paths outside working directory using security context
        outside_paths = []
        inside_paths = []
        blocked_paths = []

        for p in unique_paths:
            is_within, _ = is_within_working_dir(p)
            if is_within:
                inside_paths.append(p)
            else:
                # Check with security context
                should_proceed, warning_msg, requires_confirmation = check_path_with_security(p)
                if not should_proceed:
                    blocked_paths.append(p)
                else:
                    outside_paths.append(p)

        result = {
            "paths": unique_paths,
            "count": len(unique_paths),
            "inside_working_dir": inside_paths,
            "outside_working_dir": outside_paths,
            "truncated": truncated,
        }

        if truncated:
            result["note"] = f"Results limited to {max_results}. Increase max_results for more."

        if blocked_paths:
            result["blocked_paths"] = blocked_paths
            result["warning"] = (
                f"STRICT MODE: {len(blocked_paths)} path(s) blocked (previously denied):\n"
                f"  {blocked_paths}\n"
                "Use /security default or /security trusted to allow access."
            )
        elif outside_paths:
            # Get current security mode for context-appropriate message
            try:
                security_ctx = get_security_context()
                mode = security_ctx.mode.value.upper()
            except Exception:
                mode = "DEFAULT"

            result["warning"] = (
                f"Found {len(outside_paths)} path(s) OUTSIDE working directory ({mode} mode):\n"
                f"  Working dir: {_initial_working_dir}\n"
                f"  Outside paths: {outside_paths}"
            )

        return result

    @registry.register(
        category=ToolCategory.FILESYSTEM,
        description=(
            "GET CWD: Get the current working directory as a FULL ABSOLUTE path.\n\n"
            "USE WHEN:\n"
            "- Understanding your current location in the filesystem\n"
            "- Before constructing relative paths\n"
            "- Verifying which project/directory is active\n"
            "- Setting context for path-related operations\n\n"
            "DO NOT USE WHEN:\n"
            "- You need to list files (use list_directory)\n"
            "- You need to search for files (use find_paths)\n\n"
            "SECURITY: Always returns the session working directory. "
            "This is the reference point for boundary checking."
        ),
        parameters={},
        tags=["cwd", "pwd", "current", "directory", "absolute"]
    )
    def get_cwd() -> str:
        """Get current working directory as absolute path.

        Returns:
            Full absolute path of current working directory
        """
        return os.path.abspath(os.getcwd())

    @registry.register(
        category=ToolCategory.FILESYSTEM,
        description=(
            "PATH INFO: Get detailed info about a path (existence, type, size).\n\n"
            "USE WHEN:\n"
            "- Checking if a file/directory exists before operations\n"
            "- Determining if a path is a file, directory, or symlink\n"
            "- Getting file size before reading large files\n"
            "- Verifying path type before file operations\n\n"
            "DO NOT USE WHEN:\n"
            "- You just need to convert to absolute path (use resolve_path)\n"
            "- You need to list directory contents (use list_directory)\n"
            "- You need to search for files (use find_paths)\n\n"
            "SECURITY: Returns path info only. Does not read file contents. "
            "Safe to use for path validation."
        ),
        parameters={
            "path": {
                "type": "string",
                "required": True,
                "description": "Path to check"
            }
        },
        tags=["exists", "check", "info", "filesystem", "absolute"]
    )
    def path_info(path: str) -> dict:
        """Get information about a path including its absolute form.

        Args:
            path: Path to check

        Returns:
            Dictionary with path information
        """
        # Resolve to absolute path
        expanded = os.path.expanduser(os.path.expandvars(path))
        if not os.path.isabs(expanded):
            expanded = os.path.join(os.getcwd(), expanded)
        absolute = os.path.abspath(os.path.normpath(expanded))

        exists = os.path.exists(absolute)

        info = {
            "original_path": path,
            "absolute_path": absolute,
            "exists": exists,
            "is_file": os.path.isfile(absolute) if exists else False,
            "is_directory": os.path.isdir(absolute) if exists else False,
            "is_symlink": os.path.islink(absolute) if exists else False,
        }

        if exists:
            try:
                stat = os.stat(absolute)
                info["size_bytes"] = stat.st_size
                info["real_path"] = os.path.realpath(absolute)
            except OSError:
                pass

        return info

    return registry


# Create a default instance with all tools registered
filesystem_registry = get_filesystem_tools_registry()

# Export individual tool functions for direct use
resolve_path = filesystem_registry.get("resolve_path").function
find_paths = filesystem_registry.get("find_paths").function
get_cwd = filesystem_registry.get("get_cwd").function
path_info = filesystem_registry.get("path_info").function

__all__ = [
    "filesystem_registry",
    "get_filesystem_tools_registry",
    "resolve_path",
    "find_paths",
    "get_cwd",
    "path_info"
]
