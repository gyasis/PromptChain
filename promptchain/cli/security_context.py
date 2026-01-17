"""Security context for CLI session-level security mode management.

Provides session-wide security modes for path boundary checking:
- STRICT: Always warn on boundary violations, require confirmation
- TRUSTED: No boundary warnings (user trusts the session)
- DEFAULT: Warn once per unique outside-path, then allow

Usage:
    /security strict   - Enable strict mode
    /security trusted  - Enable trusted mode
    /security default  - Enable default mode (warn once)
    /security          - Show current mode
"""

import os
from enum import Enum
from typing import Optional, Set, Tuple
from dataclasses import dataclass, field


class SecurityMode(Enum):
    """Session-wide security mode for path boundary checking."""

    STRICT = "strict"
    """Always warn on boundary violations, require confirmation before access."""

    TRUSTED = "trusted"
    """No boundary warnings - user trusts all paths in this session."""

    DEFAULT = "default"
    """Warn once per unique outside-path, then allow subsequent access."""


@dataclass
class SecurityContext:
    """Manages session-level security settings for path access.

    Attributes:
        mode: Current security mode (STRICT, TRUSTED, or DEFAULT)
        working_directory: The base working directory for boundary checks
        approved_paths: Set of paths already approved this session (DEFAULT mode)
        denied_paths: Set of paths explicitly denied this session (STRICT mode)
    """

    mode: SecurityMode = SecurityMode.DEFAULT
    working_directory: str = field(default_factory=os.getcwd)
    approved_paths: Set[str] = field(default_factory=set)
    denied_paths: Set[str] = field(default_factory=set)

    def set_mode(self, mode: SecurityMode) -> str:
        """Set the security mode and return confirmation message.

        Args:
            mode: New security mode to set

        Returns:
            Confirmation message describing the new mode
        """
        old_mode = self.mode
        self.mode = mode

        # Clear approved/denied paths when changing modes
        if old_mode != mode:
            self.approved_paths.clear()
            self.denied_paths.clear()

        mode_descriptions = {
            SecurityMode.STRICT: (
                "STRICT mode enabled. All access to paths outside working directory "
                "will require explicit confirmation. Use `/security trusted` to disable."
            ),
            SecurityMode.TRUSTED: (
                "TRUSTED mode enabled. No boundary warnings will be shown. "
                "All paths are accessible without confirmation."
            ),
            SecurityMode.DEFAULT: (
                "DEFAULT mode enabled. First access to each outside-directory path "
                "will show a warning, then allow subsequent access."
            ),
        }
        return mode_descriptions[mode]

    def is_within_working_dir(self, path: str) -> bool:
        """Check if a path is within the working directory.

        Args:
            path: Path to check (will be resolved to absolute)

        Returns:
            True if path is within working directory, False otherwise
        """
        # Resolve to absolute path
        abs_path = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
        base_dir = os.path.abspath(self.working_directory)

        try:
            common = os.path.commonpath([abs_path, base_dir])
            return common == base_dir
        except ValueError:
            # Different drives on Windows
            return False

    def check_path_access(self, path: str) -> Tuple[bool, Optional[str], bool]:
        """Check if access to a path should be allowed based on security mode.

        Args:
            path: Path to check access for

        Returns:
            Tuple of (should_proceed, warning_message, requires_confirmation)
            - should_proceed: Whether the operation should continue
            - warning_message: Warning to display (or None)
            - requires_confirmation: Whether user confirmation is needed
        """
        abs_path = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))

        # Always allow paths within working directory
        if self.is_within_working_dir(abs_path):
            return True, None, False

        # Handle based on mode
        if self.mode == SecurityMode.TRUSTED:
            return True, None, False

        if self.mode == SecurityMode.STRICT:
            # Check if already denied
            if abs_path in self.denied_paths:
                return False, f"Access to '{abs_path}' was previously denied.", False

            # Check if already approved
            if abs_path in self.approved_paths:
                return True, None, False

            # Require confirmation
            warning = (
                f"STRICT MODE: Path '{abs_path}' is outside working directory "
                f"'{self.working_directory}'. Confirmation required."
            )
            return False, warning, True

        # DEFAULT mode - warn once
        if abs_path in self.approved_paths:
            return True, None, False

        # First access - warn and auto-approve
        self.approved_paths.add(abs_path)
        warning = (
            f"Note: Accessing path outside working directory: {abs_path}\n"
            f"(Working dir: {self.working_directory})"
        )
        return True, warning, False

    def approve_path(self, path: str) -> None:
        """Explicitly approve a path for this session.

        Args:
            path: Path to approve
        """
        abs_path = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
        self.approved_paths.add(abs_path)
        self.denied_paths.discard(abs_path)

    def deny_path(self, path: str) -> None:
        """Explicitly deny a path for this session.

        Args:
            path: Path to deny
        """
        abs_path = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
        self.denied_paths.add(abs_path)
        self.approved_paths.discard(abs_path)

    def get_status(self) -> str:
        """Get current security status as a formatted string.

        Returns:
            Formatted status string
        """
        lines = [
            f"Security Mode: {self.mode.value.upper()}",
            f"Working Directory: {self.working_directory}",
        ]

        if self.approved_paths:
            lines.append(f"Approved Paths ({len(self.approved_paths)}):")
            for p in sorted(self.approved_paths):
                lines.append(f"  + {p}")

        if self.denied_paths:
            lines.append(f"Denied Paths ({len(self.denied_paths)}):")
            for p in sorted(self.denied_paths):
                lines.append(f"  - {p}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize security context to dictionary for session storage.

        Returns:
            Dictionary representation
        """
        return {
            "mode": self.mode.value,
            "working_directory": self.working_directory,
            "approved_paths": list(self.approved_paths),
            "denied_paths": list(self.denied_paths),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SecurityContext":
        """Deserialize security context from dictionary.

        Args:
            data: Dictionary from to_dict()

        Returns:
            SecurityContext instance
        """
        return cls(
            mode=SecurityMode(data.get("mode", "default")),
            working_directory=data.get("working_directory", os.getcwd()),
            approved_paths=set(data.get("approved_paths", [])),
            denied_paths=set(data.get("denied_paths", [])),
        )


# Global security context instance for the session
_security_context: Optional[SecurityContext] = None


def get_security_context() -> SecurityContext:
    """Get the global security context, creating one if needed.

    Returns:
        The global SecurityContext instance
    """
    global _security_context
    if _security_context is None:
        _security_context = SecurityContext()
    return _security_context


def set_security_context(context: SecurityContext) -> None:
    """Set the global security context.

    Args:
        context: SecurityContext to use as global
    """
    global _security_context
    _security_context = context


def reset_security_context() -> None:
    """Reset the global security context to default."""
    global _security_context
    _security_context = SecurityContext()
