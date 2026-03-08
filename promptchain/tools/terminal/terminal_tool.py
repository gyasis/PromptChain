"""
Terminal Execution Tool for PromptChain - Main Module

A secure, modular tool for executing terminal commands within PromptChain workflows.
Supports all PromptChain patterns: simple chains, function injection, AgenticStepProcessor, and multi-agent systems.

Features:
- Comprehensive security guardrails with permission system
- Intelligent environment detection and activation
- Multi-tier executable path resolution
- Complete integration with PromptChain ecosystem
- Extensive logging and error handling

Author: PromptChain Team
License: MIT
"""

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml  # type: ignore[import-untyped]

from .environment import EnvironmentManager
from .path_resolver import PathResolver
from .security import SecurityGuardrails
from .session_manager import PersistentTerminalSession, SessionManager
from .simple_persistent_session import (SimplePersistentSession,
                                        SimpleSessionManager)

# Optional visual formatter import
try:
    from .visual_formatter import VisualTerminalFormatter

    VISUAL_AVAILABLE = True
except ImportError:
    VISUAL_AVAILABLE = False
    VisualTerminalFormatter = None  # type: ignore[assignment,misc]


@dataclass
class CommandResult:
    """Result of a terminal command execution"""

    stdout: str
    stderr: str
    return_code: int
    command: str
    execution_time: float
    timed_out: bool = False
    blocked: bool = False
    blocked_reason: str = ""


class SecurityError(Exception):
    """Raised when a command is blocked for security reasons"""

    pass


class TerminalTool:
    """
    A secure terminal execution tool for PromptChain.

    This tool provides safe command execution with security guardrails,
    environment management, and comprehensive logging. It works with all
    PromptChain patterns including simple chains, function injection,
    AgenticStepProcessor, and multi-agent systems.

    Usage Examples:
        # Simple usage
        terminal = TerminalTool()
        result = terminal("ls -la")

        # Function injection
        def run_command(cmd: str) -> str:
            return terminal(cmd)

        # Direct callable in chain
        chain = PromptChain(instructions=[terminal])

        # AgenticStepProcessor integration
        chain.register_tool_function(terminal)
    """

    def __init__(
        self,
        timeout: int = 60,
        working_directory: Optional[str] = None,
        environment_variables: Optional[Dict[str, str]] = None,
        config_file: Optional[str] = None,
        require_permission: bool = True,
        verbose: bool = False,
        log_all_commands: bool = True,
        auto_activate_env: bool = False,
        logger: Optional[logging.Logger] = None,
        use_persistent_session: bool = True,
        session_name: Optional[str] = None,
        max_sessions: int = 10,
        visual_debug: bool = False,
    ):
        """
        Initialize the TerminalTool

        Args:
            timeout: Maximum execution time in seconds
            working_directory: Working directory for commands
            environment_variables: Additional environment variables
            config_file: Path to configuration file (YAML or JSON)
            require_permission: Whether to require permission for risky commands
            verbose: Enable verbose output
            log_all_commands: Log all executed commands
            auto_activate_env: Automatically activate detected environments
            logger: Optional logger instance
            use_persistent_session: Enable persistent terminal sessions
            session_name: Name for the terminal session (auto-generated if None)
            max_sessions: Maximum number of concurrent sessions
            visual_debug: Enable visual terminal debugging with Rich formatting
        """
        self.timeout = timeout
        self.working_directory = working_directory or os.getcwd()
        self.environment_variables = environment_variables or {}
        self.verbose = verbose
        self.log_all_commands = log_all_commands
        self.auto_activate_env = auto_activate_env
        self.use_persistent_session = use_persistent_session
        self.session_name = session_name or f"terminal_{int(time.time())}"
        self.visual_debug = visual_debug

        # Initialize visual formatter if requested and available
        self.visual_formatter = None
        if visual_debug and VISUAL_AVAILABLE:
            self.visual_formatter = VisualTerminalFormatter()  # type: ignore[operator]
            if verbose:
                print("✨ Visual terminal debugging enabled!")
        elif visual_debug and not VISUAL_AVAILABLE:
            print(
                "⚠️ Visual debugging requested but Rich is not available. Install with: pip install rich"
            )

        # Load configuration
        self.config = self._load_config(config_file) if config_file else {}
        self.config.update(
            {
                "require_permission": require_permission,
                "log_all_commands": log_all_commands,
                "timeout": timeout,
                "verbose": verbose,
            }
        )

        # Setup logging first
        self.logger = logger or self._setup_logger()

        # Initialize components
        self.security = SecurityGuardrails(self.config)
        self.env_manager = EnvironmentManager(self.config)
        self.path_resolver = PathResolver(self.config)

        # Initialize session management with simplified approach
        self.session_manager = (
            SimpleSessionManager(max_sessions=max_sessions)
            if use_persistent_session
            else None
        )
        self.current_session_id = None

        # Create initial session if persistent sessions are enabled
        if self.use_persistent_session and self.session_manager is not None:
            try:
                self.current_session_id = self.session_manager.create_session(
                    name=self.session_name,
                    working_directory=self.working_directory,
                    environment_variables=self.environment_variables,
                )
                if self.verbose:
                    self.logger.info(
                        f"Created simplified persistent session: {self.session_name}"
                    )
            except Exception as e:
                self.logger.warning(f"Failed to create persistent session: {e}")
                self.use_persistent_session = False

        # Command history for debugging and analysis
        self.command_history: List[CommandResult] = []
        self.last_result: Optional[CommandResult] = None

        # Initialize environment if requested
        if self.auto_activate_env:
            self._initialize_environment()

    def _setup_logger(self) -> logging.Logger:
        """Setup default logger"""
        logger = logging.getLogger("terminal_tool")
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from file"""
        try:
            path = Path(config_file)
            if not path.exists():
                self.logger.warning(f"Config file not found: {config_file}")
                return {}

            with open(path, "r") as f:
                if path.suffix in [".yaml", ".yml"]:
                    return yaml.safe_load(f) or {}
                elif path.suffix == ".json":
                    return json.load(f)
                else:
                    self.logger.warning(f"Unsupported config format: {path.suffix}")
                    return {}
        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}")
            return {}

    def _initialize_environment(self):
        """Initialize and activate development environment"""
        try:
            detected = self.env_manager.detect_environments()
            if detected:
                self.logger.info(f"Detected environments: {list(detected.keys())}")

                # Try to auto-activate best environment
                activated = self.env_manager.auto_detect_and_activate()
                if activated:
                    env_type, env_name = activated
                    self.logger.info(
                        f"Auto-activated environment: {env_type}:{env_name}"
                    )
                else:
                    self.logger.info(
                        "No suitable environment found for auto-activation"
                    )
            else:
                self.logger.info("No development environments detected")
        except Exception as e:
            self.logger.warning(f"Environment initialization failed: {e}")

    def execute(self, command: str) -> str:
        """
        Execute a terminal command with full security and environment management

        Args:
            command: The command to execute

        Returns:
            String output (stdout + stderr) from the command

        Raises:
            SecurityError: If command is blocked
            TimeoutError: If command times out
            RuntimeError: For other execution errors
        """
        start_time = time.time()

        if not command or not command.strip():
            return ""

        # Log command attempt
        if self.log_all_commands:
            self.logger.info(f"Executing: {command}")

        # Security validation
        is_safe, risk_level, reason = self.security.check_command(command)

        if risk_level == "blocked":
            error_msg = f"🚫 Command blocked for security: {reason}"
            self.logger.error(error_msg)
            result = CommandResult(
                stdout="",
                stderr=error_msg,
                return_code=-1,
                command=command,
                execution_time=time.time() - start_time,
                blocked=True,
                blocked_reason=reason,
            )
            self.last_result = result
            self.command_history.append(result)
            raise SecurityError(error_msg)

        # Request permission for risky commands
        if risk_level in ["dangerous", "caution"]:
            if not self.security.request_permission(command, risk_level, reason):
                error_msg = "❌ Permission denied by user"
                self.logger.warning(error_msg)
                result = CommandResult(
                    stdout="",
                    stderr=error_msg,
                    return_code=-1,
                    command=command,
                    execution_time=time.time() - start_time,
                    blocked=True,
                    blocked_reason="User denied permission",
                )
                self.last_result = result
                self.command_history.append(result)
                return error_msg

        # Use persistent session if enabled
        if self.use_persistent_session and self.session_manager:
            return self._execute_in_persistent_session(command, start_time)
        else:
            return self._execute_in_subprocess(command, start_time)

    def _execute_in_persistent_session(self, command: str, start_time: float) -> str:
        """Execute command in a persistent session"""
        try:
            assert self.session_manager is not None
            session = self.session_manager.get_current_session()
            if not session:
                raise RuntimeError("No active persistent session")

            # Capture working directory BEFORE executing command
            pre_execution_dir = session.working_directory or "/tmp"

            # Resolve command path if needed
            resolved_command = self._resolve_command_path(command)

            if self.verbose:
                self.logger.debug(
                    f"Executing in persistent session '{session.name}': {resolved_command}"
                )

            # Execute in persistent session
            output, return_code = session.execute_command(
                resolved_command, self.timeout
            )

            # Create result record
            result = CommandResult(
                stdout=output,
                stderr="",
                return_code=return_code,
                command=command,
                execution_time=time.time() - start_time,
                timed_out=False,
            )

            self.last_result = result
            self.command_history.append(result)

            # Add to visual formatter if enabled - use PRE-EXECUTION directory
            if self.visual_formatter:
                try:
                    self.visual_formatter.add_command(
                        command=command,
                        output=output,
                        working_dir=pre_execution_dir,  # Use directory from BEFORE command
                        error=(return_code != 0),
                    )
                    # ✅ ALWAYS show visual output when visual_debug is enabled (not controlled by verbose)
                    self.visual_formatter.print_last_command()
                except Exception as ve:
                    # Don't let visual formatting break the actual functionality
                    if self.verbose:
                        print(f"Visual formatting warning: {ve}")

            # Log execution details
            if self.verbose:
                self.logger.debug(f"Completed in {result.execution_time:.2f}s")
                self.logger.debug(f"Return code: {return_code}")
                if output:
                    self.logger.debug(f"OUTPUT: {output[:200]}...")

            return output

        except Exception as e:
            if isinstance(e, (TimeoutError, SecurityError)):
                raise

            error_msg = f"💥 Persistent session execution failed: {str(e)}"
            self.logger.error(error_msg)

            # Record failed execution
            result = CommandResult(
                stdout="",
                stderr=error_msg,
                return_code=-1,
                command=command,
                execution_time=time.time() - start_time,
            )
            self.last_result = result
            self.command_history.append(result)

            raise RuntimeError(error_msg)

    def _execute_in_subprocess(self, command: str, start_time: float) -> str:
        """Execute command in a new subprocess (legacy behavior)"""
        # Resolve executable path if needed
        resolved_command = self._resolve_command_path(command)

        # Prepare execution environment
        env = os.environ.copy()
        env.update(self.environment_variables)

        # Execute command
        try:
            if self.verbose:
                self.logger.debug(f"Resolved command: {resolved_command}")
                self.logger.debug(f"Working directory: {self.working_directory}")

            process = subprocess.Popen(
                resolved_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.working_directory,
                env=env,
            )

            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                return_code = process.returncode
                timed_out = False

            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                return_code = -1
                stderr = f"⏰ Command timed out after {self.timeout}s\n{stderr}"
                timed_out = True

            # Create result record
            result = CommandResult(
                stdout=stdout,
                stderr=stderr,
                return_code=return_code,
                command=command,
                execution_time=time.time() - start_time,
                timed_out=timed_out,
            )

            self.last_result = result
            self.command_history.append(result)

            # Log execution details
            if self.verbose:
                self.logger.debug(f"Completed in {result.execution_time:.2f}s")
                self.logger.debug(f"Return code: {return_code}")
                if stdout:
                    self.logger.debug(f"STDOUT: {stdout[:200]}...")
                if stderr:
                    self.logger.debug(f"STDERR: {stderr[:200]}...")

            # Format output for return
            output_parts = []
            if stdout.strip():
                output_parts.append(stdout.strip())
            if stderr.strip():
                output_parts.append(f"[STDERR]: {stderr.strip()}")

            output = "\n".join(output_parts) if output_parts else ""

            # Handle errors
            if timed_out:
                raise TimeoutError(
                    f"Command timed out after {self.timeout}s: {command}"
                )

            if return_code != 0 and stderr.strip():
                self.logger.warning(f"Command failed (code {return_code}): {stderr}")

            return output

        except Exception as e:
            if isinstance(e, (TimeoutError, SecurityError)):
                raise

            error_msg = f"💥 Command execution failed: {str(e)}"
            self.logger.error(error_msg)

            # Record failed execution
            result = CommandResult(
                stdout="",
                stderr=error_msg,
                return_code=-1,
                command=command,
                execution_time=time.time() - start_time,
            )
            self.last_result = result
            self.command_history.append(result)

            raise RuntimeError(error_msg)

    def _resolve_command_path(self, command: str) -> str:
        """Resolve command path using path resolver"""
        try:
            # Extract base command
            base_cmd = command.split()[0] if " " in command else command
            resolved_path = self.path_resolver.resolve_executable(base_cmd)

            if resolved_path and resolved_path != base_cmd:
                # Replace base command with resolved path
                parts = command.split()
                parts[0] = resolved_path
                return " ".join(parts)

        except Exception as e:
            self.logger.warning(f"Path resolution failed: {e}")

        return command

    def __call__(self, command: str) -> str:
        """
        Make the tool callable for direct use in PromptChain

        This method enables the following usage patterns:
        - terminal = TerminalTool(); result = terminal("ls")
        - chain = PromptChain(instructions=[terminal])
        - chain.register_tool_function(terminal)

        Args:
            command: The command to execute

        Returns:
            String output from the command
        """
        return self.execute(command)

    # ===== Advanced Features =====

    def execute_with_parsing(
        self, command: str, parser: str = "auto"
    ) -> Dict[str, Any]:
        """
        Execute command and return structured output

        Args:
            command: Command to execute
            parser: Parsing strategy ('json', 'csv', 'table', 'auto')

        Returns:
            Dictionary with parsed output
        """
        output = self.execute(command)

        if parser == "json":
            try:
                return {"type": "json", "data": json.loads(output)}
            except json.JSONDecodeError as e:
                return {
                    "type": "json",
                    "data": None,
                    "parse_error": str(e),
                    "raw": output,
                }

        elif parser == "csv":
            lines = output.strip().split("\n")
            if lines and "," in lines[0]:
                headers = [h.strip() for h in lines[0].split(",")]
                rows = []
                for line in lines[1:]:
                    rows.append([cell.strip() for cell in line.split(",")])
                return {"type": "csv", "data": {"headers": headers, "rows": rows}}
            return {
                "type": "csv",
                "data": None,
                "parse_error": "No CSV data found",
                "raw": output,
            }

        elif parser == "table":
            lines = [line for line in output.strip().split("\n") if line.strip()]
            return {"type": "table", "data": lines}

        elif parser == "auto":
            # Try JSON first
            try:
                return {"type": "json", "data": json.loads(output)}
            except:
                pass

            # Check for CSV-like data
            if "," in output and "\n" in output:
                lines = output.strip().split("\n")
                if len(lines) > 1 and all("," in line for line in lines[:3]):
                    return self.execute_with_parsing(command, "csv")

            # Default to raw text
            return {"type": "raw", "data": output}

        return {"type": "raw", "data": output}

    def execute_with_success_check(self, command: str) -> Dict[str, Any]:
        """
        Execute command and return detailed success information

        Args:
            command: Command to execute

        Returns:
            Dictionary with execution details
        """
        try:
            output = self.execute(command)
            return {
                "success": True,
                "output": output,
                "return_code": self.last_result.return_code if self.last_result else 0,
                "execution_time": (
                    self.last_result.execution_time if self.last_result else 0
                ),
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "return_code": self.last_result.return_code if self.last_result else -1,
                "execution_time": (
                    self.last_result.execution_time if self.last_result else 0
                ),
                "error": str(e),
            }

    # ===== Utility Methods for AgenticStepProcessor =====

    def check_installation(self, package: str) -> bool:
        """
        Check if a package/program is installed

        Args:
            package: Name of the package/program

        Returns:
            True if installed, False otherwise
        """
        checks = [
            f"which {package}",
            f"command -v {package}",
            f"{package} --version",
            f"{package} -v",
        ]

        for check in checks:
            try:
                self.execute(check)
                if self.last_result and self.last_result.return_code == 0:
                    return True
            except:
                continue

        return False

    def get_version(self, program: str) -> Optional[str]:
        """
        Get version of an installed program

        Args:
            program: Program name

        Returns:
            Version string if found, None otherwise
        """
        version_cmds = [
            f"{program} --version",
            f"{program} -v",
            f"{program} version",
            f"{program} -V",
        ]

        for cmd in version_cmds:
            try:
                output = self.execute(cmd)
                if (
                    output
                    and self.last_result is not None
                    and self.last_result.return_code == 0
                ):
                    # Extract version using regex
                    import re

                    version_patterns = [
                        r"(\d+\.\d+\.\d+)",  # x.y.z
                        r"(\d+\.\d+)",  # x.y
                        r"v(\d+\.\d+\.\d+)",  # vx.y.z
                        r"version\s+(\d+\.\d+\.\d+)",  # version x.y.z
                    ]

                    for pattern in version_patterns:
                        match = re.search(pattern, output, re.IGNORECASE)
                        if match:
                            return match.group(1)

                    # Return first line if no pattern matches
                    return output.split("\n")[0].strip()
            except:
                continue

        return None

    def find_files(self, pattern: str, directory: str = ".") -> List[str]:
        """
        Find files matching a pattern

        Args:
            pattern: File pattern to search for
            directory: Directory to search in

        Returns:
            List of matching file paths
        """
        try:
            # Use find command on Unix-like systems
            if os.name != "nt":
                cmd = f"find {directory} -name '{pattern}' -type f"
            else:
                # Use dir command on Windows
                cmd = f"dir /s /b {directory}\\{pattern}"

            output = self.execute(cmd)
            if (
                output
                and self.last_result is not None
                and self.last_result.return_code == 0
            ):
                return [line.strip() for line in output.split("\n") if line.strip()]
        except:
            pass

        return []

    def check_port(self, port: int) -> bool:
        """
        Check if a port is in use

        Args:
            port: Port number to check

        Returns:
            True if port is in use, False otherwise
        """
        try:
            if os.name != "nt":
                # Unix-like systems
                cmd = f"lsof -i :{port}"
            else:
                # Windows
                cmd = f"netstat -an | findstr :{port}"

            self.execute(cmd)
            return self.last_result is not None and self.last_result.return_code == 0
        except:
            return False

    # ===== Information and Diagnostics =====

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        info = {
            "platform": os.name,
            "working_directory": self.working_directory,
            "timeout": self.timeout,
            "environment_variables": dict(self.environment_variables),
            "active_environment": self.env_manager.get_active_environment(),
            "detected_environments": self.env_manager.detected_envs,
            "command_history_length": len(self.command_history),
            "cache_stats": self.path_resolver.get_cache_stats(),
        }

        return info

    def get_command_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get command execution history

        Args:
            limit: Maximum number of commands to return

        Returns:
            List of command execution records
        """
        history = self.command_history
        if limit:
            history = history[-limit:]

        return [
            {
                "command": cmd.command,
                "return_code": cmd.return_code,
                "execution_time": cmd.execution_time,
                "stdout_length": len(cmd.stdout),
                "stderr_length": len(cmd.stderr),
                "timed_out": cmd.timed_out,
                "blocked": cmd.blocked,
                "blocked_reason": cmd.blocked_reason,
            }
            for cmd in history
        ]

    def diagnose_command(self, command: str) -> str:
        """
        Get diagnostic information about a command

        Args:
            command: Command to diagnose

        Returns:
            Diagnostic report string
        """
        return self.path_resolver.diagnose_command(command)

    def clear_history(self):
        """Clear command execution history"""
        self.command_history.clear()
        self.last_result = None

    def reset_caches(self):
        """Reset all internal caches"""
        self.path_resolver.clear_cache()
        self.clear_history()

    # ===== Session Management Methods =====

    def create_session(
        self,
        name: str,
        working_directory: Optional[str] = None,
        environment_variables: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Create a new named terminal session

        Args:
            name: Name for the session
            working_directory: Initial working directory for the session
            environment_variables: Initial environment variables

        Returns:
            Session ID
        """
        if not self.session_manager:
            raise RuntimeError("Persistent sessions are not enabled")

        return self.session_manager.create_session(
            name=name,
            working_directory=working_directory,
            environment_variables=environment_variables,
        )

    def switch_session(self, identifier: str) -> bool:
        """
        Switch to a different session

        Args:
            identifier: Session ID or name

        Returns:
            True if successful
        """
        if not self.session_manager:
            raise RuntimeError("Persistent sessions are not enabled")

        success = self.session_manager.switch_session(identifier)
        if success:
            self.current_session_id = self.session_manager.current_session_id
            if self.verbose:
                session = self.session_manager.get_current_session()
                if session is not None:
                    self.logger.info(f"Switched to session: {session.name}")
        return success

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all terminal sessions"""
        if not self.session_manager:
            return []

        return self.session_manager.list_sessions()

    def close_session(self, identifier: str) -> bool:
        """
        Close a terminal session

        Args:
            identifier: Session ID or name

        Returns:
            True if closed successfully
        """
        if not self.session_manager:
            return False

        return self.session_manager.close_session(identifier)

    def get_current_session_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current session"""
        if not self.session_manager:
            return None

        session = self.session_manager.get_current_session()
        return session.get_session_info() if session else None

    def activate_nvm_version(self, version: str) -> bool:
        """
        Activate specific NVM version in current session

        Args:
            version: NVM version (e.g., "22.9.0" or "v22.9.0")

        Returns:
            True if activation successful
        """
        if not self.use_persistent_session or not self.session_manager:
            # Fall back to regular command execution for non-persistent sessions
            try:
                output = self.execute(f"source ~/.nvm/nvm.sh && nvm use {version}")
                return "is not installed" not in output.lower()
            except:
                return False

        session = self.session_manager.get_current_session()
        if session:
            return session.activate_nvm_version(version)
        return False

    def activate_conda_environment(self, env_name: str) -> bool:
        """
        Activate conda environment in current session

        Args:
            env_name: Conda environment name

        Returns:
            True if activation successful
        """
        if not self.use_persistent_session or not self.session_manager:
            # Fall back to regular command execution for non-persistent sessions
            try:
                output = self.execute(f"conda activate {env_name}")
                return "EnvironmentNameNotFound" not in output
            except:
                return False

        session = self.session_manager.get_current_session()
        if session:
            return session.activate_conda_environment(env_name)
        return False

    def set_environment_variable(self, var_name: str, value: str):
        """
        Set environment variable in current session

        Args:
            var_name: Variable name
            value: Variable value
        """
        if not self.use_persistent_session or not self.session_manager:
            # Update local environment variables for non-persistent sessions
            self.environment_variables[var_name] = value
            return

        session = self.session_manager.get_current_session()
        if session:
            session.set_environment_variable(var_name, value)

    def get_environment_variable(self, var_name: str) -> Optional[str]:
        """
        Get environment variable from current session

        Args:
            var_name: Variable name

        Returns:
            Variable value or None
        """
        if not self.use_persistent_session or not self.session_manager:
            return self.environment_variables.get(var_name) or os.environ.get(var_name)

        session = self.session_manager.get_current_session()
        if session:
            return session.get_environment_variable(var_name)
        return None

    # ===== Visual Debugging Methods =====

    def show_visual_terminal(
        self, max_entries: int = 10, show_timestamps: bool = False
    ):
        """
        Display the visual terminal history (if visual debugging is enabled).

        Args:
            max_entries: Maximum number of recent commands to show
            show_timestamps: Whether to show timestamps
        """
        if self.visual_formatter:
            self.visual_formatter.print_terminal(max_entries, show_timestamps)
        else:
            print(
                "Visual debugging not enabled. Create TerminalTool with visual_debug=True"
            )

    def get_visual_history_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get a summary of the visual command history.

        Returns:
            Dictionary with history summary or None if visual debugging disabled
        """
        if self.visual_formatter:
            return self.visual_formatter.get_history_summary()
        return None

    def clear_visual_history(self):
        """Clear the visual terminal history."""
        if self.visual_formatter:
            self.visual_formatter.clear_history()

    def enable_visual_debug(self):
        """Enable visual debugging if Rich is available."""
        if not self.visual_debug and VISUAL_AVAILABLE:
            self.visual_debug = True
            self.visual_formatter = VisualTerminalFormatter()  # type: ignore[operator]
            if self.verbose:
                print("✨ Visual terminal debugging enabled!")
        elif not VISUAL_AVAILABLE:
            print("⚠️ Rich not available. Install with: pip install rich")

    def disable_visual_debug(self):
        """Disable visual debugging."""
        self.visual_debug = False
        self.visual_formatter = None
        if self.verbose:
            print("Visual terminal debugging disabled")

    def change_directory(self, directory: str):
        """
        Change working directory in current session

        Args:
            directory: Directory path
        """
        if not self.use_persistent_session or not self.session_manager:
            # Update local working directory for non-persistent sessions
            self.working_directory = os.path.abspath(directory)
            return

        session = self.session_manager.get_current_session()
        if session:
            session.change_directory(directory)
            self.working_directory = session.get_current_directory()

    def get_current_directory(self) -> str:
        """Get current working directory"""
        if not self.use_persistent_session or not self.session_manager:
            return self.working_directory

        session = self.session_manager.get_current_session()
        if session:
            return session.get_current_directory()
        return self.working_directory


# Convenience factory function
def create_terminal_tool(**kwargs) -> TerminalTool:
    """
    Factory function to create a TerminalTool instance with common configurations

    Args:
        **kwargs: Configuration options for TerminalTool

    Returns:
        Configured TerminalTool instance
    """
    return TerminalTool(**kwargs)


# Convenience functions for common use cases
def execute_command(command: str, **kwargs) -> str:
    """
    Quick command execution function

    Args:
        command: Command to execute
        **kwargs: TerminalTool configuration options

    Returns:
        Command output
    """
    terminal = TerminalTool(**kwargs)
    return terminal.execute(command)


def safe_execute(command: str, **kwargs) -> Dict[str, Any]:
    """
    Safe command execution with detailed results

    Args:
        command: Command to execute
        **kwargs: TerminalTool configuration options

    Returns:
        Detailed execution results
    """
    terminal = TerminalTool(**kwargs)
    return terminal.execute_with_success_check(command)
