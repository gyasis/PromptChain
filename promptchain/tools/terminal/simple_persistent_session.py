"""
Simplified Persistent Terminal Session

A reliable implementation using subprocess with shell persistence
via a wrapper approach that maintains state between commands.

Author: PromptChain Team
"""

import os
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class SimpleSessionState:
    """Simple session state tracking"""

    session_id: str
    name: str
    working_directory: str
    environment_variables: Dict[str, str]
    created_at: float
    last_accessed: float
    command_count: int = 0


class SimplePersistentSession:
    """Simplified persistent session using file-based state management"""

    def __init__(
        self,
        session_id: str,
        name: str,
        working_directory: Optional[str] = None,
        environment_variables: Optional[Dict[str, str]] = None,
    ):
        self.session_id = session_id
        self.name = name
        self.working_directory = working_directory or os.getcwd()
        self.environment_variables = environment_variables or {}
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.command_count = 0

        # Create a session state file
        self.state_dir = f"/tmp/promptchain_session_{session_id}"
        os.makedirs(self.state_dir, exist_ok=True)

        # Initialize session state
        self._save_state()

    def execute_command(self, command: str, timeout: int = 60) -> Tuple[str, int]:
        """Execute a command while maintaining session state"""
        self.last_accessed = time.time()
        self.command_count += 1

        try:
            # Create a shell script that:
            # 1. Loads the session state
            # 2. Executes the command
            # 3. Saves the new state
            script_path = f"{self.state_dir}/exec_{self.command_count}.sh"

            with open(script_path, "w") as f:
                f.write(
                    f"""#!/bin/bash
set -e

# Load session state
cd "{self.working_directory}"
"""
                )
                # Export environment variables
                for key, value in self.environment_variables.items():
                    f.write(f'export {key}="{value}"\n')

                f.write(
                    f"""
# Execute the command
{command}
COMMAND_EXIT_CODE=$?

# Save new state
echo "$(pwd)" > "{self.state_dir}/working_dir"
env | grep -E '^[A-Z_][A-Z0-9_]*=' > "{self.state_dir}/env_vars" || true

exit $COMMAND_EXIT_CODE
"""
                )

            # Make script executable
            os.chmod(script_path, 0o755)

            # Execute the script
            result = subprocess.run(
                ["/bin/bash", script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_directory,
            )

            # Load updated state
            self._load_state()

            # Clean up script
            try:
                os.remove(script_path)
            except:
                pass

            # Combine stdout and stderr if there are errors
            output = result.stdout
            if result.stderr and result.returncode != 0:
                output += f"\n[STDERR]: {result.stderr}"

            return output.strip(), result.returncode

        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Command timed out after {timeout} seconds")
        except Exception as e:
            raise RuntimeError(f"Command execution failed: {e}")

    def _save_state(self):
        """Save current session state to files"""
        try:
            # Save working directory
            with open(f"{self.state_dir}/working_dir", "w") as f:
                f.write(self.working_directory)

            # Save environment variables
            with open(f"{self.state_dir}/env_vars", "w") as f:
                for key, value in self.environment_variables.items():
                    f.write(f"{key}={value}\n")

        except Exception:
            pass  # Ignore state save errors

    def _load_state(self):
        """Load session state from files"""
        try:
            # Load working directory
            wd_file = f"{self.state_dir}/working_dir"
            if os.path.exists(wd_file):
                with open(wd_file, "r") as f:
                    new_wd = f.read().strip()
                    if new_wd and os.path.isdir(new_wd):
                        self.working_directory = new_wd

            # Load environment variables
            env_file = f"{self.state_dir}/env_vars"
            if os.path.exists(env_file):
                new_env = {}
                with open(env_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if "=" in line:
                            key, value = line.split("=", 1)
                            # Only keep variables that we care about (not system vars)
                            if not key.startswith("_") and key not in [
                                "PWD",
                                "OLDPWD",
                                "SHLVL",
                            ]:
                                new_env[key] = value

                # Update our environment variables
                self.environment_variables.update(new_env)

        except Exception:
            pass  # Ignore state load errors

    def get_current_directory(self) -> str:
        """Get current working directory"""
        return self.working_directory

    def get_environment_variable(self, var_name: str) -> Optional[str]:
        """Get environment variable value"""
        return self.environment_variables.get(var_name)

    def set_environment_variable(self, var_name: str, value: str):
        """Set environment variable"""
        self.environment_variables[var_name] = value
        self._save_state()

    def change_directory(self, directory: str):
        """Change working directory"""
        abs_dir = os.path.abspath(directory)
        if os.path.isdir(abs_dir):
            self.working_directory = abs_dir
            self._save_state()
        else:
            raise RuntimeError(f"Directory does not exist: {directory}")

    def activate_nvm_version(self, version: str) -> bool:
        """Activate NVM version"""
        try:
            # Execute NVM activation command
            nvm_cmd = f"source ~/.nvm/nvm.sh && nvm use {version}"
            output, return_code = self.execute_command(nvm_cmd)
            return return_code == 0 and "is not installed" not in output.lower()
        except:
            return False

    def activate_conda_environment(self, env_name: str) -> bool:
        """Activate conda environment"""
        try:
            # Execute conda activation
            conda_cmd = f'eval "$(conda shell.bash hook)" && conda activate {env_name}'
            output, return_code = self.execute_command(conda_cmd)
            return return_code == 0
        except:
            return False

    def get_session_info(self) -> Dict[str, Any]:
        """Get session information"""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "working_directory": self.working_directory,
            "environment_variables": self.environment_variables.copy(),
            "command_history_length": self.command_count,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "shell_ready": True,
            "process_alive": True,
        }

    def close(self):
        """Clean up session"""
        try:
            import shutil

            shutil.rmtree(self.state_dir, ignore_errors=True)
        except:
            pass


class SimpleSessionManager:
    """Simplified session manager"""

    def __init__(self, max_sessions: int = 10):
        self.max_sessions = max_sessions
        self.sessions: Dict[str, SimplePersistentSession] = {}
        self.session_names: Dict[str, str] = {}
        self.current_session_id: Optional[str] = None

    def create_session(
        self,
        name: str,
        working_directory: Optional[str] = None,
        environment_variables: Optional[Dict[str, str]] = None,
    ) -> str:
        """Create new session"""
        if name in self.session_names:
            raise ValueError(f"Session '{name}' already exists")

        # Cleanup old sessions if at limit
        if len(self.sessions) >= self.max_sessions:
            oldest_id = min(
                self.sessions.keys(), key=lambda sid: self.sessions[sid].last_accessed
            )
            self.close_session(oldest_id)

        # Create session
        session_id = str(uuid.uuid4())
        session = SimplePersistentSession(
            session_id=session_id,
            name=name,
            working_directory=working_directory,
            environment_variables=environment_variables,
        )

        self.sessions[session_id] = session
        self.session_names[name] = session_id

        if not self.current_session_id:
            self.current_session_id = session_id

        return session_id

    def get_session(self, identifier: str) -> Optional[SimplePersistentSession]:
        """Get session by ID or name"""
        # Try as session ID
        if identifier in self.sessions:
            return self.sessions[identifier]

        # Try as name
        if identifier in self.session_names:
            session_id = self.session_names[identifier]
            return self.sessions.get(session_id)

        return None

    def get_current_session(self) -> Optional[SimplePersistentSession]:
        """Get current session"""
        if self.current_session_id:
            return self.sessions.get(self.current_session_id)
        return None

    def switch_session(self, identifier: str) -> bool:
        """Switch to session"""
        session = self.get_session(identifier)
        if session:
            self.current_session_id = session.session_id
            return True
        return False

    def list_sessions(self) -> list:
        """List all sessions"""
        return [
            {
                **session.get_session_info(),
                "is_current": session.session_id == self.current_session_id,
            }
            for session in self.sessions.values()
        ]

    def close_session(self, identifier: str) -> bool:
        """Close session"""
        session = self.get_session(identifier)
        if not session:
            return False

        session.close()

        # Remove tracking
        session_id = session.session_id
        session_name = session.name

        if session_id in self.sessions:
            del self.sessions[session_id]

        if session_name in self.session_names:
            del self.session_names[session_name]

        # Update current if needed
        if self.current_session_id == session_id:
            if self.sessions:
                self.current_session_id = next(iter(self.sessions.keys()))
            else:
                self.current_session_id = None

        return True

    def close_all_sessions(self):
        """Close all sessions"""
        for session in list(self.sessions.values()):
            session.close()

        self.sessions.clear()
        self.session_names.clear()
        self.current_session_id = None
